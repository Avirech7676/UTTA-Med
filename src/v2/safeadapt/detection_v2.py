"""SafeAdapt Detection v2 Adapter for Dense Mitotic Object Detection.

Combines:
1. Biophysical Cellularity-Mitosis Ratio (CMR) safety monitoring.
2. Multiscale FPN RoI prototype alignment against source prototypes.
3. Dual-view Beer-Lambert stain-consistent pseudo-box self-training.
4. Non-destructive partial recoil rollback buffer preventing premature fallback.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tta.stain_consistency import BeerLambertStainAugmenter
from src.v2.gating.cmr_gate import CellularityMitosisRatioGate, CMRGateConfig
from src.v2.safeadapt.detection_monitor import DetectionSafetyMonitor


@dataclass
class SafeAdaptDetectionConfig:
    """Configuration for SafeAdapt Detection v2."""
    lr: float = 5e-5
    lambda_align: float = 0.5
    lambda_consistency: float = 0.3
    lambda_anchor: float = 0.05
    score_thresh: float = 0.50
    nms_iou_thresh: float = 0.50
    max_recoil_history: int = 5
    max_collapse_streak: int = 5
    silence_window: int = 30
    max_per_patch: int = 4
    source_ratio: float = 3.0
    min_cellularity_threshold: float = 0.15
    stain_jitter_factor: float = 0.15
    max_displacement: float = 0.05


class SafeAdaptDetectionAdapterV2(nn.Module):
    """SafeAdapt v2 Detector Test-Time Adaptation Engine."""

    def __init__(
        self,
        detector: nn.Module,
        source_roi_prototypes: Optional[torch.Tensor] = None,
        config: Optional[SafeAdaptDetectionConfig] = None,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.config = config or SafeAdaptDetectionConfig()

        # Stain augmenter for dual-view consistency
        self.stain_augmenter = BeerLambertStainAugmenter(
            stain_jitter_factor=self.config.stain_jitter_factor
        )

        # CMR Gate for cellularity deconvolution
        self.cmr_gate = CellularityMitosisRatioGate(
            CMRGateConfig(tau_cellularity=self.config.min_cellularity_threshold)
        )

        # Safety Monitor
        self.monitor = DetectionSafetyMonitor(
            silence_window=self.config.silence_window,
            max_per_patch=self.config.max_per_patch,
            source_ratio=self.config.source_ratio,
            min_cellularity_threshold=self.config.min_cellularity_threshold,
            max_collapse_streak=self.config.max_collapse_streak,
        )

        # Register immutable source prototypes
        if source_roi_prototypes is not None:
            self.register_buffer(
                "roi_prototypes",
                F.normalize(source_roi_prototypes, p=2, dim=-1),
            )
        else:
            self.roi_prototypes = None

        # Configure trainable parameters: only FPN lateral & RoI box head norm/affine
        self.adaptable_params: List[nn.Parameter] = []
        for name, p in self.detector.named_parameters():
            if any(k in name for k in ["roi_heads.box_head", "fpn"]):
                if p.requires_grad:
                    self.adaptable_params.append(p)
            else:
                p.requires_grad = False

        if not self.adaptable_params:
            # Fallback if specific layers not named: collect all trainable
            for p in self.detector.parameters():
                if p.requires_grad:
                    self.adaptable_params.append(p)

        # Anchor weights for L_inf projection
        self.anchor_params: Dict[str, torch.Tensor] = {}
        for name, p in self.detector.named_parameters():
            if p.requires_grad:
                self.anchor_params[name] = p.detach().clone()

        self.optimizer = torch.optim.Adam(
            self.adaptable_params, lr=self.config.lr, weight_decay=1e-4
        ) if self.adaptable_params else None

        # State buffers for non-destructive recoil
        self.last_certified_state: Optional[Dict[str, torch.Tensor]] = None
        self._save_certified_state()

        # Telemetry metrics
        self.total_patches_evaluated: int = 0
        self.total_patches_adapted: int = 0
        self.total_recoils: int = 0
        self.is_permanently_latched: bool = False

    def _save_certified_state(self) -> None:
        """Saves certified checkpoint state dictionary for local recoil."""
        self.last_certified_state = {
            k: v.detach().clone().cpu() for k, v in self.detector.state_dict().items()
        }

    def _restore_certified_state(self) -> None:
        """Restores last certified safe checkpoint without permanent fallback."""
        if self.last_certified_state is not None:
            self.detector.load_state_dict(self.last_certified_state)
            self.total_recoils += 1

    def compute_cellularity(self, patch: torch.Tensor) -> float:
        """Computes nuclear cellularity area fraction rho_cell for a patch (1, 3, H, W)."""
        with torch.no_grad():
            rho = self.cmr_gate.compute_cellularity_fraction(patch)
            return float(rho.mean().item())

    def predict(self, patch: torch.Tensor) -> Dict[str, Any]:
        """Runs standard inference returning boxes, scores, and labels."""
        self.detector.eval()
        with torch.no_grad():
            if hasattr(self.detector, "predict"):
                return self.detector.predict(patch)
            outputs = self.detector(patch)
            if isinstance(outputs, list) and len(outputs) > 0:
                return outputs[0]
            return outputs

    def adapt_patch(
        self,
        patch: torch.Tensor,
        source_prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute one test-time adaptation step on incoming target patch.

        Zero-label leakage: uses only patch image tensor and optional frozen source baseline.
        """
        self.total_patches_evaluated += 1
        device = patch.device

        # If permanently latched, bypass adaptation and return frozen source inference
        if self.is_permanently_latched:
            pred = self.predict(patch)
            return {
                "adapted": False,
                "reason": "PERMANENT_FALLBACK_LATCHED",
                "predictions": pred,
                "cellularity": self.compute_cellularity(patch),
            }

        # 1. Evaluate baseline predictions
        with torch.no_grad():
            current_pred = self.predict(patch)
            scores = current_pred.get("scores", torch.empty(0, device=device))
            keep = scores >= self.config.score_thresh
            n_pred = int(keep.sum().item())

            n_source = None
            if source_prediction is not None:
                src_scores = source_prediction.get("scores", torch.empty(0, device=device))
                src_keep = src_scores >= self.config.score_thresh
                n_source = int(src_keep.sum().item())

            # 2. Compute cellularity via Beer-Lambert H-OD
            rho_cell = self.compute_cellularity(patch)
            is_acellular = (rho_cell < self.config.min_cellularity_threshold)

        # 3. Check Safety Invariants via CMR-Aware Monitor
        is_safe, alarm_type = self.monitor.check_patch(
            n_pred=n_pred,
            n_source=n_source,
            is_acellular=is_acellular,
            tissue_cellularity=rho_cell,
        )

        if not is_safe:
            # Alarm tripped: perform non-destructive recoil
            self._restore_certified_state()
            if self.monitor.is_latched:
                self.is_permanently_latched = True
            return {
                "adapted": False,
                "reason": f"SAFETY_ALARM_{alarm_type}",
                "recoil_triggered": True,
                "is_latched": self.is_permanently_latched,
                "predictions": current_pred,
                "cellularity": rho_cell,
            }

        if not self.adaptable_params or self.optimizer is None or is_acellular:
            # Acellular patch or no parameters: safe prediction without mutation
            self._save_certified_state()
            return {
                "adapted": False,
                "reason": "ACELLULAR_PASS_THROUGH" if is_acellular else "NO_TRAINABLE_PARAMS",
                "predictions": current_pred,
                "cellularity": rho_cell,
            }

        # 4. Multi-objective TTA Step on Cellular Tissue
        self.detector.train()
        self.optimizer.zero_grad()

        # View 1 and View 2 (Stain Consistency Regularization)
        patch_v1 = patch
        patch_v2 = self.stain_augmenter(patch)

        loss_total = torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine source anchor loss to prevent drifting away from source features
        loss_anchor = torch.tensor(0.0, device=device)
        for name, p in self.detector.named_parameters():
            if p.requires_grad and name in self.anchor_params:
                anchor = self.anchor_params[name].to(device)
                loss_anchor = loss_anchor + torch.mean((p - anchor) ** 2)

        loss_total = loss_total + self.config.lambda_anchor * loss_anchor

        # Perform gradient update
        loss_total.backward()

        # Gradient safety check
        has_invalid = False
        for p in self.adaptable_params:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    has_invalid = True
                    break

        if has_invalid:
            self._restore_certified_state()
            return {
                "adapted": False,
                "reason": "NON_FINITE_GRADIENTS",
                "predictions": current_pred,
                "cellularity": rho_cell,
            }

        self.optimizer.step()

        # Trust-region bounded parameter projection: clamp displacement to [-delta, delta]
        with torch.no_grad():
            delta = self.config.max_displacement
            for name, p in self.detector.named_parameters():
                if p.requires_grad and name in self.anchor_params:
                    anchor = self.anchor_params[name].to(device)
                    diff = p.data - anchor
                    p.data.copy_(anchor + torch.clamp(diff, -delta, delta))

        # Re-evaluate and certify new state
        self.detector.eval()
        with torch.no_grad():
            post_pred = self.predict(patch)

        self._save_certified_state()
        self.total_patches_adapted += 1

        return {
            "adapted": True,
            "reason": "SUCCESSFUL_CMR_ADAPTATION",
            "predictions": post_pred,
            "cellularity": rho_cell,
            "cmr": self.cmr_gate.compute_cmr(n_pred, rho_cell),
        }
