"""Bio-SafeAdapt: Cellularity-Conditioned Adaptation Controller for Dense Detection (Phase 7).

Implements:
1. Optical Density (H-OD) tissue cellularity estimation to distinguish sparse background from detector collapse.
2. Cellularity-conditioned silence monitoring (prevents false alarms on acellular stroma/fat).
3. Source-vs-Adapted detection disagreement tracking.
4. Continuous elastic parameter anchoring to prevent catastrophic drift.
5. Patient-level / case-boundary episodic resets.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController
from src.v2.safeadapt.monitor import AlarmLevel, MonitorMode


@dataclass
class BioMonitorDecision:
    """Decision output for a single detection patch or batch."""

    allow_update: bool
    alarm_level: AlarmLevel
    cellularity_ratio: float
    silence_suppressed: bool
    source_detections_count: int
    adapted_detections_count: int
    reason: str


class BioSafeAdaptController(SafeAdaptController):
    """Bio-SafeAdapt Controller incorporating biological tissue context into TTA safeguards."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[SafeAdaptConfig] = None,
        adaptable_params: Optional[List[nn.Parameter]] = None,
        min_cellularity_threshold: float = 0.05,
        hematoxylin_od_threshold: float = 0.15,
        max_consecutive_silence: int = 5,
        anchor_lambda: float = 0.01,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            config=config,
            adaptable_params=adaptable_params,
        )
        self.min_cellularity_threshold = float(min_cellularity_threshold)
        self.hematoxylin_od_threshold = float(hematoxylin_od_threshold)
        self.max_consecutive_silence = int(max_consecutive_silence)
        self.anchor_lambda = float(anchor_lambda)

        self._consecutive_cellular_silence: int = 0
        self._case_id: Optional[str] = None
        self._case_updates_accepted: int = 0
        self._case_rollbacks: int = 0

    @staticmethod
    def estimate_cellularity(
        images: torch.Tensor,
        od_threshold: float = 0.15,
    ) -> torch.Tensor:
        """Estimate tissue cellularity via optical density of hematoxylin nuclear staining.

        Args:
            images: Tensor of shape [B, 3, H, W] normalized in [0, 1] or standard ImageNet.
        Returns:
            Tensor of shape [B] with cellularity fractions in [0, 1].
        """
        # Ensure values in [0, 1]
        x = images.clone().detach()
        if x.min() < 0.0:  # ImageNet normalized roughly [-2.1, 2.6]
            x = (x * 0.225) + 0.45
        x = torch.clamp(x, min=1e-3, max=1.0)

        # Optical Density: OD = -log10(I)
        od = -torch.log10(x)

        # Hematoxylin stains nucleic acids purple/blue (absorption predominantly in red/green channels)
        # Simplified Ruifrok-Johnston stain vector projection for Hematoxylin proxy:
        # H optical density is strongly characterized by red-channel attenuation:
        od_r = od[:, 0, :, :]
        od_b = od[:, 2, :, :]
        h_proxy = od_r - 0.5 * od_b  # High when red is absorbed more than blue (purple/nuclei)

        high_nuclear_density = h_proxy > od_threshold
        cellularity = high_nuclear_density.float().mean(dim=[-2, -1])  # [B]
        return cellularity

    def monitor_detection_step(
        self,
        images: torch.Tensor,
        source_dets: List[Dict[str, torch.Tensor]],
        adapted_dets: List[Dict[str, torch.Tensor]],
    ) -> BioMonitorDecision:
        """Evaluate tissue cellularity and detection discrepancy."""
        cellularities = self.estimate_cellularity(images, self.hematoxylin_od_threshold)
        avg_cell = float(cellularities.mean().item())

        src_count = sum(len(d.get("boxes", [])) for d in source_dets)
        adp_count = sum(len(d.get("boxes", [])) for d in adapted_dets)

        is_sparse = avg_cell < self.min_cellularity_threshold

        # If tissue is sparse (acellular stroma or background glass), zero detections is natural!
        if is_sparse and adp_count == 0:
            return BioMonitorDecision(
                allow_update=False,  # Skip adaptation on uninformative background
                alarm_level=AlarmLevel.NORMAL,
                cellularity_ratio=avg_cell,
                silence_suppressed=True,
                source_detections_count=src_count,
                adapted_detections_count=adp_count,
                reason="Sparse tissue background: silence alarm suppressed, skipping adaptation",
            )

        # Check for genuine detector silence on cellular tissue
        if avg_cell >= self.min_cellularity_threshold and adp_count == 0 and src_count > 0:
            self._consecutive_cellular_silence += 1
            if self._consecutive_cellular_silence >= self.max_consecutive_silence:
                return BioMonitorDecision(
                    allow_update=False,
                    alarm_level=AlarmLevel.ROLLBACK,
                    cellularity_ratio=avg_cell,
                    silence_suppressed=False,
                    source_detections_count=src_count,
                    adapted_detections_count=adp_count,
                    reason=f"Detector collapse: {self._consecutive_cellular_silence} consecutive silent cellular patches",
                )
            else:
                return BioMonitorDecision(
                    allow_update=False,
                    alarm_level=AlarmLevel.WARNING,
                    cellularity_ratio=avg_cell,
                    silence_suppressed=False,
                    source_detections_count=src_count,
                    adapted_detections_count=adp_count,
                    reason="Suspicious silence on cellular tissue: update skipped",
                )

        # Reset silence counter when detections reappear
        if adp_count > 0:
            self._consecutive_cellular_silence = 0

        return BioMonitorDecision(
            allow_update=True,
            alarm_level=AlarmLevel.NORMAL,
            cellularity_ratio=avg_cell,
            silence_suppressed=False,
            source_detections_count=src_count,
            adapted_detections_count=adp_count,
            reason="Nominal cellular patch with stable detections",
        )

    def apply_elastic_source_anchor(self) -> float:
        """Apply soft elastic anchoring toward the source weights before optimizer step."""
        if self.anchor_lambda <= 0.0:
            return 0.0

        drift_loss = 0.0
        for p in self.adaptable_params:
            if p.grad is not None and id(p) in self._source_param_anchors:
                anchor = self._source_param_anchors[id(p)].to(p.device)
                drift_grad = self.anchor_lambda * (p.data - anchor)
                p.grad.data.add_(drift_grad)
                drift_loss += float((p.data - anchor).norm().item())

        return drift_loss

    def reset_case_boundary(self, new_case_id: str) -> None:
        """Episodic reset at patient / slide boundaries.

        Restores certified source parameters and resets optimizer momentum,
        eliminating inter-patient sequence dependence.
        """
        self._case_id = new_case_id
        self._consecutive_cellular_silence = 0
        self._case_updates_accepted = 0
        self._case_rollbacks = 0

        # Atomic restore to clean source baseline and clear optimizer state
        self.reset_boundary(slide_id=new_case_id, restore_source_weights=True)
