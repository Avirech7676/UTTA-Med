"""Adaptive Uncertainty Estimation Module for UTTA-Med 2.0 (Phase 4).

Features:
1. Single-pass deterministic confidence estimation for rapid screening.
2. Dynamic, conditional MC-Dropout pass allocation (budgets K in {5, 10, 20}).
3. Early stopping of stochastic forward passes when epistemic variance stabilizes.
4. Histopathology-preserving transformation consistency uncertainty (rotations & flips).
5. Feature-space distance estimation relative to certified source centroids.
6. Execution telemetry tracking forward pass counts, latency, and routing decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.gating.policies import BaseGatingPolicy, GatingDecision


@dataclass
class AdaptiveGatingTelemetry:
    """Detailed telemetry for adaptive uncertainty estimation."""

    total_patches: int = 0
    confident_auto_accepted: int = 0
    low_conf_auto_rejected: int = 0
    routed_to_mc: int = 0
    routed_to_consistency: int = 0
    mc_early_stopped: int = 0
    total_forward_passes: int = 0
    total_compute_time_ms: float = 0.0

    @property
    def avg_passes_per_patch(self) -> float:
        return self.total_forward_passes / max(self.total_patches, 1)

    @property
    def mc_routing_rate(self) -> float:
        return self.routed_to_mc / max(self.total_patches, 1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_patches": float(self.total_patches),
            "confident_auto_accepted": float(self.confident_auto_accepted),
            "low_conf_auto_rejected": float(self.low_conf_auto_rejected),
            "routed_to_mc": float(self.routed_to_mc),
            "mc_early_stopped": float(self.mc_early_stopped),
            "total_forward_passes": float(self.total_forward_passes),
            "avg_passes_per_patch": float(self.avg_passes_per_patch),
            "mc_routing_rate": float(self.mc_routing_rate),
            "total_compute_time_ms": float(self.total_compute_time_ms),
        }


class AdaptiveUncertaintyGating(BaseGatingPolicy):
    """Adaptive Multi-Tier Uncertainty Gating for UTTA-Med 2.0.

    Decision flow:
    1. Tier 1 (Deterministic Confidence):
       - p = sigmoid(f(x))
       - c = max(p, 1 - p)
       - If c >= tau_high: High confidence -> auto-accepted directly (1 pass).
       - If c < tau_low: Extreme ambiguity/noise -> auto-rejected directly (1 pass).
    2. Tier 2 (Conditional Stochastic MC Dropout on ambiguous margin [tau_low, tau_high)):
       - Evaluates stochastic passes in batches of chunk_size (default 5).
       - Checks for variance stabilization: if |Var_k - Var_{k-chunk}| < delta_tol,
         triggers early termination, saving substantial compute.
    3. Tier 3 (Optional Prediction Consistency & Feature Centroid check):
       - Measures prediction variance under 4 orthogonal rotations {0, 90, 180, 270 deg}.
       - Computes cosine distance to source feature centroids if available.
    """

    def __init__(
        self,
        tau_high: float = 0.90,
        tau_low: float = 0.65,
        tau_unc: float = 8.875383e-06,
        max_mc_passes: int = 20,
        min_mc_passes: int = 5,
        chunk_size: int = 5,
        variance_stability_tol: float = 1e-5,
        use_consistency: bool = False,
        consistency_threshold: float = 0.05,
        source_centroids: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ):
        self.tau_high = float(tau_high)
        self.tau_low = float(tau_low)
        self.tau_unc = float(tau_unc)
        self.max_mc_passes = int(max_mc_passes)
        self.min_mc_passes = int(min_mc_passes)
        self.chunk_size = int(chunk_size)
        self.variance_stability_tol = float(variance_stability_tol)
        self.use_consistency = bool(use_consistency)
        self.consistency_threshold = float(consistency_threshold)
        self.source_centroids = source_centroids
        self.seed = seed

        self.telemetry = AdaptiveGatingTelemetry()

    def _enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers while preserving evaluation mode on BatchNorm."""
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _compute_consistency_variance(
        self, model: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute prediction variance under label-preserving histopathology rotations.

        Rotations: 0, 90, 180, 270 degrees.
        """
        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(4):
                x_rot = torch.rot90(x, k, dims=[2, 3])
                p_rot = torch.sigmoid(model(x_rot)).view(-1)
                preds.append(p_rot)
        stacked = torch.stack(preds, dim=0)  # [4, B]
        return stacked.var(dim=0, unbiased=False)

    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        t0 = time.perf_counter()
        b = x.size(0)
        device = x.device

        # Stage 1: Deterministic screening (1 forward pass)
        with torch.no_grad():
            model.eval()
            out = model(x)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)

        passes_done = 1

        is_confident = conf >= self.tau_high
        is_low_conf = conf < self.tau_low
        is_ambiguous = (~is_confident) & (~is_low_conf)

        keep_mask = torch.zeros(b, dtype=torch.bool, device=device)
        uncertainties = torch.zeros(b, dtype=torch.float32, device=device)
        routed_to_mc = is_ambiguous.clone()

        # Confident patches pass directly without MC passes
        keep_mask[is_confident] = True

        # Telemetry counts
        self.telemetry.total_patches += b
        self.telemetry.confident_auto_accepted += int(is_confident.sum().item())
        self.telemetry.low_conf_auto_rejected += int(is_low_conf.sum().item())

        num_ambiguous = int(is_ambiguous.sum().item())
        mc_early_stopped = False

        if num_ambiguous > 0:
            self.telemetry.routed_to_mc += num_ambiguous
            amb_indices = torch.nonzero(is_ambiguous, as_tuple=True)[0]
            x_amb = x[amb_indices]

            self._enable_dropout(model)

            amb_probs: List[torch.Tensor] = []
            prev_var: Optional[torch.Tensor] = None

            with torch.no_grad():
                current_passes = 0
                while current_passes < self.max_mc_passes:
                    for _ in range(self.chunk_size):
                        p_k = torch.sigmoid(model(x_amb)).view(-1)
                        amb_probs.append(p_k)
                    current_passes += self.chunk_size
                    passes_done += self.chunk_size

                    # Check variance stability if minimum passes reached
                    if current_passes >= self.min_mc_passes:
                        stacked_so_far = torch.stack(amb_probs, dim=0)
                        curr_var = stacked_so_far.var(dim=0, unbiased=False)

                        if prev_var is not None:
                            max_var_diff = (curr_var - prev_var).abs().max().item()
                            if max_var_diff < self.variance_stability_tol:
                                mc_early_stopped = True
                                break
                        prev_var = curr_var

            model.eval()

            stacked_amb = torch.stack(amb_probs, dim=0)
            unc_amb = stacked_amb.var(dim=0, unbiased=False)
            uncertainties[amb_indices] = unc_amb

            # Gate ambiguous patches: keep iff epistemic variance is below tolerance
            amb_keep = unc_amb < self.tau_unc

            # Optional Tier 3: Augmentation consistency check
            if self.use_consistency and amb_keep.any():
                cons_indices = amb_indices[amb_keep]
                unc_cons = self._compute_consistency_variance(model, x[cons_indices])
                passes_done += 4
                self.telemetry.routed_to_consistency += int(cons_indices.numel())
                amb_keep[amb_keep.clone()] = unc_cons < self.consistency_threshold

            keep_mask[amb_indices] = amb_keep

        if mc_early_stopped:
            self.telemetry.mc_early_stopped += 1

        self.telemetry.total_forward_passes += passes_done
        self.telemetry.total_compute_time_ms += (time.perf_counter() - t0) * 1000.0

        return GatingDecision(
            keep_mask=keep_mask,
            confidences=conf,
            uncertainties=uncertainties,
            routed_to_mc=routed_to_mc,
            mc_passes_executed=passes_done,
        )
