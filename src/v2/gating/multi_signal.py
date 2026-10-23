"""Multi-Signal Uncertainty Gating Module for UTTA-Med 2.0 (Phase 5).

Implements three orthogonal, configurable uncertainty signals:
Signal 1 (U_MC): Monte Carlo Dropout epistemic predictive variance.
Signal 2 (U_consistency): Prediction disagreement under label-preserving histopathology transforms.
Signal 3 (U_feature): Normalized feature-distribution distance to source domain prototypes.

Convex combination:
    U_total = alpha * U_MC_norm + beta * U_consistency_norm + gamma * U_feature_norm
Subject to:
    alpha + beta + gamma = 1.0,  alpha, beta, gamma >= 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.gating.policies import BaseGatingPolicy, GatingDecision


@dataclass
class MultiSignalDecision(GatingDecision):
    """Extended GatingDecision containing decomposed uncertainty signals."""

    u_mc: Optional[torch.Tensor] = None
    u_consistency: Optional[torch.Tensor] = None
    u_feature: Optional[torch.Tensor] = None
    u_total: Optional[torch.Tensor] = None


class MultiSignalGatingPolicy(BaseGatingPolicy):
    """Configurable Multi-Signal Uncertainty Gating.

    Evaluates whether multi-modal uncertainty estimates provide information
    beyond simple confidence, supporting rigorous matched-compute ablations.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        tau_total: float = 0.05,
        n_mc_passes: int = 10,
        source_centroids: Optional[torch.Tensor] = None,
        scale_mc: float = 1e-4,
        scale_consistency: float = 0.02,
        scale_feature: float = 1.0,
    ):
        assert abs((alpha + beta + gamma) - 1.0) < 1e-5, "alpha + beta + gamma must equal 1.0"
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.tau_total = float(tau_total)
        self.n_mc_passes = int(n_mc_passes)
        self.source_centroids = source_centroids

        # Normalization scale constants (calibrated on out-of-domain dev center 1)
        self.scale_mc = float(scale_mc)
        self.scale_consistency = float(scale_consistency)
        self.scale_feature = float(scale_feature)

    def _enable_dropout(self, model: nn.Module) -> None:
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def _compute_mc_uncertainty(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Signal 1: Stochastic MC Dropout predictive variance."""
        self._enable_dropout(model)
        probs = []
        with torch.no_grad():
            for _ in range(self.n_mc_passes):
                p_k = torch.sigmoid(model(x)).view(-1)
                probs.append(p_k)
        model.eval()
        stacked = torch.stack(probs, dim=0)
        return stacked.var(dim=0, unbiased=False)

    def _compute_consistency_uncertainty(
        self, model: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Signal 2: Prediction variance across 4 orthogonal rotations."""
        model.eval()
        preds = []
        with torch.no_grad():
            for k in range(4):
                x_rot = torch.rot90(x, k, dims=[2, 3])
                p_rot = torch.sigmoid(model(x_rot)).view(-1)
                preds.append(p_rot)
        stacked = torch.stack(preds, dim=0)
        return stacked.var(dim=0, unbiased=False)

    def _compute_feature_uncertainty(
        self, model: nn.Module, x: torch.Tensor
    ) -> torch.Tensor:
        """Signal 3: Cosine distance to certified source feature centroid."""
        b = x.size(0)
        device = x.device
        if self.source_centroids is None:
            return torch.zeros(b, dtype=torch.float32, device=device)

        # Feature extraction hook if model supports forward_features or penultimate layer
        with torch.no_grad():
            model.eval()
            if hasattr(model, "forward_features"):
                feat = model.forward_features(x)
            elif hasattr(model, "backbone"):
                feat = model.backbone(x)
            else:
                # Fallback: dummy zero distance if penultimate representation is unexposed
                return torch.zeros(b, dtype=torch.float32, device=device)

            if feat.dim() > 2:
                feat = feat.mean(dim=[-2, -1])  # Global average pool

            centroids = self.source_centroids.to(device)
            if centroids.dim() == 1:
                centroids = centroids.unsqueeze(0)

            # Strict feature dimension validation - do not truncate across incompatible spaces
            if centroids.size(-1) != feat.size(-1):
                if centroids.size(0) == feat.size(-1):
                    centroids = centroids.t()
                else:
                    raise ValueError(
                        f"Feature centroid dimension mismatch: centroids have {centroids.size(-1)}-D, "
                        f"but model produces {feat.size(-1)}-D. Reusing or truncating vectors across "
                        f"different architectures (e.g. ResNet-50 vs ResNet-18) is strictly prohibited."
                    )

            # Cosine distance = 1 - cosine_similarity
            feat_norm = F.normalize(feat, p=2, dim=1)
            cent_norm = F.normalize(centroids, p=2, dim=1)
            sim = torch.mm(feat_norm, cent_norm.t()).max(dim=1)[0]
            dist = torch.clamp(1.0 - sim, min=0.0, max=2.0)
            return dist

    def select(self, model: nn.Module, x: torch.Tensor) -> MultiSignalDecision:
        b = x.size(0)
        device = x.device

        # Deterministic forward pass for base confidence
        with torch.no_grad():
            model.eval()
            out = model(x)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)

        passes_done = 1

        # Signal 1: MC-Dropout
        if self.alpha > 0.0:
            u_mc = self._compute_mc_uncertainty(model, x)
            passes_done += self.n_mc_passes
        else:
            u_mc = torch.zeros(b, dtype=torch.float32, device=device)

        # Signal 2: Consistency
        if self.beta > 0.0:
            u_cons = self._compute_consistency_uncertainty(model, x)
            passes_done += 4
        else:
            u_cons = torch.zeros(b, dtype=torch.float32, device=device)

        # Signal 3: Feature Distance
        if self.gamma > 0.0:
            u_feat = self._compute_feature_uncertainty(model, x)
        else:
            u_feat = torch.zeros(b, dtype=torch.float32, device=device)

        # Normalize signals
        u_mc_norm = u_mc / max(self.scale_mc, 1e-8)
        u_cons_norm = u_cons / max(self.scale_consistency, 1e-8)
        u_feat_norm = u_feat / max(self.scale_feature, 1e-8)

        # Convex combination
        u_total = (
            self.alpha * u_mc_norm
            + self.beta * u_cons_norm
            + self.gamma * u_feat_norm
        )

        keep_mask = u_total < self.tau_total

        return MultiSignalDecision(
            keep_mask=keep_mask,
            confidences=conf,
            uncertainties=u_total,
            routed_to_mc=torch.ones(b, dtype=torch.bool, device=device) if self.alpha > 0 else torch.zeros(b, dtype=torch.bool, device=device),
            mc_passes_executed=passes_done,
            u_mc=u_mc,
            u_consistency=u_cons,
            u_feature=u_feat,
            u_total=u_total,
        )
