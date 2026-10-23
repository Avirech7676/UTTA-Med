"""Confidence + Feature-Distance Gating Policy (UTTA-Med 2.0 Advanced Development).

Combines:
1. Single-pass deterministic predictive confidence S_conf = 2 * (max(p, 1-p) - 0.5) in [0, 1].
2. ResNet-18 source-feature centroid cosine proximity S_feat = clamp(1 - d_feat / d_max, 0, 1).

Joint Quality Formulation:
    S_joint = w_conf * S_conf + w_feat * S_feat
    keep_mask = S_joint >= tau_joint

Enables lightweight, single-pass out-of-distribution detection to filter out
overconfident mispredictions that lie far from source training prototypes.
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
class ConfFeatureDecision(GatingDecision):
    """Extended GatingDecision containing decomposed confidence and feature scores."""

    conf_scores: Optional[torch.Tensor] = None      # S_conf in [0, 1]
    feat_scores: Optional[torch.Tensor] = None      # S_feat in [0, 1]
    joint_scores: Optional[torch.Tensor] = None     # S_joint in [0, 1]
    cosine_distances: Optional[torch.Tensor] = None # raw cosine distances d_feat in [0, 2]


class ConfidenceFeatureGatingPolicy(BaseGatingPolicy):
    """Lightweight Single-Pass Gating Combining Confidence and Feature Centroid Proximity."""

    def __init__(
        self,
        w_conf: float = 0.5,
        w_feat: float = 0.5,
        tau_joint: float = 0.85,
        source_centroids: Optional[torch.Tensor] = None,
        d_max: float = 0.80,
    ):
        assert w_conf >= 0.0 and w_feat >= 0.0, "Weights must be non-negative"
        total_w = w_conf + w_feat
        assert total_w > 0.0, "Total weight must be positive"
        self.w_conf = float(w_conf / total_w)
        self.w_feat = float(w_feat / total_w)
        self.tau_joint = float(tau_joint)
        self.source_centroids = source_centroids
        self.d_max = float(d_max)

        # Validate centroids if provided
        if self.source_centroids is not None:
            assert self.source_centroids.dim() == 2, f"Centroids must be 2D [K, D], got {self.source_centroids.shape}"
            assert self.source_centroids.size(0) >= 2, f"Centroids must have at least 2 classes, got {self.source_centroids.size(0)}"

    def _extract_features_and_logits(
        self, model: nn.Module, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract latent bottleneck features (512-D) and final classification logits."""
        if hasattr(model, "forward_features"):
            feat = model.forward_features(x)
            if hasattr(model, "classifier"):
                logits = model.classifier(feat).view(-1)
            elif hasattr(model, "fc"):
                logits = model.fc(feat).view(-1)
            else:
                logits = model(x).view(-1)
            return feat, logits
        elif hasattr(model, "extract_features"):
            feat = model.extract_features(x)
            if hasattr(model, "fc"):
                logits = model.fc(feat).view(-1)
            elif hasattr(model, "classifier"):
                logits = model.classifier(feat).view(-1)
            else:
                out = model(x)
                logits = out.view(-1)
            return feat, logits

        # Standard torchvision / custom ResNet-18 forward hook fallback
        features: List[torch.Tensor] = []

        def hook_fn(module, input, output):
            # output of avgpool is [B, 512, 1, 1]
            features.append(torch.flatten(output, 1))

        # Find avgpool layer
        target_layer = None
        for name, module in model.named_modules():
            if "avgpool" in name or isinstance(module, nn.AdaptiveAvgPool2d):
                target_layer = module
                break

        if target_layer is not None:
            hook = target_layer.register_forward_hook(hook_fn)
            out = model(x)
            hook.remove()
            if features:
                feat = features[0]
            else:
                feat = torch.zeros(x.size(0), 512, device=x.device)
            logits = out.view(-1)
            return feat, logits

        # Generic fallback
        out = model(x)
        logits = out.view(-1)
        feat = torch.zeros(x.size(0), 512, device=x.device)
        return feat, logits

    def select(self, model: nn.Module, x: torch.Tensor) -> ConfFeatureDecision:
        t0 = time.perf_counter()
        device = x.device
        b = x.size(0)

        with torch.no_grad():
            model.eval()
            feat, logits = self._extract_features_and_logits(model, x)
            p = torch.sigmoid(logits).view(-1)

        # 1. Predictive Confidence Score: S_conf in [0, 1]
        conf = torch.maximum(p, 1.0 - p)
        s_conf = (conf - 0.5) * 2.0  # Maps [0.5, 1.0] -> [0.0, 1.0]

        # 2. Source Feature Distance Score: S_feat in [0, 1]
        if self.source_centroids is not None and feat.size(-1) == self.source_centroids.size(-1):
            centroids = self.source_centroids.to(device)
            # Normalize embeddings and centroids
            feat_norm = F.normalize(feat, p=2, dim=-1)
            cent_norm = F.normalize(centroids, p=2, dim=-1)

            # Assign to predicted class prototype
            pred_class = (p >= 0.5).long()
            selected_centroids = cent_norm[pred_class]  # [B, 512]

            # Cosine similarity and distance
            cos_sim = torch.sum(feat_norm * selected_centroids, dim=-1).clamp(-1.0, 1.0)
            d_feat = 1.0 - cos_sim  # [0, 2]
            s_feat = torch.clamp(1.0 - (d_feat / self.d_max), min=0.0, max=1.0)
        else:
            d_feat = torch.zeros(b, device=device)
            s_feat = s_conf.clone()  # Degrade gracefully to confidence if no centroid

        # 3. Combined Joint Score
        s_joint = self.w_conf * s_conf + self.w_feat * s_feat
        keep_mask = s_joint >= self.tau_joint
        uncertainties = 1.0 - s_joint

        return ConfFeatureDecision(
            keep_mask=keep_mask,
            confidences=conf,
            uncertainties=uncertainties,
            routed_to_mc=torch.zeros(b, dtype=torch.bool, device=device),
            mc_passes_executed=1,
            conf_scores=s_conf,
            feat_scores=s_feat,
            joint_scores=s_joint,
            cosine_distances=d_feat,
        )
