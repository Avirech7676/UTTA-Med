"""Optimal Transport Prototype Alignment (OT-TTA) & Anti-Attractor Regularization.

Implements geometry-preserving test-time domain adaptation via Sinkhorn-Knopp
Optimal Transport against preserved source class prototypes, eliminating single-class
attractor collapse under severe cross-center stain shifts.

Mathematical Formulation:
1. Distance/Cost Matrix:
   C_{ik} = 1 - (z_i^T μ_k) / (||z_i|| * ||μ_k||)
2. Entropic Optimal Transport (Sinkhorn-Knopp):
   min_{Q in Π} <Q, C> - ε * H(Q), where marginal constraints enforce
   Q @ 1_K = (1/B) * 1_B and Q^T @ 1_B = r (prior class distribution).
3. OT Alignment Loss:
   L_OT = sum_{i=1}^B sum_{k=1}^K Q^*_{ik} * C_{ik}
4. Anti-Attractor Diversity Loss:
   L_div = D_KL(p_bar || r) = sum_{k=1}^K p_bar_k * log((p_bar_k + δ) / r_k)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tta.bn import collect_bn_affine, configure_tta_mode
from src.tta.stain_consistency import BeerLambertStainAugmenter


def sinkhorn_knopp_transport(
    cost_matrix: torch.Tensor,
    class_priors: torch.Tensor,
    epsilon: float = 0.05,
    max_iters: int = 50,
    tol: float = 1e-4,
) -> torch.Tensor:
    """Computes soft assignment matrix Q* via Sinkhorn-Knopp iterations."""
    batch_size, num_classes = cost_matrix.shape
    device = cost_matrix.device

    # Kernel matrix
    K = torch.exp(-cost_matrix / epsilon)

    # Marginals: u in R^B (uniform 1/B), v in R^K (class_priors)
    u = torch.ones(batch_size, device=device) / batch_size
    v = class_priors.to(device)

    for _ in range(max_iters):
        u_prev = u
        u = (1.0 / batch_size) / (torch.matmul(K, v) + 1e-12)
        v = class_priors / (torch.matmul(K.t(), u) + 1e-12)

        if torch.max(torch.abs(u - u_prev)) < tol:
            break

    # Final normalization step on u to guarantee exact row marginals
    u = (1.0 / batch_size) / (torch.matmul(K, v) + 1e-12)

    # Reconstruct optimal coupling matrix Q*
    Q = torch.diag_embed(u) @ K @ torch.diag_embed(v)
    return Q


class PrototypeOT_TTA(nn.Module):
    """Test-Time Adaptation controller combining Optimal Transport Prototype Alignment and Diversity."""

    def __init__(
        self,
        model: nn.Module,
        source_prototypes: torch.Tensor,
        class_priors: Optional[torch.Tensor] = None,
        lr: float = 1e-5,
        lambda_ot: float = 1.0,
        lambda_div: float = 0.5,
        lambda_anchor: float = 0.01,
        lambda_cr: float = 0.5,
        epsilon_ot: float = 0.05,
        max_displacement: float = 0.05,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lambda_ot = lambda_ot
        self.lambda_div = lambda_div
        self.lambda_anchor = lambda_anchor
        self.lambda_cr = lambda_cr
        self.epsilon_ot = epsilon_ot
        self.max_displacement = max_displacement
        self.stain_augmenter = BeerLambertStainAugmenter()

        # Register immutable source prototypes: Shape (K, D)
        self.register_buffer("prototypes", F.normalize(source_prototypes, p=2, dim=-1))
        self.num_classes, self.feature_dim = self.prototypes.shape

        # Prior class distribution (default uniform: 1/K)
        if class_priors is None:
            priors = torch.ones(self.num_classes) / self.num_classes
        else:
            priors = class_priors / class_priors.sum()
        self.register_buffer("class_priors", priors)

        # Freeze model and enable strictly BatchNorm affine parameters
        configure_tta_mode(self.model, bn_mode="bn_freeze_stats")
        self.adaptable_params = collect_bn_affine(self.model)

        # Snapshot immutable source anchor weights
        self.anchor_params: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.anchor_params[name] = p.data.clone().detach()

        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)

    def compute_cosine_cost(self, features: torch.Tensor) -> torch.Tensor:
        """Computes cosine distance cost matrix between features and prototypes."""
        norm_feats = F.normalize(features, p=2, dim=-1)
        sim = torch.matmul(norm_feats, self.prototypes.t())  # Shape (B, K)
        cost = 1.0 - sim
        return cost

    def adapt_batch(self, x: torch.Tensor) -> Dict[str, Any]:
        """Performs one step of Optimal Transport, Diversity & Stain Consistency adaptation."""
        self.model.eval()

        # Extract features and logits
        if hasattr(self.model, "extract_features"):
            feats = self.model.extract_features(x)
            logits = self.model.fc(feats)
        else:
            # Fallback for standard ResNet
            feats = self.model.stem(x) if hasattr(self.model, "stem") else x
            feats = torch.flatten(feats, 1)
            logits = self.model.classifier(feats) if hasattr(self.model, "classifier") else self.model.fc(feats)

        batch_size = x.size(0)
        probs = F.softmax(logits, dim=-1) if logits.size(-1) > 1 else torch.sigmoid(logits)
        if logits.size(-1) == 1:
            probs = torch.cat([1.0 - probs, probs], dim=-1)

        # 1. Optimal Transport Cost and Soft Assignment
        cost = self.compute_cosine_cost(feats)
        with torch.no_grad():
            Q_opt = sinkhorn_knopp_transport(
                cost, self.class_priors, epsilon=self.epsilon_ot
            )

        # OT Assignment Loss: pull each feature toward its optimal prototype cluster
        loss_ot = torch.sum(Q_opt * cost)

        # 2. Anti-Attractor Class Diversity Loss
        p_bar = torch.mean(probs, dim=0)
        loss_div = torch.sum(p_bar * torch.log((p_bar + 1e-8) / (self.class_priors + 1e-8)))

        # 3. Source Anchor Regularization Loss
        loss_anchor = torch.tensor(0.0, device=x.device)
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.anchor_params:
                anchor = self.anchor_params[name].to(x.device)
                loss_anchor = loss_anchor + torch.sum((p - anchor) ** 2)

        # 4. Beer-Lambert Stain Consistency Regularization Loss
        loss_cr = torch.tensor(0.0, device=x.device)
        if self.lambda_cr > 0.0:
            x_cr = self.stain_augmenter(x)
            if hasattr(self.model, "extract_features"):
                feats_cr = self.model.extract_features(x_cr)
                logits_cr = self.model.fc(feats_cr)
            else:
                feats_cr = self.model.stem(x_cr) if hasattr(self.model, "stem") else x_cr
                feats_cr = torch.flatten(feats_cr, 1)
                logits_cr = self.model.classifier(feats_cr) if hasattr(self.model, "classifier") else self.model.fc(feats_cr)
            probs_cr = F.softmax(logits_cr, dim=-1) if logits_cr.size(-1) > 1 else torch.sigmoid(logits_cr)
            if logits_cr.size(-1) == 1:
                probs_cr = torch.cat([1.0 - probs_cr, probs_cr], dim=-1)
            loss_cr = F.mse_loss(probs, probs_cr)

        total_loss = (
            self.lambda_ot * loss_ot
            + self.lambda_div * loss_div
            + self.lambda_anchor * loss_anchor
            + self.lambda_cr * loss_cr
        )

        # Gradient update
        self.optimizer.zero_grad()
        total_loss.backward()

        # Check for invalid gradients
        has_invalid = False
        for p in self.adaptable_params:
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_invalid = True
                break

        if not has_invalid:
            self.optimizer.step()
            # Trust-Region Bounded Projection
            with torch.no_grad():
                for name, p in self.model.named_parameters():
                    if p.requires_grad and name in self.anchor_params:
                        anchor = self.anchor_params[name].to(x.device)
                        diff = p.data - anchor
                        clipped = torch.clamp(diff, -self.max_displacement, self.max_displacement)
                        p.data.copy_(anchor + clipped)

        # Post-adaptation predictions
        with torch.no_grad():
            if hasattr(self.model, "extract_features"):
                post_feats = self.model.extract_features(x)
                post_logits = self.model.fc(post_feats)
            else:
                post_logits = logits

        return {
            "adapted": not has_invalid,
            "loss_total": float(total_loss.item()),
            "loss_ot": float(loss_ot.item()),
            "loss_div": float(loss_div.item()),
            "loss_cr": float(loss_cr.item()),
            "logits": post_logits.detach(),
        }

