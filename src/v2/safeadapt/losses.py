"""Safety-Aware Loss Functions for Test-Time Adaptation (Phase 6).

Implements:
1. Entropy Minimization (Shannon entropy on predicted probabilities).
2. Augmentation Consistency Loss (L2 or KL divergence across label-preserving rotations).
3. Elastic Source Anchoring (L2 parameter drift regularization against frozen source weights).
4. Optional Fisher-weighted quadratic penalty.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafetyAwareAdaptationLoss(nn.Module):
    """Configurable multi-objective test-time adaptation loss.

    L_total = w_ent * L_ent + w_cons * L_cons + w_anchor * L_anchor
    """

    def __init__(
        self,
        weight_entropy: float = 1.0,
        weight_consistency: float = 0.5,
        weight_anchor: float = 0.1,
        source_anchors: Optional[Dict[str, torch.Tensor]] = None,
        fisher_diagonal: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.w_ent = float(weight_entropy)
        self.w_cons = float(weight_consistency)
        self.w_anchor = float(weight_anchor)
        self.source_anchors = source_anchors or {}
        self.fisher_diagonal = fisher_diagonal or {}

    def set_source_anchors(self, model: nn.Module) -> None:
        """Store immutable snapshot of current model parameters for anchoring."""
        self.source_anchors = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Binary or multiclass Shannon entropy loss."""
        if logits.dim() == 1 or logits.size(-1) == 1:
            p = torch.sigmoid(logits).view(-1)
            eps = 1e-8
            ent = -p * torch.log(p + eps) - (1.0 - p) * torch.log(1.0 - p + eps)
            return ent.mean()
        else:
            p = F.softmax(logits, dim=-1)
            eps = 1e-8
            ent = -torch.sum(p * torch.log(p + eps), dim=-1)
            return ent.mean()

    def compute_consistency_loss(
        self, model: nn.Module, x: torch.Tensor, base_logits: torch.Tensor
    ) -> torch.Tensor:
        """Prediction consistency across a random 90-degree rotation."""
        k = torch.randint(1, 4, (1,)).item()
        x_rot = torch.rot90(x, k, dims=[2, 3])
        rot_logits = model(x_rot)
        p_base = torch.sigmoid(base_logits)
        p_rot = torch.sigmoid(rot_logits)
        return F.mse_loss(p_rot, p_base)

    def compute_anchor_loss(self, model: nn.Module) -> torch.Tensor:
        """L2 parameter drift regularization: sum_i ||theta_i - theta_0_i||^2."""
        if not self.source_anchors or self.w_anchor <= 0.0:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.source_anchors and param.requires_grad:
                anchor = self.source_anchors[name].to(param.device)
                diff = param - anchor
                if name in self.fisher_diagonal:
                    fisher = self.fisher_diagonal[name].to(param.device)
                    loss = loss + 0.5 * torch.sum(fisher * (diff ** 2))
                else:
                    loss = loss + 0.5 * torch.sum(diff ** 2)
        return loss

    def forward(
        self, model: nn.Module, x: torch.Tensor, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = torch.tensor(0.0, device=x.device)
        breakdown: Dict[str, float] = {}

        if self.w_ent > 0.0:
            l_ent = self.compute_entropy_loss(logits)
            loss = loss + self.w_ent * l_ent
            breakdown["loss_entropy"] = float(l_ent.item())

        if self.w_cons > 0.0:
            l_cons = self.compute_consistency_loss(model, x, logits)
            loss = loss + self.w_cons * l_cons
            breakdown["loss_consistency"] = float(l_cons.item())

        if self.w_anchor > 0.0:
            l_anchor = self.compute_anchor_loss(model)
            loss = loss + self.w_anchor * l_anchor
            breakdown["loss_anchor"] = float(l_anchor.item())

        breakdown["loss_total"] = float(loss.item())
        return loss, breakdown
