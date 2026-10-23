"""Loss functions for UTTA-Med training and evaluation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance or hard samples."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 2 and inputs.shape[1] > 1:
            # Multi-class / 2-class softmax logits
            ce_loss = F.cross_entropy(inputs, targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
            if self.alpha is not None:
                alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
                focal_loss = alpha_t * focal_loss
        else:
            # Binary 1-logit
            inputs_flat = inputs.view(-1)
            targets_flat = targets.view(-1).float()
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs_flat, targets_flat, reduction="none"
            )
            pt = torch.exp(-bce_loss)
            focal_loss = ((1.0 - pt) ** self.gamma) * bce_loss
            if self.alpha is not None:
                alpha_t = torch.where(targets_flat == 1.0, self.alpha, 1.0 - self.alpha)
                focal_loss = alpha_t * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def get_loss_function(
    loss_name: str = "cross_entropy",
    num_classes: int = 2,
    weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function returning the configured loss function.

    Supports:
        - 'cross_entropy': standard CE for num_classes >= 2
        - 'bce': BCEWithLogitsLoss for 1-logit or 2-class binary targets
        - 'focal': Focal loss for focal weighting on hard examples
    """
    name = loss_name.lower().replace("-", "_")

    if name in ("cross_entropy", "ce"):
        if num_classes == 1:
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    elif name in ("bce", "bce_with_logits"):
        return nn.BCEWithLogitsLoss(pos_weight=weight)

    elif name == "focal":
        return FocalLoss(gamma=gamma)

    else:
        raise ValueError(
            f"Unsupported loss function '{loss_name}'. Expected one of ['cross_entropy', 'bce', 'focal']."
        )
