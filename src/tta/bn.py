"""Shared BatchNorm-affine-only update-scope helpers (parity across all TTA methods)."""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn


def collect_bn_affine(model: nn.Module) -> List[nn.Parameter]:
    """Freeze everything except BatchNorm2d affine parameters (γ, β)."""
    for p in model.parameters():
        p.requires_grad = False
    params: List[nn.Parameter] = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and m.affine:
            m.weight.requires_grad = True
            m.bias.requires_grad = True
            params.append(m.weight)
            params.append(m.bias)
    if not params:
        raise RuntimeError("No BatchNorm affine parameters found")
    return params


def configure_tta_mode(
    model: nn.Module,
    freeze_running_stats: bool = True,
    bn_mode: Optional[str] = None,
) -> None:
    """Official protocol: bn_mode='bn_freeze_stats' (running mean/var frozen).

    bn_mode='bn_train' is original Tent and collapsed on Camelyon17 even at lr=1e-5.
    Dropout is always eval so MC-Dropout is not mixed into the TTA step.
    """
    if bn_mode is not None:
        freeze_running_stats = bn_mode != "bn_train"
    model.train()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
        elif isinstance(m, nn.BatchNorm2d):
            if freeze_running_stats:
                m.eval()
            else:
                m.train()


def binary_entropy_from_logits(logits):
    import torch

    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).view(-1)
