"""Gate-control scoring functions (Random / Confidence / Entropy / MC-Dropout)."""

from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn

from src.tta.bn import binary_entropy_from_logits

GateName = Literal["random", "confidence", "entropy", "mc_dropout"]


@torch.no_grad()
def score_confidence(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Softmax/sigmoid confidence = max(p, 1-p). Higher = more confident."""
    model.eval()
    p = torch.sigmoid(model(x)).view(-1)
    return torch.maximum(p, 1.0 - p)


@torch.no_grad()
def score_entropy(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Predictive entropy. Lower = more certain. Returned as *negative* entropy so
    higher score still means 'more likely to be accepted' when using a threshold."""
    model.eval()
    return -binary_entropy_from_logits(model(x))


@torch.no_grad()
def score_random(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.rand(x.size(0), device=x.device, generator=generator)


@torch.no_grad()
def score_mc_dropout(model: nn.Module, x: torch.Tensor, n_passes: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean_prob, uncertainty=predictive variance). Lower U = more certain."""
    if hasattr(model, "mc_predict"):
        mean_p, unc = model.mc_predict(x, n_passes=n_passes)
        return mean_p, unc
    model.eval()
    if hasattr(model, "enable_dropout"):
        model.enable_dropout()
    probs = []
    for _ in range(n_passes):
        p = torch.sigmoid(model(x)).view(-1)
        probs.append(p)
    stacked = torch.stack(probs, dim=0)
    mean_p = stacked.mean(dim=0)
    unc = stacked.var(dim=0, unbiased=False)
    return mean_p, unc
