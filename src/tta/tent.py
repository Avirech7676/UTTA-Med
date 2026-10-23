"""Tent: entropy minimization on BatchNorm affine parameters only.

Official for this project: Adam lr=1e-5, bn_freeze_stats.
Original Tent paper lr=1e-3 / BN.train() collapsed here.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode

OFFICIAL_TTA_LR = 1e-5


class Tent:
    def __init__(
        self,
        model: nn.Module,
        lr: float = OFFICIAL_TTA_LR,
        steps: int = 1,
        bn_mode: str = "bn_freeze_stats",
    ):
        self.model = model
        self.bn_mode = bn_mode
        configure_tta_mode(self.model, bn_mode=self.bn_mode)
        params = collect_bn_affine(model)
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.steps = steps
        self._anchor = deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.n_adapted = 0
        self.n_seen = 0

    def reset(self) -> None:
        self.model.load_state_dict(self._anchor)
        self.optimizer.state.clear()
        self.n_adapted = 0
        self.n_seen = 0

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        configure_tta_mode(self.model, bn_mode=self.bn_mode)
        self.n_seen += int(x.size(0))
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = binary_entropy_from_logits(logits).mean()
            loss.backward()
            self.optimizer.step()
        self.n_adapted += int(x.size(0))
        with torch.no_grad():
            self.model.eval()
            return torch.sigmoid(self.model(x)).view(-1)
