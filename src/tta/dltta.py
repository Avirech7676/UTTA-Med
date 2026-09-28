"""DLTTA-style dynamic-LR entropy minimization (Yang et al., TMI 2022, simplified).

Official DLTTA uses a teacher–student discrepancy to scale the test-time
learning rate. This implementation keeps **BN-affine update-scope parity**
and scales Adam lr per batch by a clipped function of mean predictive entropy.

This is an optional medical-TTA baseline, not a mandatory control.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn

from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode


class DLTTA:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        steps: int = 1,
        lr_min: float = 1e-6,
        lr_max: float = 5e-5,
    ):
        self.model = model
        configure_tta_mode(self.model)
        params = collect_bn_affine(model)
        self.base_lr = lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.steps = steps
        self._anchor = deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.n_adapted = 0
        self.n_seen = 0
        self.last_lr = lr

    def reset(self) -> None:
        self.model.load_state_dict(self._anchor)
        self.optimizer.state.clear()
        self.n_adapted = 0
        self.n_seen = 0

    def _set_lr(self, lr: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.last_lr = lr

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        configure_tta_mode(self.model)
        self.n_seen += int(x.size(0))
        with torch.no_grad():
            ent0 = binary_entropy_from_logits(self.model(x)).mean()
        # Low entropy → smaller step (already confident); high entropy → larger step.
        # H_binary max is ln(2) ≈ 0.693.
        scale = float(ent0.clamp(0, 0.693).item() / 0.693)
        lr = self.lr_min + (self.lr_max - self.lr_min) * scale
        self._set_lr(lr)
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            loss = binary_entropy_from_logits(self.model(x)).mean()
            loss.backward()
            self.optimizer.step()
        self.n_adapted += int(x.size(0))
        with torch.no_grad():
            self.model.eval()
            return torch.sigmoid(self.model(x)).view(-1)

    @property
    def coverage(self) -> float:
        return 1.0
