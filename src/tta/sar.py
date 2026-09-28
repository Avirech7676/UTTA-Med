"""SAR — Sharpness-Aware TTA (Niu et al., ICLR 2023).

Distinct from EATA (entropy + diversity filter) and from EATA-C (this repo:
entropy + cosine redundancy filter). SAR adds:
  1. entropy filter (skip unreliable samples)
  2. SAM (two-step sharpness-aware) update on BN-affine params
  3. reset to source if an EMA of entropy exceeds a recovery threshold

Update-scope parity with Tent / EATA / UTTA-Med: BN-affine only.
"""

from __future__ import annotations

from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode


class _SAM:
    """Minimal SAM wrapper around Adam on a frozen parameter list."""

    def __init__(self, params: List[nn.Parameter], lr: float, rho: float = 0.05):
        self.params = [p for p in params if p.requires_grad]
        self.rho = rho
        self.opt = torch.optim.Adam(self.params, lr=lr)
        self._eps: dict[int, torch.Tensor] = {}

    def _grad_norm(self) -> torch.Tensor:
        norms = [p.grad.norm(p=2) for p in self.params if p.grad is not None]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), p=2)

    def first_step(self) -> None:
        scale = self.rho / (self._grad_norm() + 1e-12)
        self._eps.clear()
        for p in self.params:
            if p.grad is None:
                continue
            eps = p.grad * scale
            self._eps[id(p)] = eps.detach().clone()
            p.data.add_(eps)

    def second_step(self) -> None:
        for p in self.params:
            eps = self._eps.get(id(p))
            if eps is not None:
                p.data.sub_(eps)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self._eps.clear()

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)


class SAR:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        steps: int = 1,
        e_margin: float = 0.4,
        rho: float = 0.05,
        reset_constant: float = 0.2,
    ):
        self.model = model
        configure_tta_mode(self.model)
        params = collect_bn_affine(model)
        self.optimizer = _SAM(params, lr=lr, rho=rho)
        self.steps = steps
        self.e_margin = e_margin
        self.reset_constant = reset_constant
        self._anchor = deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.ema_ent: float | None = None
        self.n_adapted = 0
        self.n_seen = 0
        self.n_reset = 0
        self.n_skipped = 0

    def reset(self) -> None:
        self.model.load_state_dict(self._anchor)
        self.optimizer.opt.state.clear()
        self.ema_ent = None
        self.n_reset += 1

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        configure_tta_mode(self.model)
        self.n_seen += int(x.size(0))
        with torch.no_grad():
            ent0 = binary_entropy_from_logits(self.model(x))
            keep = ent0 < self.e_margin
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            self.n_skipped += int(x.size(0))
        else:
            for _ in range(self.steps):
                self.optimizer.zero_grad()
                ent = binary_entropy_from_logits(self.model(x))
                loss = ent[keep].mean()
                loss.backward()
                self.optimizer.first_step()
                ent2 = binary_entropy_from_logits(self.model(x))
                loss2 = ent2[keep].mean()
                loss2.backward()
                self.optimizer.second_step()
                val = float(loss2.detach().item())
                self.ema_ent = val if self.ema_ent is None else 0.9 * self.ema_ent + 0.1 * val
                if self.ema_ent is not None and self.ema_ent > self.reset_constant:
                    self.reset()
                    break
            self.n_adapted += n_keep
            self.n_skipped += int(x.size(0)) - n_keep
        with torch.no_grad():
            self.model.eval()
            return torch.sigmoid(self.model(x)).view(-1)

    @property
    def coverage(self) -> float:
        return self.n_adapted / max(self.n_seen, 1)
