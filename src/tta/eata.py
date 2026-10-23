"""EATA family with BN-affine-only updates (parity with Tent / UTTA-Med / SAR).

Differentiation (Master Plan v4, frozen):

* **EATA** — entropy filter only (this was the mandatory 5-seed baseline).
* **EATA-C** — EATA + cosine diversity filter (original EATA sample selection
  without Fisher). Distinct from SAR (SAM + reset) and from UTTA (MC-Dropout U).
* **EATA-F** — EATA-C + diagonal Fisher anti-forgetting, if a source Fisher
  dict is provided. Optional; not in the 5-seed headline.

Deviation from the original paper (BN-affine-only) must be stated in Setup.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode


def _features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "backbone"):
        z = model.backbone(x)
        return F.normalize(z.flatten(1), dim=1)
    z = model(x)
    return F.normalize(z.flatten(1), dim=1)


class EATA:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        steps: int = 1,
        e_margin: float = 0.4,
    ):
        self.model = model
        configure_tta_mode(self.model)
        params = collect_bn_affine(model)
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.steps = steps
        self.e_margin = e_margin
        self._anchor = deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.n_adapted = 0
        self.n_seen = 0
        self.n_skipped = 0

    def reset(self) -> None:
        self.model.load_state_dict(self._anchor)
        self.optimizer.state.clear()
        self.n_adapted = 0
        self.n_seen = 0
        self.n_skipped = 0

    def _keep_mask(self, x: torch.Tensor) -> torch.Tensor:
        logits0 = self.model(x)
        ent = binary_entropy_from_logits(logits0)
        return ent < self.e_margin

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        configure_tta_mode(self.model)
        self.n_seen += int(x.size(0))
        with torch.no_grad():
            keep = self._keep_mask(x)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            self.n_skipped += int(x.size(0))
        else:
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                ent = binary_entropy_from_logits(logits)
                loss = ent[keep].mean()
                loss = self._regularize(loss)
                loss.backward()
                self.optimizer.step()
            self.n_adapted += n_keep
            self.n_skipped += int(x.size(0)) - n_keep
        with torch.no_grad():
            self.model.eval()
            return torch.sigmoid(self.model(x)).view(-1)

    def _regularize(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    @property
    def coverage(self) -> float:
        return self.n_adapted / max(self.n_seen, 1)


class EATAC(EATA):
    """EATA + cosine diversity (d_margin). No Fisher."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-5,
        steps: int = 1,
        e_margin: float = 0.4,
        d_margin: float = 0.4,
        memory: int = 64,
    ):
        super().__init__(model, lr=lr, steps=steps, e_margin=e_margin)
        self.d_margin = d_margin
        self.memory = memory
        self._bank: Optional[torch.Tensor] = None

    def reset(self) -> None:
        super().reset()
        self._bank = None

    def _keep_mask(self, x: torch.Tensor) -> torch.Tensor:
        keep = super()._keep_mask(x)
        feat = _features(self.model, x)
        if self._bank is None or self._bank.numel() == 0:
            if keep.any():
                self._bank = feat[keep].detach()
            return keep
        sim = feat @ self._bank.t()  # [B, M]
        redundant = sim.max(dim=1).values > self.d_margin
        keep = keep & (~redundant)
        if keep.any():
            added = feat[keep].detach()
            self._bank = torch.cat([self._bank, added], dim=0)
            if self._bank.size(0) > self.memory:
                self._bank = self._bank[-self.memory :]
        return keep


class EATAF(EATAC):
    """EATA-C + diagonal Fisher anti-forgetting on BN-affine params."""

    def __init__(
        self,
        model: nn.Module,
        fisher: Optional[Dict[str, torch.Tensor]] = None,
        fishers_alpha: float = 2000.0,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.fisher = fisher or {}
        self.fishers_alpha = fishers_alpha
        self._source = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

    def _regularize(self, loss: torch.Tensor) -> torch.Tensor:
        if not self.fisher:
            return loss
        extra = loss.new_zeros(())
        for n, p in self.model.named_parameters():
            if n not in self.fisher or n not in self._source:
                continue
            extra = extra + (self.fisher[n].to(p.device) * (p - self._source[n].to(p.device)).pow(2)).sum()
        return loss + self.fishers_alpha * extra


def estimate_fisher(model: nn.Module, loader, device: torch.device, max_batches: int = 40) -> Dict[str, torch.Tensor]:
    """Diagonal Fisher on BN-affine params from **source-train** unlabeled entropy.

    Uses model predictions as pseudo-labels so target labels are never touched.
    """
    from src.tta.bn import collect_bn_affine

    collect_bn_affine(model)
    model.train()
    fisher: Dict[str, torch.Tensor] = {}
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0].to(device)
        model.zero_grad(set_to_none=True)
        logits = model(x).view(-1)
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        # Bernoulli NLL under model's own p (unlabeled)
        loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean()
        loss.backward()
        for name, p_ in model.named_parameters():
            if p_.grad is None:
                continue
            g2 = p_.grad.detach().pow(2)
            if name not in fisher:
                fisher[name] = g2.clone()
            else:
                fisher[name] += g2
        n += 1
    if n:
        for k in fisher:
            fisher[k] /= n
    return fisher
