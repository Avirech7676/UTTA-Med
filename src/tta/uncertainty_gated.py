"""Uncertainty-gated TTA (UTTA-Med) and shared gated adapter.

Zero-accepted batches are skipped entirely (no backward, no optimizer step).
Official TTA: Adam lr=1e-5, bn_freeze_stats. τ = 70th percentile of val U
(see configs/tau_by_seed.yaml). Do NOT use tau=0.05 for MC-Dropout.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn

from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode
from src.tta.gates import score_confidence, score_entropy, score_mc_dropout

OFFICIAL_TTA_LR = 1e-5


class GatedTTA:
    def __init__(
        self,
        model: nn.Module,
        gate: str = "mc_dropout",
        tau: Optional[float] = None,
        lr: float = OFFICIAL_TTA_LR,
        steps: int = 1,
        n_passes: int = 20,
        seed: int = 42,
        bn_mode: str = "bn_freeze_stats",
    ):
        self.model = model
        self.gate = gate
        self.tau = tau
        self.n_passes = n_passes
        self.bn_mode = bn_mode
        configure_tta_mode(self.model, bn_mode=self.bn_mode)
        params = collect_bn_affine(model)
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.steps = steps
        self._anchor = deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        self.n_adapted = 0
        self.n_seen = 0
        self.n_skipped_batches = 0
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    def reset(self) -> None:
        self.model.load_state_dict(self._anchor)
        self.optimizer.state.clear()
        self.n_adapted = 0
        self.n_seen = 0
        self.n_skipped_batches = 0

    def _keep_mask(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate == "mc_dropout":
            if self.tau is None:
                raise ValueError(
                    "UTTA-Med tau must be the 70th percentile of hospital-1 U "
                    "(configs/tau_by_seed.yaml). Do not use 0.05."
                )
            _, unc = score_mc_dropout(self.model, x, n_passes=self.n_passes)
            return unc < float(self.tau)
        if self.gate == "entropy":
            # score_entropy returns -H; accept if H < tau
            return (-score_entropy(self.model, x)) < float(self.tau)
        if self.gate == "confidence":
            return score_confidence(self.model, x) > float(self.tau)
        if self.gate == "random":
            cov = 0.5 if self.tau is None else float(self.tau)
            r = torch.rand(x.size(0), generator=self._rng)
            return r.to(x.device) < cov
        raise ValueError(f"Unknown gate {self.gate}")

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        self.n_seen += int(x.size(0))
        with torch.no_grad():
            keep = self._keep_mask(x)
        n_keep = int(keep.sum().item())
        if n_keep == 0:
            self.n_skipped_batches += 1
        else:
            configure_tta_mode(self.model, bn_mode=self.bn_mode)
            w = keep.float()
            for _ in range(self.steps):
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                ent = binary_entropy_from_logits(logits)
                loss = (w * ent).sum() / w.sum()
                loss.backward()
                self.optimizer.step()
            self.n_adapted += n_keep
        with torch.no_grad():
            self.model.eval()
            return torch.sigmoid(self.model(x)).view(-1)

    @property
    def coverage(self) -> float:
        return self.n_adapted / max(self.n_seen, 1)


def build_adapter(model: nn.Module, method: str, **kwargs):
    method = method.lower()
    lr = kwargs.get("lr", OFFICIAL_TTA_LR)
    steps = kwargs.get("steps", 1)
    bn_mode = kwargs.get("bn_mode", "bn_freeze_stats")
    if method == "tent":
        from src.tta.tent import Tent

        return Tent(model, lr=lr, steps=steps, bn_mode=bn_mode)
    if method == "eata":
        from src.tta.eata import EATA

        return EATA(model, lr=lr, steps=steps, e_margin=kwargs.get("e_margin", 0.4))
    if method in ("eata_c", "eatac"):
        from src.tta.eata import EATAC

        return EATAC(
            model,
            lr=lr,
            e_margin=kwargs.get("e_margin", 0.4),
            d_margin=kwargs.get("d_margin", 0.4),
        )
    if method in ("eata_f", "eataf"):
        from src.tta.eata import EATAF

        return EATAF(model, lr=lr, e_margin=kwargs.get("e_margin", 0.4))
    if method == "sar":
        from src.tta.sar import SAR

        return SAR(model, lr=lr, steps=steps, e_margin=kwargs.get("e_margin", 0.4))
    if method == "dltta":
        from src.tta.dltta import DLTTA

        return DLTTA(model, lr=lr, steps=steps)
    gate_map = {
        "utta": "mc_dropout",
        "utta_med": "mc_dropout",
        "mc_dropout": "mc_dropout",
        "random": "random",
        "confidence": "confidence",
        "entropy": "entropy",
    }
    if method in gate_map:
        return GatedTTA(
            model,
            gate=gate_map[method],
            tau=kwargs.get("tau"),
            lr=lr,
            steps=steps,
            n_passes=kwargs.get("n_passes", 20),
            seed=kwargs.get("seed", 42),
            bn_mode=bn_mode,
        )
    raise ValueError(f"Unknown method {method}")
