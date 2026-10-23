"""Post-hoc temperature scaling (Guo et al., 2017).

Fit a single scalar T on **OOD validation (hospital 1) only**.
Never fit on the target test hospital. AUROC is invariant to T for binary
sigmoid scores (monotonic); ECE / Brier / NLL can change.

This is optional calibration, not a TTA method.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_logits(logits_or_prob: torch.Tensor) -> torch.Tensor:
    x = logits_or_prob.reshape(-1).float()
    # Heuristic: values already in (0, 1) are treated as probabilities.
    if torch.all((x > 0) & (x < 1)) and x.max() <= 1.0:
        p = x.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(p) - torch.log(1.0 - p)
    return x


class TemperatureScaler:
    """Scalar temperature T > 0.  p = sigmoid(logit / T)."""

    def __init__(self, init_t: float = 1.0):
        self.temperature = float(init_t)

    def fit(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 200,
        device: Optional[torch.device] = None,
    ) -> "TemperatureScaler":
        z = _to_logits(logits.detach())
        y = y.reshape(-1).float()
        if device is None:
            device = z.device
        z = z.to(device)
        y = y.to(device)
        t = nn.Parameter(torch.ones(1, device=device) * self.temperature)
        opt = torch.optim.LBFGS([t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            # softplus keeps T > 0
            temp = F.softplus(t) + 1e-3
            loss = F.binary_cross_entropy_with_logits(z / temp, y)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.temperature = float((F.softplus(t) + 1e-3).item())
        return self

    def transform_logits(self, logits: torch.Tensor) -> torch.Tensor:
        z = _to_logits(logits)
        return z / max(self.temperature, 1e-6)

    def transform_prob(self, logits_or_prob: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.transform_logits(logits_or_prob))

    def summary(self) -> Dict[str, float]:
        return {"temperature": float(self.temperature)}


def fit_temperature_on_val(model: nn.Module, loader, device: torch.device) -> TemperatureScaler:
    """Collect hospital-1 logits and labels, fit T. Target labels never used."""
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            z = model(x).view(-1)
            zs.append(z.cpu())
            ys.append(y.view(-1).float().cpu())
    scaler = TemperatureScaler()
    scaler.fit(torch.cat(zs), torch.cat(ys), device=torch.device("cpu"))
    return scaler


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    if np.all((z > 0) & (z < 1)) and z.max() <= 1.0:
        p = np.clip(z, 1e-6, 1.0 - 1e-6)
        z = np.log(p) - np.log(1.0 - p)
    return 1.0 / (1.0 + np.exp(-(z / max(t, 1e-6))))
