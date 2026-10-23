"""WSI-level bootstrap CIs. Primary inference must resample slides, not patches."""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


def bootstrap_ci_by_slide(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slide_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    slide_ids = np.asarray(slide_ids).reshape(-1)
    rng = np.random.default_rng(seed)

    unique = np.unique(slide_ids)
    groups = {int(s): np.where(slide_ids == s)[0] for s in unique}
    keys = np.array(list(groups.keys()))

    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[int(s)] for s in sampled])
        stats[i] = metric_fn(y_true[idx], y_prob[idx])

    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1.0 - alpha / 2))
    return {
        "mean": float(np.mean(stats)),
        "std": float(np.std(stats, ddof=1)),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": n_boot,
        "n_slides": int(len(keys)),
    }
