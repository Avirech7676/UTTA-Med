#!/usr/bin/env python3
"""Shared helpers for Phase-1 offline analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

SEEDS: List[int] = [42, 123, 2024, 7, 99]
METHODS: List[str] = [
    "source",
    "tent",
    "eata",
    "random",
    "confidence",
    "entropy",
    "utta",
]
GATED = ["confidence", "entropy", "utta"]
COLLAPSE_SEEDS = [7, 99]

# Resolve project root: scripts/ -> repo root
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR.parent.parent
NPZ_DIR = ROOT / "results" / "data" / "wsi_bootstrap"
OUT_STAT = ROOT / "results" / "statistical"
OUT_TABLES = ROOT / "results" / "tables"
OUT_FIG = ROOT / "figures" / "phase1"


def ensure_dirs() -> None:
    OUT_STAT.mkdir(parents=True, exist_ok=True)
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    if y.min() == y.max():
        return float("nan")
    order = np.argsort(-p)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    tpr = tp / float(tp[-1])
    fpr = fp / float(fp[-1])
    return float(np.trapezoid(tpr, fpr))


def f1_score(y: np.ndarray, p: np.ndarray, t: float = 0.5) -> float:
    pred = (p >= t).astype(int)
    y = y.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp + 1e-12)
    rec = tp / (tp + fn + 1e-12)
    return float(2 * prec * rec / (prec + rec + 1e-12))


def error_rate(y: np.ndarray, p: np.ndarray, t: float = 0.5) -> float:
    pred = (p >= t).astype(int)
    return float((pred != y.astype(int)).mean())


def mean_sd(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(list(xs), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), 0.0
    return float(a.mean()), float(a.std(ddof=1))


def load(seed: int, method: str) -> Dict[str, np.ndarray]:
    path = NPZ_DIR / f"s{seed}_{method}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\n"
            f"Expected dumps under {NPZ_DIR}\n"
            "Keys required: y, p, p_before, slide"
        )
    z = np.load(path)
    return {
        "y": np.asarray(z["y"]),
        "p": np.asarray(z["p"]),
        "p_before": np.asarray(z["p_before"]),
        "slide": np.asarray(z["slide"]),
    }


def proxy_uncertainty(p: np.ndarray) -> np.ndarray:
    """Binary proxy for U when MC variance is not stored in the dump.

    U_proxy = 1 - |2p - 1|  (max at p=0.5, min at extremes).
    For coverage–risk ranking this matches confidence/entropy order on binary tasks.
    """
    return 1.0 - np.abs(2.0 * p - 1.0)


def try_plot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None
