from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.classification import binary_metrics, brier, ece
from src.evaluation.statistics import bootstrap_ci_by_slide


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    m = binary_metrics(y, p)
    assert m["accuracy"] == 1.0
    assert m["auroc"] == 1.0
    assert brier(y, p) == 0.0
    assert ece(y, p) == 0.0


def test_wsi_bootstrap_runs():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    p = rng.random(50)
    slides = np.repeat(np.arange(10), 5)
    out = bootstrap_ci_by_slide(
        y, p, slides, lambda yt, yp: float((yt == (yp >= 0.5)).mean()), n_boot=20
    )
    assert "ci_low" in out and out["n_slides"] == 10
    assert out["ci_low"] <= out["ci_high"]
