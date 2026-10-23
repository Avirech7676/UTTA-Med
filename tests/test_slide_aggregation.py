from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.slide_aggregation import (
    slide_burden_metrics,
    slide_macro_metrics,
    slide_tumor_burdens,
)


def test_slide_tumor_burdens_basic():
    # 2 slides:
    # slide 1 has 4 patches, 2 positive -> burden 0.5
    # slide 2 has 4 patches, 1 positive -> burden 0.25
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.1, 0.2, 0.3])
    slides = np.array(["s1", "s1", "s1", "s1", "s2", "s2", "s2", "s2"])

    burdens = slide_tumor_burdens(y, p, slides)
    assert len(burdens) == 2
    assert np.isclose(burdens["s1"]["burden_true"], 0.5)
    assert np.isclose(burdens["s1"]["burden_pred_binary"], 0.5)
    assert np.isclose(burdens["s2"]["burden_true"], 0.25)
    assert np.isclose(burdens["s2"]["burden_pred_binary"], 0.25)


def test_slide_burden_metrics_perfect():
    y = np.array([1, 0, 1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.9, 0.9, 0.1, 0.1])
    # 3 slides with varying burdens: 0.5, 1.0, 0.0
    slides = np.array([1, 1, 2, 2, 3, 3])

    m = slide_burden_metrics(y, p, slides)
    assert m["n_slides"] == 3
    assert np.isclose(m["mae_binary"], 0.0)
    assert np.isclose(m["pearson_r_binary"], 1.0)
    assert np.isclose(m["spearman_rho_binary"], 1.0)


def test_slide_burden_metrics_collapse_simulation():
    # True burdens across 3 slides: 0.8, 0.5, 0.1
    y = np.array([1, 1, 1, 1, 0,  1, 1, 0, 0, 0,  1, 0, 0, 0, 0])
    slides = np.array([1, 1, 1, 1, 1,  2, 2, 2, 2, 2,  3, 3, 3, 3, 3])
    # Collapsed predictions (predicts near-zero everywhere)
    p_collapsed = np.full(len(y), 0.01)

    m = slide_burden_metrics(y, p_collapsed, slides)
    assert m["mae_binary"] > 0.4
    assert np.isclose(m["pearson_r_binary"], 0.0) or np.isnan(m["pearson_r_binary"])


def test_slide_macro_metrics():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    slides = np.array([1, 1, 2, 2])

    macro = slide_macro_metrics(y, p, slides)
    assert macro["n_evaluated_slides"] == 2
    assert np.isclose(macro["macro_auroc"], 1.0)
    assert np.isclose(macro["macro_accuracy"], 1.0)


def test_slide_aggregation_length_mismatch():
    with pytest.raises(ValueError, match="Array length mismatch"):
        slide_tumor_burdens([1, 0], [0.5], [1, 2])
