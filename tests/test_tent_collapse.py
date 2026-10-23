from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.tent_collapse import (
    binary_entropy,
    class_specific_stats,
    confidence,
    confidence_correctness_bins,
    entropy_confidence_stats,
    prediction_distribution_stats,
    slide_level_collapse_breakdown,
)


def test_binary_entropy():
    p = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
    ent = binary_entropy(p)
    assert np.isclose(ent[0], 0.0, atol=1e-5)
    assert np.isclose(ent[1], 1.0)
    assert np.isclose(ent[2], 0.0, atol=1e-5)
    assert np.isclose(ent[3], ent[4])  # Symmetry


def test_confidence():
    p = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
    conf = confidence(p)
    assert np.isclose(conf[0], 1.0)
    assert np.isclose(conf[1], 0.5)
    assert np.isclose(conf[2], 1.0)
    assert np.isclose(conf[3], 0.8)
    assert np.isclose(conf[4], 0.8)


def test_prediction_distribution_stats():
    p = np.array([0.005, 0.02, 0.1, 0.96])
    stats = prediction_distribution_stats(p)
    assert stats["n"] == 4
    assert np.isclose(stats["frac_p_lt_001"], 0.25)
    assert np.isclose(stats["frac_p_lt_005"], 0.50)
    assert np.isclose(stats["frac_p_gt_095"], 0.25)
    assert np.isclose(stats["positive_rate"], 0.25)


def test_class_specific_stats():
    # 2 tumor (p=0.9, 0.2), 2 non-tumor (p=0.01, 0.02)
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.2, 0.01, 0.02])
    res = class_specific_stats(y, p)

    assert res["n_tumor"] == 2
    assert res["n_non_tumor"] == 2
    assert np.isclose(res["tumor"]["sensitivity"], 0.5)
    assert np.isclose(res["tumor"]["mean_p"], 0.55)
    assert np.isclose(res["non_tumor"]["specificity"], 1.0)
    assert np.isclose(res["non_tumor"]["mean_p"], 0.015)


def test_entropy_confidence_stats():
    p = np.array([0.99, 0.01, 0.5])
    ec = entropy_confidence_stats(p, high_conf_thresh=0.95)
    assert np.isclose(ec["frac_high_confidence"], 2 / 3)
    assert ec["mean_confidence"] > 0.8


def test_confidence_correctness_bins():
    # 4 samples:
    # 2 high confidence correct (p=0.95 y=1, p=0.05 y=0)
    # 2 high confidence WRONG (p=0.95 y=0, p=0.05 y=1)
    y = np.array([1, 0, 0, 1])
    p = np.array([0.95, 0.05, 0.95, 0.05])
    bins = confidence_correctness_bins(y, p, conf_bins=[0.5, 0.8, 1.0])

    # All 4 are in [0.8, 1.0], 2 correct and 2 wrong -> accuracy 0.5
    assert len(bins) == 2
    b_high = bins[1]
    assert b_high["count"] == 4
    assert np.isclose(b_high["accuracy"], 0.5)
    assert np.isclose(b_high["error_rate"], 0.5)


def test_slide_level_collapse_breakdown():
    # 1 slide, 4 patches
    y = np.array([1, 1, 0, 0])
    p_b = np.array([0.9, 0.8, 0.1, 0.2])  # Good before
    p_a = np.array([0.01, 0.02, 0.01, 0.01])  # Collapsed after
    slides = np.array([29, 29, 29, 29])

    bd = slide_level_collapse_breakdown(y, p_b, p_a, slides)
    assert len(bd) == 1
    row = bd[0]
    assert row["slide"] == 29
    assert row["n_patches"] == 4
    assert row["harm_count"] == 2  # The 2 tumor patches were flipped from correct to wrong
    assert np.isclose(row["after_sensitivity"], 0.0)
    assert np.isclose(row["delta_sensitivity"], -1.0)
