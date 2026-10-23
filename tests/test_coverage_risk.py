from __future__ import annotations

import sys
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.coverage_risk import (
    compute_coverage_slice,
    compute_selective_adaptation_slice,
    get_accepted_indices,
    proxy_uncertainty,
    sweep_coverage_risk,
    sweep_selective_adaptation,
)


def test_proxy_uncertainty():
    p = np.array([0.0, 1.0, 0.5, 0.2, 0.8])
    u = proxy_uncertainty(p)
    assert np.isclose(u[0], 0.0)
    assert np.isclose(u[1], 0.0)
    assert np.isclose(u[2], 1.0)
    assert np.isclose(u[3], 0.4)
    assert np.isclose(u[4], 0.4)


def test_ranking_direction_explicit():
    # 4 samples with uncertainty scores [0.1, 0.4, 0.2, 0.8]
    # Corresponding confidence scores [0.9, 0.6, 0.8, 0.2]
    uncertainty = np.array([0.1, 0.4, 0.2, 0.8])
    confidence = 1.0 - uncertainty

    # Top 50% (2 samples): should be samples 0 (u=0.1) and 2 (u=0.2)
    idx_u = get_accepted_indices(uncertainty, coverage=0.5, ascending=True)
    idx_c = get_accepted_indices(confidence, coverage=0.5, ascending=False)

    np.testing.assert_array_equal(idx_u, [0, 2])
    np.testing.assert_array_equal(idx_c, [0, 2])


def test_deterministic_tie_breaking():
    # 5 tied scores
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    # 60% coverage (3 samples) -> must pick indices [0, 1, 2] deterministically
    idx = get_accepted_indices(scores, coverage=0.6, ascending=True)
    np.testing.assert_array_equal(idx, [0, 1, 2])

    idx_desc = get_accepted_indices(scores, coverage=0.6, ascending=False)
    np.testing.assert_array_equal(idx_desc, [0, 1, 2])


def test_coverage_1_returns_complete_dataset():
    scores = np.random.default_rng(0).random(25)
    idx = get_accepted_indices(scores, coverage=1.0, ascending=True)
    assert len(idx) == 25
    np.testing.assert_array_equal(idx, np.arange(25))


def test_rejected_samples_never_contribute_to_accepted_metrics():
    # 4 samples: 2 certain & correct, 2 uncertain
    y_true = np.array([1, 0, 1, 0])
    p_good = np.array([0.95, 0.05, 0.6, 0.4])
    uncertainty = np.array([0.1, 0.1, 0.8, 0.8])

    # Slice at coverage=0.5 (keeps the 2 certain correct samples)
    m1 = compute_coverage_slice(y_true, p_good, uncertainty, coverage=0.5, ascending=True)
    assert m1["n_accepted"] == 2
    assert m1["accuracy"] == 1.0
    assert m1["error_rate"] == 0.0

    # Modify rejected samples (samples 2 & 3) to extreme wrong predictions
    p_corrupted = np.array([0.95, 0.05, 0.01, 0.99])
    m2 = compute_coverage_slice(y_true, p_corrupted, uncertainty, coverage=0.5, ascending=True)

    # Accepted metrics must be 100% identical!
    assert m2["accuracy"] == m1["accuracy"]
    assert m2["error_rate"] == m1["error_rate"]
    assert m2["brier"] == m1["brier"]


def test_coverage_slice_validation():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    u = np.array([0.2, 0.2])

    with pytest.raises(ValueError, match="Coverage must be in"):
        compute_coverage_slice(y, p, u, coverage=0.0)

    with pytest.raises(ValueError, match="Coverage must be in"):
        compute_coverage_slice(y, p, u, coverage=1.5)

    with pytest.raises(ValueError, match="Cannot evaluate on empty"):
        compute_coverage_slice([], [], [], coverage=0.5)

    with pytest.raises(ValueError, match="Array length mismatch"):
        compute_coverage_slice(y, p, u[:1], coverage=0.5)


def test_compute_selective_adaptation_slice():
    # 4 samples:
    # Sample 0: correct before, correct after (stable)
    # Sample 1: wrong before, correct after (corrected)
    # Sample 2: correct before, wrong after (harmed, but in rejected set)
    # Sample 3: wrong before, wrong after (persistent, in rejected set)
    y_true = np.array([1, 1, 0, 0])
    p_before = np.array([0.9, 0.2, 0.1, 0.8])
    p_after = np.array([0.95, 0.85, 0.9, 0.9])
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9])

    # Accept top 50% (samples 0 and 1)
    res = compute_selective_adaptation_slice(
        y_true=y_true,
        p_before=p_before,
        p_after=p_after,
        ranking_scores=uncertainty,
        coverage=0.5,
        ascending=True,
    )
    assert res["n_accepted"] == 2
    assert res["n_rejected"] == 2
    assert res["accepted_flip"]["correction_count"] == 1
    assert res["accepted_flip"]["harm_count"] == 0
    assert res["delta_accuracy"] == 0.5  # Acc went from 0.5 to 1.0 on accepted

    # Check rejected flip
    assert res["rejected_flip"]["harm_count"] == 1


def test_sweeps_coverage_and_adaptation():
    y = np.array([1, 1, 0, 0])
    p_b = np.array([0.8, 0.7, 0.3, 0.2])
    p_a = np.array([0.9, 0.8, 0.2, 0.1])
    u = np.array([0.1, 0.2, 0.3, 0.4])

    cov_sweep = sweep_coverage_risk(y, p_b, u, coverages=[1.0, 0.5])
    assert "1.00" in cov_sweep and "0.50" in cov_sweep
    assert cov_sweep["1.00"]["n_accepted"] == 4
    assert cov_sweep["0.50"]["n_accepted"] == 2

    adapt_sweep = sweep_selective_adaptation(y, p_b, p_a, u, coverages=[1.0, 0.5])
    assert "1.00" in adapt_sweep and "0.50" in adapt_sweep
    assert adapt_sweep["1.00"]["n_accepted"] == 4
    assert adapt_sweep["0.50"]["n_accepted"] == 2
