"""Coverage-Risk / Reject-Option evaluation module for UTTA-Med.

Separates two distinct scientific workflows:
1. Selective evaluation / reject option: Evaluate performance of source (or adapted)
   predictions when unconfident/uncertain samples are deferred to human triage.
2. Selective adaptation: Evaluate adaptation dynamics (via flip_table) specifically on
   the accepted subset vs the rejected subset.

Ranking Direction:
- ranking_scores: 1D array of scores.
- ascending=True: Lower score = accepted first (e.g., uncertainty where lower is more certain).
- ascending=False: Higher score = accepted first (e.g., confidence where higher is more certain).
Ties are broken deterministically by initial sample index (stable sort).
Ranking NEVER uses ground truth labels y_true.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.adaptation import flip_table
from src.evaluation.classification import binary_metrics, brier, ece


def proxy_uncertainty(p: np.ndarray) -> np.ndarray:
    """Compute proxy uncertainty for binary classification: 1 - |2p - 1|.

    Lower values indicate higher confidence/certainty (p close to 0 or 1).
    Use with ascending=True.
    """
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    return 1.0 - np.abs(2.0 * p_arr - 1.0)


def get_accepted_indices(
    ranking_scores: Sequence[float] | np.ndarray,
    coverage: float,
    ascending: bool = True,
) -> np.ndarray:
    """Return indices of accepted samples based on ranking scores and coverage fraction.

    Guarantees deterministic tie-breaking using the sample's original position.
    Ground truth y_true is never accessed or used.

    Args:
        ranking_scores: 1D array of scores.
        coverage: Target acceptance fraction in (0.0, 1.0].
        ascending: If True, lower scores are accepted first (e.g. uncertainty).
                   If False, higher scores are accepted first (e.g. confidence).

    Returns:
        1D array of integer indices for accepted instances.
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"Coverage must be in (0, 1], got {coverage}")

    scores = np.asarray(ranking_scores, dtype=float).reshape(-1)
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot rank empty score array.")

    k = max(1, int(round(n * coverage)))
    if k >= n or coverage == 1.0:
        return np.arange(n)

    indices = np.arange(n)
    if ascending:
        # Secondary key: indices (ascending). Primary key: scores (ascending).
        order = np.lexsort((indices, scores))
    else:
        # Secondary key: indices (ascending). Primary key: -scores (ascending, so scores descending).
        order = np.lexsort((indices, -scores))

    return order[:k]


def compute_coverage_slice(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    ranking_scores: Sequence[float] | np.ndarray,
    coverage: float,
    ascending: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Selective evaluation / reject-option metrics on the accepted subset.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted positive class probabilities.
        ranking_scores: Scores used to rank samples (without seeing y_true).
        coverage: Target acceptance fraction in (0.0, 1.0].
        ascending: True if lower ranking_scores mean accept first; False otherwise.
        threshold: Decision threshold for discrete classification metrics.

    Returns:
        Dictionary containing accepted sample metrics (accuracy, error_rate, AUROC, etc.).
        Rejected samples NEVER contribute to accepted metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)
    scores_arr = np.asarray(ranking_scores, dtype=float).reshape(-1)

    n_total = len(y_true_arr)
    if n_total == 0:
        raise ValueError("Cannot evaluate on empty arrays.")
    if len(y_prob_arr) != n_total or len(scores_arr) != n_total:
        raise ValueError(
            f"Array length mismatch: y_true={n_total}, y_prob={len(y_prob_arr)}, scores={len(scores_arr)}"
        )

    accepted_idx = get_accepted_indices(scores_arr, coverage=coverage, ascending=ascending)
    y_acc = y_true_arr[accepted_idx]
    p_acc = y_prob_arr[accepted_idx]

    metrics = binary_metrics(y_acc, p_acc, threshold=threshold)
    k = len(accepted_idx)

    metrics["coverage"] = float(coverage)
    metrics["effective_coverage"] = float(k / n_total)
    metrics["n_accepted"] = int(k)
    metrics["n_total"] = int(n_total)
    metrics["error_rate"] = float(1.0 - metrics["accuracy"])
    metrics["brier"] = brier(y_acc, p_acc)
    metrics["ece"] = ece(y_acc, p_acc)

    return metrics


def compute_selective_adaptation_slice(
    y_true: Sequence[int] | np.ndarray,
    p_before: Sequence[float] | np.ndarray,
    p_after: Sequence[float] | np.ndarray,
    ranking_scores: Sequence[float] | np.ndarray,
    coverage: float,
    ascending: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Selective adaptation analysis evaluating adaptation behavior on the accepted subset.

    Uses flip_table to track correction, harm, and stability on the accepted
    subset versus the rejected subset.
    """
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    p_before_arr = np.asarray(p_before, dtype=float).reshape(-1)
    p_after_arr = np.asarray(p_after, dtype=float).reshape(-1)
    scores_arr = np.asarray(ranking_scores, dtype=float).reshape(-1)

    n_total = len(y_true_arr)
    if n_total == 0:
        raise ValueError("Cannot evaluate on empty arrays.")
    if len(p_before_arr) != n_total or len(p_after_arr) != n_total or len(scores_arr) != n_total:
        raise ValueError("Array length mismatch among inputs.")

    accepted_idx = get_accepted_indices(scores_arr, coverage=coverage, ascending=ascending)
    mask = np.zeros(n_total, dtype=bool)
    mask[accepted_idx] = True
    rejected_idx = np.where(~mask)[0]

    # Accepted flip analysis
    y_acc = y_true_arr[accepted_idx]
    p_bef_acc = p_before_arr[accepted_idx]
    p_aft_acc = p_after_arr[accepted_idx]
    acc_flip = flip_table(y_acc, p_bef_acc, p_aft_acc, threshold=threshold)

    acc_before_metrics = binary_metrics(y_acc, p_bef_acc, threshold=threshold)
    acc_after_metrics = binary_metrics(y_acc, p_aft_acc, threshold=threshold)

    out: Dict[str, Any] = {
        "coverage": float(coverage),
        "n_accepted": int(len(accepted_idx)),
        "n_rejected": int(len(rejected_idx)),
        "accepted_flip": acc_flip,
        "accepted_before": acc_before_metrics,
        "accepted_after": acc_after_metrics,
        "delta_accuracy": float(acc_after_metrics["accuracy"] - acc_before_metrics["accuracy"]),
    }
    if not (np.isnan(acc_after_metrics["auroc"]) or np.isnan(acc_before_metrics["auroc"])):
        out["delta_auroc"] = float(acc_after_metrics["auroc"] - acc_before_metrics["auroc"])
    else:
        out["delta_auroc"] = float("nan")

    # If any rejected, compute rejected flip dynamics
    if len(rejected_idx) > 0:
        y_rej = y_true_arr[rejected_idx]
        p_bef_rej = p_before_arr[rejected_idx]
        p_aft_rej = p_after_arr[rejected_idx]
        out["rejected_flip"] = flip_table(y_rej, p_bef_rej, p_aft_rej, threshold=threshold)
    else:
        out["rejected_flip"] = None

    return out


def sweep_coverage_risk(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    ranking_scores: Optional[Sequence[float] | np.ndarray] = None,
    coverages: Optional[Sequence[float]] = None,
    ascending: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Compute selective evaluation / reject-option metrics across a sweep of coverages.

    Default:
        If ranking_scores is None, proxy_uncertainty(y_prob) is computed and ascending=True.
        Default coverages: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3].
    """
    if ranking_scores is None:
        ranking_scores = proxy_uncertainty(np.asarray(y_prob))
        ascending = True

    if coverages is None:
        coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    out = {}
    for cov in coverages:
        key = f"{float(cov):.2f}"
        out[key] = compute_coverage_slice(
            y_true=y_true,
            y_prob=y_prob,
            ranking_scores=ranking_scores,
            coverage=float(cov),
            ascending=ascending,
            threshold=threshold,
        )
    return out


def sweep_selective_adaptation(
    y_true: Sequence[int] | np.ndarray,
    p_before: Sequence[float] | np.ndarray,
    p_after: Sequence[float] | np.ndarray,
    ranking_scores: Sequence[float] | np.ndarray,
    coverages: Optional[Sequence[float]] = None,
    ascending: bool = True,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, Any]]:
    """Compute selective adaptation dynamics (flip table) across a sweep of coverages."""
    if coverages is None:
        coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    out = {}
    for cov in coverages:
        key = f"{float(cov):.2f}"
        out[key] = compute_selective_adaptation_slice(
            y_true=y_true,
            p_before=p_before,
            p_after=p_after,
            ranking_scores=ranking_scores,
            coverage=float(cov),
            ascending=ascending,
            threshold=threshold,
        )
    return out
