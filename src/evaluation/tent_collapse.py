"""Tent Collapse Mechanism Analysis for UTTA-Med.

Provides fine-grained diagnostic analytics to characterize failure modes of
unconstrained test-time entropy minimization (Tent) on Camelyon17:
- Prediction distribution shifts (suppression vs inflation)
- Class-specific metrics (tumor sensitivity collapse vs non-tumor specificity)
- Slide-level damage localization across WSIs
- Entropy and confidence dynamics
- Flip table decomposition (Correct -> Wrong harm quantification)
- Confidence-vs-correctness calibration across empirical bins
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix

from src.evaluation.adaptation import flip_table
from src.evaluation.classification import binary_metrics


def binary_entropy(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute binary Shannon entropy in bits: - (p log2 p + (1-p) log2 (1-p))."""
    p_arr = np.clip(np.asarray(p, dtype=float).reshape(-1), 1e-15, 1.0 - 1e-15)
    return -(p_arr * np.log2(p_arr) + (1.0 - p_arr) * np.log2(1.0 - p_arr))


def confidence(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """Compute confidence score: max(p, 1 - p) in [0.5, 1.0]."""
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    return np.maximum(p_arr, 1.0 - p_arr)


def prediction_distribution_stats(
    p: Sequence[float] | np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute descriptive statistics of the prediction probability distribution."""
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    n = len(p_arr)
    if n == 0:
        return {}

    return {
        "n": int(n),
        "mean_p": float(np.mean(p_arr)),
        "median_p": float(np.median(p_arr)),
        "std_p": float(np.std(p_arr, ddof=1)) if n > 1 else 0.0,
        "frac_p_lt_001": float(np.mean(p_arr < 0.01)),
        "frac_p_lt_005": float(np.mean(p_arr < 0.05)),
        "frac_p_gt_095": float(np.mean(p_arr > 0.95)),
        "positive_rate": float(np.mean(p_arr >= threshold)),
    }


def class_specific_stats(
    y_true: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate class-conditional prediction statistics for tumor (y=1) vs non-tumor (y=0)."""
    y_arr = np.asarray(y_true, dtype=int).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)

    tumor_mask = y_arr == 1
    non_tumor_mask = y_arr == 0

    n_tumor = int(np.sum(tumor_mask))
    n_non_tumor = int(np.sum(non_tumor_mask))

    out: Dict[str, Any] = {
        "n_total": len(y_arr),
        "n_tumor": n_tumor,
        "n_non_tumor": n_non_tumor,
    }

    if n_tumor > 0:
        p_tumor = p_arr[tumor_mask]
        out["tumor"] = {
            "mean_p": float(np.mean(p_tumor)),
            "median_p": float(np.median(p_tumor)),
            "positive_rate": float(np.mean(p_tumor >= threshold)),  # Sensitivity
            "sensitivity": float(np.mean(p_tumor >= threshold)),
            "frac_p_lt_001": float(np.mean(p_tumor < 0.01)),
            "frac_p_lt_005": float(np.mean(p_tumor < 0.05)),
        }
    else:
        out["tumor"] = None

    if n_non_tumor > 0:
        p_non = p_arr[non_tumor_mask]
        spec = float(np.mean(p_non < threshold))
        out["non_tumor"] = {
            "mean_p": float(np.mean(p_non)),
            "median_p": float(np.median(p_non)),
            "positive_rate": float(np.mean(p_non >= threshold)),  # False positive rate
            "specificity": spec,
            "frac_p_lt_001": float(np.mean(p_non < 0.01)),
            "frac_p_gt_095": float(np.mean(p_non > 0.95)),
        }
    else:
        out["non_tumor"] = None

    return out


def entropy_confidence_stats(
    p: Sequence[float] | np.ndarray,
    high_conf_thresh: float = 0.95,
) -> Dict[str, float]:
    """Calculate Shannon entropy and predictive confidence statistics."""
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    if len(p_arr) == 0:
        return {}

    ent = binary_entropy(p_arr)
    conf = confidence(p_arr)

    return {
        "mean_entropy": float(np.mean(ent)),
        "median_entropy": float(np.median(ent)),
        "std_entropy": float(np.std(ent, ddof=1)) if len(p_arr) > 1 else 0.0,
        "mean_confidence": float(np.mean(conf)),
        "median_confidence": float(np.median(conf)),
        "frac_high_confidence": float(np.mean(conf >= high_conf_thresh)),
    }


def confidence_correctness_bins(
    y_true: Sequence[int] | np.ndarray,
    p: Sequence[float] | np.ndarray,
    conf_bins: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Partition predictions into confidence bins and compute empirical correctness (accuracy).

    Tests whether adaptation induces overconfident wrong predictions (high confidence, low correctness).
    """
    y_arr = np.asarray(y_true, dtype=int).reshape(-1)
    p_arr = np.asarray(p, dtype=float).reshape(-1)
    conf = confidence(p_arr)
    y_pred = (p_arr >= threshold).astype(int)
    correct = (y_pred == y_arr).astype(float)

    n_total = len(y_arr)
    bins = sorted(conf_bins)
    results = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(np.sum(mask))
        if count == 0:
            results.append(
                {
                    "bin_range": f"[{lo:.1f}, {hi:.1f}]",
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "count": 0,
                    "fraction": 0.0,
                    "mean_confidence": float("nan"),
                    "accuracy": float("nan"),
                    "error_rate": float("nan"),
                }
            )
            continue

        acc = float(np.mean(correct[mask]))
        results.append(
            {
                "bin_range": f"[{lo:.1f}, {hi:.1f}]",
                "bin_low": float(lo),
                "bin_high": float(hi),
                "count": count,
                "fraction": float(count / n_total),
                "mean_confidence": float(np.mean(conf[mask])),
                "accuracy": acc,
                "error_rate": float(1.0 - acc),
            }
        )

    return results


def slide_level_collapse_breakdown(
    y_true: Sequence[int] | np.ndarray,
    p_before: Sequence[float] | np.ndarray,
    p_after: Sequence[float] | np.ndarray,
    slides: Sequence[int | str] | np.ndarray,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Breakdown prediction shift, flip table, and performance degradation per slide."""
    y_arr = np.asarray(y_true, dtype=int).reshape(-1)
    p_b_arr = np.asarray(p_before, dtype=float).reshape(-1)
    p_a_arr = np.asarray(p_after, dtype=float).reshape(-1)
    slides_arr = np.asarray(slides).reshape(-1)

    unique_slides = sorted(set(slides_arr), key=lambda x: str(x))
    breakdown = []

    for sid in unique_slides:
        mask = slides_arr == sid
        n_patches = int(np.sum(mask))
        if n_patches == 0:
            continue

        y_s = y_arr[mask]
        p_b_s = p_b_arr[mask]
        p_a_s = p_a_arr[mask]

        m_b = binary_metrics(y_s, p_b_s, threshold=threshold)
        m_a = binary_metrics(y_s, p_a_s, threshold=threshold)
        fl = flip_table(y_s, p_b_s, p_a_s, threshold=threshold)

        sid_val = int(sid) if isinstance(sid, (np.integer, int)) else str(sid)
        row = {
            "slide": sid_val,
            "n_patches": n_patches,
            "n_tumor": int(np.sum(y_s)),
            "tumor_burden": float(np.mean(y_s)),
            "before_auroc": m_b["auroc"],
            "after_auroc": m_a["auroc"],
            "delta_auroc": (
                float(m_a["auroc"] - m_b["auroc"])
                if not (np.isnan(m_a["auroc"]) or np.isnan(m_b["auroc"]))
                else float("nan")
            ),
            "before_f1": m_b["f1"],
            "after_f1": m_a["f1"],
            "delta_f1": float(m_a["f1"] - m_b["f1"]),
            "before_sensitivity": m_b["sensitivity"],
            "after_sensitivity": m_a["sensitivity"],
            "delta_sensitivity": float(m_a["sensitivity"] - m_b["sensitivity"]),
            "before_specificity": m_b["specificity"],
            "after_specificity": m_a["specificity"],
            "delta_specificity": float(m_a["specificity"] - m_b["specificity"]),
            "before_mean_p": float(np.mean(p_b_s)),
            "after_mean_p": float(np.mean(p_a_s)),
            "before_pos_rate": float(np.mean(p_b_s >= threshold)),
            "after_pos_rate": float(np.mean(p_a_s >= threshold)),
            "harm_count": fl["harm_count"],
            "correction_count": fl["correction_count"],
            "stable_count": fl["stable_count"],
            "persistent_wrong_count": fl["persistent_wrong_count"],
            "harm_rate": fl["harm_rate"],
            "correction_rate": fl["correction_rate"],
        }
        breakdown.append(row)

    return breakdown
