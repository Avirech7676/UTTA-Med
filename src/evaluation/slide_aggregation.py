"""Slide-Level Aggregation & Whole-Slide Tumor Burden Estimation module for UTTA-Med.

Computes clinical whole-slide level metrics from patch predictions:
- Slide tumor burden estimation (fraction of tumor patches per WSI)
- Slide-level burden concordance (MAE, RMSE, Pearson r, Spearman rho)
- Slide-macro patch performance (mean AUROC and F1 across slides)
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.evaluation.classification import binary_metrics


def slide_tumor_burdens(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    slides: Sequence[int | str] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str | int, Dict[str, float | int]]:
    """Compute patch-level tumor burden fractions per whole slide image (WSI).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in [0, 1].
        slides: Slide identifiers for each patch.
        threshold: Decision threshold for discrete classification.

    Returns:
        Mapping from slide ID to slide summary:
        n_patches, n_tumor_true, burden_true, burden_pred_binary, burden_pred_prob.
    """
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)
    slides_arr = np.asarray(slides).reshape(-1)

    if len(y_true_arr) != len(y_prob_arr) or len(y_true_arr) != len(slides_arr):
        raise ValueError(
            f"Array length mismatch: y_true={len(y_true_arr)}, y_prob={len(y_prob_arr)}, slides={len(slides_arr)}"
        )

    unique_slides = sorted(set(slides_arr), key=lambda x: str(x))
    out: Dict[str | int, Dict[str, float | int]] = {}

    for sid in unique_slides:
        mask = slides_arr == sid
        n_patches = int(np.sum(mask))
        if n_patches == 0:
            continue
        y_s = y_true_arr[mask]
        p_s = y_prob_arr[mask]

        n_tumor = int(np.sum(y_s))
        b_true = float(n_tumor / n_patches)
        b_pred_bin = float(np.mean(p_s >= threshold))
        b_pred_prob = float(np.mean(p_s))

        sid_key = int(sid) if isinstance(sid, (np.integer, int)) else str(sid)
        out[sid_key] = {
            "slide": sid_key,
            "n_patches": n_patches,
            "n_tumor_true": n_tumor,
            "burden_true": b_true,
            "burden_pred_binary": b_pred_bin,
            "burden_pred_prob": b_pred_prob,
        }

    return out


def slide_burden_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    slides: Sequence[int | str] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute concordance metrics between predicted and true slide tumor burden.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        slides: Slide identifiers.
        threshold: Decision threshold.

    Returns:
        Dictionary with MAE, RMSE, Pearson r, and Spearman rho for both binary
        and probabilistic burden estimators.
    """
    burdens = slide_tumor_burdens(y_true, y_prob, slides, threshold=threshold)
    if len(burdens) < 2:
        return {
            "n_slides": len(burdens),
            "mae_binary": float("nan"),
            "rmse_binary": float("nan"),
            "pearson_r_binary": float("nan"),
            "spearman_rho_binary": float("nan"),
            "mae_prob": float("nan"),
            "rmse_prob": float("nan"),
            "pearson_r_prob": float("nan"),
            "spearman_rho_prob": float("nan"),
        }

    b_true = np.array([v["burden_true"] for v in burdens.values()])
    b_bin = np.array([v["burden_pred_binary"] for v in burdens.values()])
    b_prob = np.array([v["burden_pred_prob"] for v in burdens.values()])

    r_bin, pval_r_bin = (
        pearsonr(b_true, b_bin)
        if np.std(b_true) > 0 and np.std(b_bin) > 0
        else (float("nan"), float("nan"))
    )
    rho_bin, pval_rho_bin = (
        spearmanr(b_true, b_bin)
        if np.std(b_true) > 0 and np.std(b_bin) > 0
        else (float("nan"), float("nan"))
    )

    r_prob, pval_r_prob = (
        pearsonr(b_true, b_prob)
        if np.std(b_true) > 0 and np.std(b_prob) > 0
        else (float("nan"), float("nan"))
    )
    rho_prob, pval_rho_prob = (
        spearmanr(b_true, b_prob)
        if np.std(b_true) > 0 and np.std(b_prob) > 0
        else (float("nan"), float("nan"))
    )

    return {
        "n_slides": len(burdens),
        "mae_binary": float(mean_absolute_error(b_true, b_bin)),
        "rmse_binary": float(root_mean_squared_error(b_true, b_bin)),
        "pearson_r_binary": float(r_bin),
        "pearson_pval_binary": float(pval_r_bin),
        "spearman_rho_binary": float(rho_bin),
        "spearman_pval_binary": float(pval_rho_bin),
        "mae_prob": float(mean_absolute_error(b_true, b_prob)),
        "rmse_prob": float(root_mean_squared_error(b_true, b_prob)),
        "pearson_r_prob": float(r_prob),
        "pearson_pval_prob": float(pval_r_prob),
        "spearman_rho_prob": float(rho_prob),
        "spearman_pval_prob": float(pval_rho_prob),
    }


def slide_macro_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    slides: Sequence[int | str] | np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute slide-macro average patch classification metrics.

    Calculates AUROC, F1, Accuracy for each slide separately, and averages
    them across all slides (macro average).
    """
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob_arr = np.asarray(y_prob, dtype=float).reshape(-1)
    slides_arr = np.asarray(slides).reshape(-1)

    unique_slides = sorted(set(slides_arr), key=lambda x: str(x))
    per_slide: Dict[str | int, Dict[str, float]] = {}
    aurocs: List[float] = []
    f1s: List[float] = []
    accuracies: List[float] = []

    for sid in unique_slides:
        mask = slides_arr == sid
        y_s = y_true_arr[mask]
        p_s = y_prob_arr[mask]

        m = binary_metrics(y_s, p_s, threshold=threshold)
        sid_key = int(sid) if isinstance(sid, (np.integer, int)) else str(sid)
        per_slide[sid_key] = m
        if not np.isnan(m["auroc"]):
            aurocs.append(m["auroc"])
        if not np.isnan(m["f1"]):
            f1s.append(m["f1"])
        if not np.isnan(m["accuracy"]):
            accuracies.append(m["accuracy"])

    return {
        "macro_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "n_evaluated_slides": len(unique_slides),
        "per_slide": per_slide,
    }
