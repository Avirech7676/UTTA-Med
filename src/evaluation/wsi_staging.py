"""Clinical Whole-Slide Image (WSI) Lymph Node Staging Evaluation.

Maps patch predictions and tumor burden estimates to official AJCC 8th Edition
pathological staging (pTNM) for breast cancer lymph node metastases.

AJCC 8th Edition Staging Criteria:
- Negative (pN0): No tumor cells detected (tumor size = 0.0 mm).
- Isolated Tumor Cells (ITC / pN0(i+)): Single cells or cluster <= 0.2 mm (< 200 μm).
- Micro-metastasis (pN1mi): Cluster > 0.2 mm and <= 2.0 mm.
- Macro-metastasis (pN1a): Cluster > 2.0 mm.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def compute_metastasis_diameter_mm(
    positive_patch_coords: List[Tuple[int, int]],
    patch_size_px: int = 96,
    mpp: float = 0.50,  # Microns per pixel at level 0 (Camelyon nominal ~0.24-0.50 μm/px)
) -> float:
    """
    Estimates the maximum linear metastasis dimension (in millimeters) from connected patch coordinates.
    Args:
        positive_patch_coords: List of (x, y) coordinates of tumor-positive patches.
        patch_size_px: Side length of each patch in pixels.
        mpp: Microns per pixel.
    Returns:
        Estimated metastasis diameter in millimeters.
    """
    if not positive_patch_coords:
        return 0.0

    coords = np.array(positive_patch_coords)  # Shape (N, 2)
    # Bounding envelope of positive clusters
    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0) + patch_size_px

    # Diagonal extent in pixels
    dx_px = max_x - min_x
    dy_px = max_y - min_y
    diag_px = math.sqrt(dx_px ** 2 + dy_px ** 2)

    # Convert to millimeters: (pixels * mpp) / 1000
    diameter_mm = (diag_px * mpp) / 1000.0
    return float(diameter_mm)


def classify_ajcc_ptnm(diameter_mm: float) -> str:
    """Classifies estimated tumor diameter into AJCC pTNM nodal stage."""
    if diameter_mm <= 0.0:
        return "pN0"  # Negative
    elif diameter_mm <= 0.20:
        return "pN0(i+)"  # Isolated Tumor Cells (ITC)
    elif diameter_mm <= 2.00:
        return "pN1mi"  # Micro-metastasis
    else:
        return "pN1a"  # Macro-metastasis


def evaluate_slide_staging(
    patch_probs: np.ndarray,
    patch_coords: List[Tuple[int, int]],
    true_tumor_fraction: float,
    threshold: float = 0.50,
    mpp: float = 0.50,
) -> Dict[str, Any]:
    """
    Evaluates clinical slide-level diagnostic staging and tumor burden metrics.
    """
    pred_positive_mask = (patch_probs >= threshold)
    pred_tumor_frac = float(np.mean(pred_positive_mask))
    true_positive_coords = [
        coord for i, coord in enumerate(patch_coords) if pred_positive_mask[i]
    ]

    pred_diameter_mm = compute_metastasis_diameter_mm(true_positive_coords, mpp=mpp)
    pred_stage = classify_ajcc_ptnm(pred_diameter_mm)

    # True staging estimate from known slide burden
    if true_tumor_fraction <= 0.0:
        true_stage = "pN0"
    elif true_tumor_fraction < 0.01:
        true_stage = "pN0(i+)"
    elif true_tumor_fraction < 0.10:
        true_stage = "pN1mi"
    else:
        true_stage = "pN1a"

    is_stage_correct = (pred_stage == true_stage)
    burden_error = abs(pred_tumor_frac - true_tumor_fraction)

    return {
        "pred_tumor_fraction": pred_tumor_frac,
        "true_tumor_fraction": true_tumor_fraction,
        "burden_absolute_error": burden_error,
        "pred_diameter_mm": pred_diameter_mm,
        "pred_stage": pred_stage,
        "true_stage": true_stage,
        "stage_correct": is_stage_correct,
    }
