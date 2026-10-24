"""
UTTA-Med Stage H3.0b: Official MIDOG 2021 Centroid-Distance Matching Metric & Geometric Unit Tests.

Verifies Gate 2 requirements:
1. Pure Euclidean centroid-distance matching: d <= 7.5 micrometers.
2. Disproves false equivalence between d <= 7.5 um and IoU >= 0.50.
3. Tests greedy one-to-one matching.
4. Tests duplicate prediction handling (highest confidence = TP, duplicates = FP).
5. Tests empty predictions handling.
6. Tests empty ground truth handling.
7. Tests exact matching-radius boundary conditions (d == 7.5 um vs d > 7.5 um).
"""

import numpy as np
import pytest
from typing import List, Tuple, Dict, Any


def evaluate_mitosis_detections_centroid(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    pixel_spacing_um: float = 0.25,
    matching_radius_um: float = 7.5,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Official MIDOG 2021 Evaluation Metric:
    Centroid Euclidean distance matching within matching_radius_um (default 7.5 um).
    Greedy assignment by descending confidence score.
    Each ground truth can match at most one prediction.
    """
    # Filter predictions by confidence threshold
    valid_preds = [p for p in predictions if p.get("score", 1.0) >= confidence_threshold]
    # Sort predictions by score descending
    valid_preds.sort(key=lambda p: p.get("score", 1.0), reverse=True)

    n_gt = len(ground_truth)
    n_pred = len(valid_preds)

    if n_gt == 0 and n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    if n_gt == 0:
        return {"tp": 0, "fp": n_pred, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": n_gt, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Extract coordinates in micrometers
    gt_coords_um = np.array([[g["x"] * pixel_spacing_um, g["y"] * pixel_spacing_um] for g in ground_truth])
    matched_gt_indices = set()

    tp = 0
    fp = 0

    for pred in valid_preds:
        p_coord_um = np.array([pred["x"] * pixel_spacing_um, pred["y"] * pixel_spacing_um])
        # Compute distances to all ground truths
        distances_um = np.linalg.norm(gt_coords_um - p_coord_um, axis=1)

        # Find closest ground truth within matching radius
        within_radius = np.where(distances_um <= matching_radius_um)[0]
        if len(within_radius) == 0:
            fp += 1
            continue

        # Sort candidate ground truths by distance
        sorted_candidates = within_radius[np.argsort(distances_um[within_radius])]
        match_found = False
        for cand_idx in sorted_candidates:
            if cand_idx not in matched_gt_indices:
                matched_gt_indices.add(cand_idx)
                tp += 1
                match_found = True
                break

        if not match_found:
            # All ground truths within radius are already matched to a higher-scoring prediction -> Duplicate FP!
            fp += 1

    fn = n_gt - tp
    precision = tp / max(1, (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = tp / max(1, n_gt) if n_gt > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def compute_box_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute standard bounding box IoU [x_min, y_min, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_disprove_iou_centroid_equivalence():
    """
    Gate 2 Mandate: Remove false assertion that d <= 7.5 um is equivalent to IoU >= 0.50.
    Proves mathematically that two 50x50 px boxes separated by 7.5 um (30 px) have IoU == 0.25, NOT >= 0.50!
    """
    pixel_spacing = 0.25  # um/px
    # Ground truth box centered at (100, 100), width 50, height 50
    # Centroid: (100, 100), box: [75, 75, 50, 50]
    gt_box = (75, 75, 50, 50)

    # Shift prediction by exactly 7.5 um = 30 px along X axis
    # Prediction centroid: (130, 100), box: [105, 75, 50, 50]
    pred_box_30px = (105, 75, 50, 50)

    iou = compute_box_iou(gt_box, pred_box_30px)
    # Theoretical IoU: overlap is 20x50 = 1000, union is 5000 - 1000 = 4000 -> IoU = 0.25!
    assert abs(iou - 0.25) < 1e-5, f"Expected IoU 0.25, got {iou}"
    assert iou < 0.50, "Centroid distance of 7.5 um must NOT satisfy IoU >= 0.50!"


def test_one_to_one_exact_matching():
    """Requirement: One-to-one matching within matching radius."""
    gt = [{"x": 100.0, "y": 100.0}]
    # Prediction at d = 5.0 um (20 px at 0.25 um/px)
    pred = [{"x": 120.0, "y": 100.0, "score": 0.95}]

    res = evaluate_mitosis_detections_centroid(pred, gt, pixel_spacing_um=0.25, matching_radius_um=7.5)
    assert res["tp"] == 1
    assert res["fp"] == 0
    assert res["fn"] == 0
    assert res["f1"] == 1.0


def test_duplicate_prediction_penalty():
    """Requirement: Duplicate predictions near same ground truth are penalized as FPs."""
    gt = [{"x": 100.0, "y": 100.0}]
    # Two predictions near the same ground truth:
    # Pred 1: d = 2.5 um (10 px), score 0.90
    # Pred 2: d = 5.0 um (20 px), score 0.80
    preds = [
        {"x": 110.0, "y": 100.0, "score": 0.90},
        {"x": 120.0, "y": 100.0, "score": 0.80},
    ]

    res = evaluate_mitosis_detections_centroid(preds, gt, pixel_spacing_um=0.25, matching_radius_um=7.5)
    assert res["tp"] == 1  # Pred 1 matches
    assert res["fp"] == 1  # Pred 2 is duplicate -> FP
    assert res["fn"] == 0
    assert abs(res["precision"] - 0.5) < 1e-5
    assert abs(res["recall"] - 1.0) < 1e-5
    assert abs(res["f1"] - (2 * 0.5 * 1.0 / 1.5)) < 1e-5  # F1 = 2/3


def test_empty_predictions_handling():
    """Requirement: Empty predictions handled cleanly."""
    gt = [{"x": 100.0, "y": 100.0}, {"x": 200.0, "y": 200.0}]
    preds = []

    res = evaluate_mitosis_detections_centroid(preds, gt)
    assert res["tp"] == 0
    assert res["fp"] == 0
    assert res["fn"] == 2
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_empty_ground_truth_handling():
    """Requirement: Empty ground truth handled cleanly."""
    gt = []
    preds = [{"x": 100.0, "y": 100.0, "score": 0.90}]

    res = evaluate_mitosis_detections_centroid(preds, gt)
    assert res["tp"] == 0
    assert res["fp"] == 1
    assert res["fn"] == 0
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_matching_radius_boundary_cases():
    """Requirement: Exact boundary matching at 7.5 um vs 7.501 um."""
    gt = [{"x": 100.0, "y": 100.0}]

    # Case A: exactly d = 7.5 um (30.0 pixels at 0.25 um/px)
    pred_boundary = [{"x": 130.0, "y": 100.0, "score": 0.90}]
    res_a = evaluate_mitosis_detections_centroid(pred_boundary, gt, pixel_spacing_um=0.25, matching_radius_um=7.5)
    assert res_a["tp"] == 1
    assert res_a["fp"] == 0

    # Case B: just outside at d = 7.51 um (30.04 pixels at 0.25 um/px)
    pred_outside = [{"x": 130.04, "y": 100.0, "score": 0.90}]
    res_b = evaluate_mitosis_detections_centroid(pred_outside, gt, pixel_spacing_um=0.25, matching_radius_um=7.5)
    assert res_b["tp"] == 0
    assert res_b["fp"] == 1
    assert res_b["fn"] == 1
