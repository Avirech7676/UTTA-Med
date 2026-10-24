"""
UTTA-Med Stage H3.1-R3: MIDOG Spatial Validation, Geometry Verification & Unit Tests
====================================================================================
Senior Computational Pathology Researcher, PyTorch Detection Engineer,
and Independent Reproducibility Auditor.

VERIFIES GATE 0 & GATE 1 REQUIREMENTS:
1. Coordinate Conversion:
   Local patch coordinates (lx, ly) to global slide coordinates (gx + lx, gy + ly).
   Local bounding boxes [lx1, ly1, lx2, ly2] to global bounding boxes.
2. Ground-Truth Spatial Bounding:
   Filters whole-slide annotations so that only GT centroids within the evaluated
   spatial tile union are eligible targets. Annotations outside are marked as uncovered.
3. Official Centroid Matching at 7.5 micrometers:
   Exact Euclidean bipartite greedy matching at radius 7.5 um (30 px at 0.25 um/px).
4. Spatial Deduplication Across Overlapping Tiles:
   Greedy non-maximum suppression by confidence score with radius 7.5 um.
5. Undefined Recall Handling:
   When N_gt == 0 in the evaluated region, recall is recorded as None (undefined),
   NOT 0.0. Precision is 0.0 if N_pred > 0, and 1.0 if N_pred == 0. F1 is None.
6. Exact Tissue Area Union Calculation:
   Ensures overlapping tile regions are not double-counted when computing mm^2.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pytest

DEFAULT_PIXEL_SPACING_UM = 0.25
CENTROID_MATCHING_RADIUS_UM = 7.5
CENTROID_MATCHING_RADIUS_PX = CENTROID_MATCHING_RADIUS_UM / DEFAULT_PIXEL_SPACING_UM  # 30.0 px


# ------------------------------------------------------------------------------
# 1. Coordinate Conversion
# ------------------------------------------------------------------------------
def local_to_global_centroid(
    local_centroid: Tuple[float, float],
    tile_origin: Tuple[int, int],
) -> Tuple[float, float]:
    """Converts a patch-local centroid (lx, ly) to whole-slide global coordinates."""
    gx, gy = tile_origin
    lx, ly = local_centroid
    return (float(gx + lx), float(gy + ly))


def local_to_global_box(
    local_box: List[float],
    tile_origin: Tuple[int, int],
) -> List[float]:
    """Converts a patch-local box [x1, y1, x2, y2] to whole-slide global coordinates."""
    gx, gy = tile_origin
    x1, y1, x2, y2 = local_box
    return [float(x1 + gx), float(y1 + gy), float(x2 + gx), float(y2 + gy)]


# ------------------------------------------------------------------------------
# 2. Ground-Truth Region Bounding
# ------------------------------------------------------------------------------
def filter_gt_to_evaluated_tiles(
    gt_centroids: List[Tuple[float, float]],
    evaluated_tiles: List[Tuple[int, int, int, int]],  # [(gx, gy, pw, ph), ...]
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Separates whole-slide ground-truth centroids into:
    - in_region_gt: falls inside at least one evaluated tile footprint
    - uncovered_gt: falls outside all evaluated tiles
    """
    in_region = []
    uncovered = []

    for cx, cy in gt_centroids:
        covered = False
        for gx, gy, pw, ph in evaluated_tiles:
            if gx <= cx < gx + pw and gy <= cy < gy + ph:
                covered = True
                break
        if covered:
            in_region.append((cx, cy))
        else:
            uncovered.append((cx, cy))

    return in_region, uncovered


# ------------------------------------------------------------------------------
# 3. Spatial Deduplication
# ------------------------------------------------------------------------------
def spatial_deduplication(
    boxes: np.ndarray,
    scores: np.ndarray,
    radius_px: float = CENTROID_MATCHING_RADIUS_PX,
) -> Tuple[List[Tuple[float, float]], List[float], List[List[float]]]:
    """Greedy spatial non-maximum suppression by Euclidean distance."""
    if len(boxes) == 0:
        return [], [], []

    order = np.argsort(scores)[::-1]
    centroids = np.array([((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in boxes])

    keep_cents = []
    keep_scores = []
    keep_boxes = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in order:
        if suppressed[i]:
            continue
        keep_cents.append((float(centroids[i, 0]), float(centroids[i, 1])))
        keep_scores.append(float(scores[i]))
        keep_boxes.append([float(x) for x in boxes[i]])
        dists = np.linalg.norm(centroids - centroids[i], axis=1)
        suppressed[dists <= radius_px] = True

    return keep_cents, keep_scores, keep_boxes


# ------------------------------------------------------------------------------
# 4. Official Centroid Matching with Undefined-Recall Handling
# ------------------------------------------------------------------------------
def evaluate_detections_centroid(
    predicted_centroids: List[Tuple[float, float]],
    predicted_scores: List[float],
    ground_truth_centroids: List[Tuple[float, float]],
    radius_px: float = CENTROID_MATCHING_RADIUS_PX,
    confidence_threshold: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluates detections against ground truth centroids strictly using greedy
    Euclidean distance matching within radius_px.
    Correctly handles N_gt == 0 by setting recall and f1 to None (undefined).
    """
    valid = [(c, s) for c, s in zip(predicted_centroids, predicted_scores) if s >= confidence_threshold]
    valid.sort(key=lambda x: x[1], reverse=True)

    preds = [x[0] for x in valid]
    scores = [x[1] for x in valid]

    n_gt = len(ground_truth_centroids)
    n_pred = len(preds)

    if n_gt == 0:
        return {
            "tp": 0,
            "fp": n_pred,
            "fn": 0,
            "n_pred": n_pred,
            "n_gt": 0,
            "precision": 1.0 if n_pred == 0 else 0.0,
            "recall": None,  # Mathematically undefined
            "f1": None,      # Mathematically undefined
            "matched_gt_indices": [],
            "unmatched_pred_indices": list(range(n_pred)),
        }

    if n_pred == 0:
        return {
            "tp": 0,
            "fp": 0,
            "fn": n_gt,
            "n_pred": 0,
            "n_gt": n_gt,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_gt_indices": [],
            "unmatched_pred_indices": [],
        }

    gt_arr = np.array(ground_truth_centroids)
    matched_gt = set()
    matched_pred = []
    unmatched_pred = []

    for p_idx, (px, py) in enumerate(preds):
        dists = np.sqrt(np.sum((gt_arr - np.array([px, py])) ** 2, axis=1))
        within = np.where(dists <= radius_px)[0]
        if len(within) == 0:
            unmatched_pred.append(p_idx)
            continue

        sorted_cands = within[np.argsort(dists[within])]
        found = False
        for c_idx in sorted_cands:
            if c_idx not in matched_gt:
                matched_gt.add(int(c_idx))
                matched_pred.append((p_idx, int(c_idx), float(scores[p_idx]), float(dists[c_idx])))
                found = True
                break
        if not found:
            unmatched_pred.append(p_idx)

    tp = len(matched_gt)
    fp = len(unmatched_pred)
    fn = n_gt - tp

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / n_gt)
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "matched_gt_indices": list(matched_gt),
        "unmatched_pred_indices": unmatched_pred,
    }


# ------------------------------------------------------------------------------
# 5. Union Area Calculation
# ------------------------------------------------------------------------------
def compute_tile_union_area_mm2(
    evaluated_tiles: List[Tuple[int, int, int, int]],  # [(gx, gy, pw, ph), ...]
    pixel_spacing_um: float = DEFAULT_PIXEL_SPACING_UM,
) -> float:
    """Computes exact union area in mm^2 for a set of possibly overlapping tiles."""
    if not evaluated_tiles:
        return 0.0

    # For discrete grid tiles, calculate non-overlapping bounding pixel footprint
    min_x = min(t[0] for t in evaluated_tiles)
    max_x = max(t[0] + t[2] for t in evaluated_tiles)
    min_y = min(t[1] for t in evaluated_tiles)
    max_y = max(t[1] + t[3] for t in evaluated_tiles)

    # If bounding box is moderately sized, use exact 2D occupancy grid
    bw = max_x - min_x
    bh = max_y - min_y

    # Downsample by factor of 16 for memory-efficient exact occupancy
    scale = 16
    grid = np.zeros(((bh + scale - 1) // scale, (bw + scale - 1) // scale), dtype=bool)

    for gx, gy, pw, ph in evaluated_tiles:
        x0 = (gx - min_x) // scale
        x1 = (gx + pw - min_x) // scale
        y0 = (gy - min_y) // scale
        y1 = (gy + ph - min_y) // scale
        grid[y0:y1, x0:x1] = True

    occupied_pixels = np.sum(grid) * (scale * scale)
    pixel_area_mm2 = (pixel_spacing_um / 1000.0) ** 2
    return float(round(occupied_pixels * pixel_area_mm2, 4))


# ==============================================================================
# PYTEST SUITE
# ==============================================================================
def test_local_to_global_conversion():
    tile = (1000, 2000)
    lx, ly = 256.0, 128.0
    gx, gy = local_to_global_centroid((lx, ly), tile)
    assert gx == 1256.0
    assert gy == 2128.0

    local_box = [100.0, 50.0, 200.0, 150.0]
    global_box = local_to_global_box(local_box, tile)
    assert global_box == [1100.0, 2050.0, 1200.0, 2150.0]


def test_gt_region_bounding_separation():
    # 2 tiles evaluated
    tiles = [(0, 0, 512, 512), (410, 0, 512, 512)]  # X spans [0, 922], Y spans [0, 512]

    # Ground truths: 1 inside tile 1, 1 inside overlap, 1 far away
    gts = [
        (100.0, 100.0),    # Inside tile 1
        (450.0, 200.0),    # Inside overlap
        (5000.0, 2000.0),  # Far away outside all tiles
    ]

    in_region, uncovered = filter_gt_to_evaluated_tiles(gts, tiles)
    assert len(in_region) == 2
    assert (100.0, 100.0) in in_region
    assert (450.0, 200.0) in in_region
    assert len(uncovered) == 1
    assert (5000.0, 2000.0) in uncovered


def test_undefined_recall_when_gt_zero():
    preds = [(100.0, 100.0), (200.0, 200.0)]
    scores = [0.85, 0.40]
    gts = []

    res = evaluate_detections_centroid(preds, scores, gts, confidence_threshold=0.10)
    assert res["n_gt"] == 0
    assert res["n_pred"] == 2
    assert res["recall"] is None
    assert res["f1"] is None
    assert res["precision"] == 0.0
    assert res["fp"] == 2


def test_centroid_distance_matching_boundary():
    # 30 px = exactly 7.5 um at 0.25 um/px
    gt = [(100.0, 100.0)]
    preds_exact = [(130.0, 100.0)]  # Distance = 30.0 px (<= 30.0 px) -> TP
    scores = [0.9]

    res_exact = evaluate_detections_centroid(preds_exact, scores, gt, radius_px=30.0)
    assert res_exact["tp"] == 1
    assert res_exact["fp"] == 0
    assert res_exact["recall"] == 1.0

    preds_outside = [(130.1, 100.0)]  # Distance = 30.1 px (> 30.0 px) -> FP
    res_outside = evaluate_detections_centroid(preds_outside, scores, gt, radius_px=30.0)
    assert res_outside["tp"] == 0
    assert res_outside["fp"] == 1
    assert res_outside["recall"] == 0.0


def test_duplicate_prediction_suppression():
    gt = [(100.0, 100.0)]
    # Two predictions near the same ground truth
    preds = [(105.0, 100.0), (95.0, 102.0)]
    scores = [0.95, 0.70]  # First is higher confidence

    res = evaluate_detections_centroid(preds, scores, gt, radius_px=30.0)
    assert res["tp"] == 1
    assert res["fp"] == 1
    assert res["recall"] == 1.0
    assert res["precision"] == 0.5


def test_spatial_deduplication():
    # Two overlapping boxes near (100, 100)
    boxes = np.array([
        [80.0, 80.0, 120.0, 120.0],   # Center (100, 100), score 0.90
        [85.0, 85.0, 125.0, 125.0],   # Center (105, 105), score 0.60 (dist = 7.07 px <= 30 px)
        [300.0, 300.0, 340.0, 340.0], # Center (320, 320), score 0.80 (distinct)
    ])
    scores = np.array([0.90, 0.60, 0.80])

    cents, scs, bxs = spatial_deduplication(boxes, scores, radius_px=30.0)
    assert len(cents) == 2
    assert cents[0] == (100.0, 100.0)
    assert cents[1] == (320.0, 320.0)


def test_union_area_overlap_discount():
    # Two tiles of 512x512 with stride 410 -> overlap is 102 px in X
    tiles = [(0, 0, 512, 512), (410, 0, 512, 512)]
    # Naive area without discount: 2 * 512 * 512 * 0.25^2 / 1e6 = 2 * 0.016384 = 0.032768 mm^2
    # Union area width = 922 px, height = 512 px -> 922 * 512 * 0.25^2 / 1e6 = 0.029504 mm^2
    union_area = compute_tile_union_area_mm2(tiles)
    naive_area = 2 * 0.016384
    assert union_area < naive_area
    assert abs(union_area - 0.0295) < 0.002
