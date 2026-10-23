#!/usr/bin/env python3
"""
UTTA-Med — Stage H3.1-Repaired: 3-Seed Source Detector Training (Seeds 42, 7, 123)
================================================================================
Senior Computational Pathology Researcher, PyTorch Detection Engineer,
and Independent Reproducibility Auditor.

OBJECTIVE:
Train three independently initialized Faster R-CNN ResNet-50-FPN source detectors
(Seeds 42, 7, and 123) on MIDOG 2021 source scanners (Scanners 1 & 2, Cases 1-100)
with certified spatial coordinate repairs:
1. Parse MIDOG.json bounding boxes correctly: [x1, y1, x2, y2] where x2 - x1 == 50.
   Eliminate earlier coordinate bug where x2 was treated as width (ballooning boxes).
2. Proper negative patch handling: Use torch.zeros((0, 4)) and torch.zeros((0,), dtype=torch.int64).
   Eliminate dummy box [[0, 0, 1, 1]] with label 0 which corrupted background class assignment.
3. Strict disjoint source split: 80 train cases / 20 canonical val cases.
   Scanner 3 (Cases 101-150) is STRICTLY EXCLUDED and untouched.
4. Balanced tissue sampling: Automated tissue fraction >= 0.25, nonzero >= 0.80, variance >= 20.0.
5. Calibrated training: SGD (lr=0.005, momentum=0.9, weight_decay=0.0005, CosineAnnealingLR, 8 epochs).
6. Export checkpoints, training history, and checkpoint_hashes.json.
7. Package into utta_med_h3_1_repaired_checkpoints.zip.

EXECUTION TARGET:
- Kaggle Tesla T4 GPU (~15 minutes total for all 3 seeds).
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Enable unbuffered real-time stdout
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

STAGE = "H3.1-REPAIRED-TRAIN"
DEFAULT_PATCH_SIZE = 512
DEFAULT_PIXEL_SPACING_UM = 0.25
CENTROID_MATCHING_RADIUS_UM = 7.5
CENTROID_MATCHING_RADIUS_PX = CENTROID_MATCHING_RADIUS_UM / DEFAULT_PIXEL_SPACING_UM  # 30.0 px
DEFAULT_BOX_RADIUS_PX = 25.0  # Standard 50x50 px mitosis box

CATEGORY_MITOTIC_FIGURE = 1
CATEGORY_NOT_MITOTIC_FIGURE = 2

SOURCE_SCANNERS = [1, 2]
TARGET_HELD_OUT_SCANNER = 3

SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".svs", ".png", ".jpg", ".jpeg"}

# Canonical 20 validation cases (10 Scanner 1, 10 Scanner 2)
CANONICAL_VAL_CASES: Set[int] = {
    8, 11, 15, 19, 21, 23, 29, 39, 43, 50,  # Scanner 1 (10 cases)
    52, 57, 58, 59, 64, 70, 76, 78, 85, 94   # Scanner 2 (10 cases)
}
CANONICAL_TRAIN_CASES: Set[int] = set(range(1, 101)) - CANONICAL_VAL_CASES

EXECUTION_LOG_BUFFER: List[str] = []


def log(msg: str) -> None:
    """Print to console with immediate line flush and append to log buffer."""
    print(msg, flush=True)
    EXECUTION_LOG_BUFFER.append(msg)


def seed_all(seed: int = 42) -> None:
    """Set deterministic random seeds across all libraries."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_sha256(file_path: Path) -> str:
    """Calculate the cryptographic SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def parse_case_number_from_filename(filename: str) -> Optional[int]:
    """Extract numeric case ID from WSI file path stem."""
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def get_scanner_id_for_case(case_num: int) -> int:
    """Map case ID to MIDOG 2021 scanner domain."""
    if 1 <= case_num <= 50:
        return 1
    elif 51 <= case_num <= 100:
        return 2
    elif 101 <= case_num <= 150:
        return 3
    elif 151 <= case_num <= 200:
        return 4
    return 0


def parse_midog_annotation_box(abox: List[float]) -> Tuple[float, float, float, float, float, float]:
    """
    Parses MIDOG annotation bounding box safely.
    In MIDOG.json, bbox is formatted as [x1, y1, x2, y2] where x2 - x1 == 50 and y2 - y1 == 50.
    Returns: (x1, y1, x2, y2, cx, cy)
    """
    if len(abox) != 4:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    b0, b1, b2, b3 = float(abox[0]), float(abox[1]), float(abox[2]), float(abox[3])
    if b2 > b0 and (b2 - b0) <= 200:
        x1, y1, x2, y2 = b0, b1, b2, b3
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
    elif b2 <= 200 and b3 <= 200:
        # Standard COCO [x, y, w, h] format
        x1, y1 = b0, b1
        x2, y2 = x1 + b2, y1 + b3
        cx = x1 + b2 / 2.0
        cy = y1 + b3 / 2.0
    else:
        # Fallback [x1, y1, x2, y2]
        x1, y1, x2, y2 = b0, b1, b2, b3
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
    return x1, y1, x2, y2, cx, cy


def check_patch_quality(crop: Image.Image) -> Tuple[bool, float, float, str]:
    """
    Standard automated tissue filter:
    - Nonzero pixel fraction >= 0.80 (eliminates empty borders)
    - Tissue pixel fraction >= 0.25 (staining brightness threshold < 220)
    - Grayscale pixel variance >= 20.0 (eliminates blank glass)
    """
    arr = np.array(crop)
    nonzero_mask = np.any(arr > 10, axis=-1)
    nonzero_frac = float(np.mean(nonzero_mask))
    if nonzero_frac < 0.80:
        return False, nonzero_frac, 0.0, "nonzero_fraction_below_0.80"

    gray = arr.mean(axis=-1)
    tissue_mask = (gray < 220) & nonzero_mask
    tissue_frac = float(np.mean(tissue_mask))
    if tissue_frac < 0.25:
        return False, tissue_frac, 0.0, "tissue_fraction_below_0.25"

    variance = float(np.var(gray))
    if variance < 20.0:
        return False, tissue_frac, variance, "pixel_variance_below_20.0"

    return True, tissue_frac, variance, "ACCEPTED"


def extract_patch_lazy(img_path: Path, box_xywh: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    """Safely crop patch from WSI image with boundary padding if necessary."""
    x, y, w, h = box_xywh
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
            if x >= iw or y >= ih:
                return None
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)
            crop = im.crop((x, y, x2, y2)).convert("RGB")
            if crop.size != (w, h):
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                canvas.paste(crop, (0, 0))
                return canvas
            return crop
    except Exception:
        return None


def build_detector(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    """Build standard Faster R-CNN ResNet-50-FPN detector."""
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    weights_backbone = None
    if pretrained_backbone:
        try:
            from torchvision.models import ResNet50_Weights
            weights_backbone = ResNet50_Weights.DEFAULT
            log("  [+] Initialized with ImageNet pre-trained ResNet-50 backbone weights.")
        except Exception:
            weights_backbone = None

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        min_size=DEFAULT_PATCH_SIZE,
        max_size=DEFAULT_PATCH_SIZE,
    )
    return model


def evaluate_case_centroid_matching(
    pred_centroids: List[Tuple[float, float]],
    pred_scores: List[float],
    gt_centroids: List[Tuple[float, float]],
    radius_px: float = CENTROID_MATCHING_RADIUS_PX,
    confidence_threshold: float = 0.10,
) -> Dict[str, Any]:
    """Greedy bipartite centroid matching at radius <= 30 px."""
    valid = [(c, s) for c, s in zip(pred_centroids, pred_scores) if s >= confidence_threshold]
    valid.sort(key=lambda x: x[1], reverse=True)

    preds = [x[0] for x in valid]
    scores = [x[1] for x in valid]

    n_gt = len(gt_centroids)
    n_pred = len(preds)

    if n_gt == 0:
        return {"tp": 0, "fp": n_pred, "fn": 0, "n_pred": n_pred, "n_gt": 0, "precision": 1.0 if n_pred == 0 else 0.0, "recall": 0.0, "f1": 0.0}

    if n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": n_gt, "n_pred": 0, "n_gt": n_gt, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    gt_arr = np.array(gt_centroids)
    matched_gt: Set[int] = set()
    fp = 0

    for p_idx, (px, py) in enumerate(preds):
        dists = np.sqrt(np.sum((gt_arr - np.array([px, py])) ** 2, axis=1))
        within = np.where(dists <= radius_px)[0]
        if len(within) == 0:
            fp += 1
            continue

        sorted_cands = within[np.argsort(dists[within])]
        found = False
        for c_idx in sorted_cands:
            if c_idx not in matched_gt:
                matched_gt.add(int(c_idx))
                found = True
                break
        if not found:
            fp += 1

    tp = len(matched_gt)
    fn = n_gt - tp

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = float(tp / n_gt)
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "n_pred": n_pred, "n_gt": n_gt, "precision": prec, "recall": rec, "f1": f1}


# ------------------------------------------------------------------------------
# Main Training Routine
# ------------------------------------------------------------------------------
def main() -> None:
    t_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="UTTA-Med Stage H3.1-Repaired 3-Seed Source Detector Trainer")
    parser.add_argument("--json_path", type=str, default="", help="Explicit path to MIDOG.json")
    parser.add_argument("--images_dir", type=str, default="", help="Explicit directory containing MIDOG WSI image files (.tiff)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/h3_1_repaired_checkpoints", help="Output directory")
    parser.add_argument("--zip_name", type=str, default="/kaggle/working/utta_med_h3_1_repaired_checkpoints.zip", help="Final ZIP archive name")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs per seed (default: 8)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for Faster R-CNN (default: 2)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--max_pos_patches_per_case", type=int, default=2, help="Max positive patches harvested per case")
    parser.add_argument("--max_neg_patches_per_case", type=int, default=2, help="Max negative patches harvested per case")
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretraining on backbone")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log("UTTA-MED — STAGE H3.1-REPAIRED: 3-SEED SOURCE DETECTOR TRAINING")
    log("================================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[*] Compute Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log(f"[*] GPU Model: {gpu_name} ({vram_gb:.2f} GB)")

    # 1. Resolve Dataset & Images
    log("\n[*] Resolving MIDOG dataset and WSI images...")
    json_path: Optional[Path] = None
    if args.json_path and Path(args.json_path).is_file():
        json_path = Path(args.json_path).resolve()

    if not json_path:
        search_dirs = [Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")]
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            for root, _, files in os.walk(sdir):
                for f in files:
                    if f.lower() == "midog.json":
                        json_path = Path(root) / f
                        break
                if json_path:
                    break
            if json_path:
                break

    if not json_path or not json_path.is_file():
        log("[!] Fatal: MIDOG.json not found.")
        sys.exit(1)

    log(f"  [+] Found MIDOG.json: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load()

    image_search_dirs = []
    if args.images_dir and Path(args.images_dir).exists():
        image_search_dirs.append(Path(args.images_dir).resolve())
    image_search_dirs.extend([Path("/kaggle/input"), Path("/kaggle/working"), Path("data"), Path(".")])

    image_files: List[Path] = []
    seen_names: Set[str] = set()
    for sdir in image_search_dirs:
        if not sdir.exists():
            continue
        for root, _, files in os.walk(sdir):
            for f in files:
                if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    fp = Path(root) / f
                    if fp.name.lower() not in seen_names:
                        seen_names.add(fp.name.lower())
                        image_files.append(fp)
        if args.images_dir and len(image_files) > 0 and sdir == Path(args.images_dir).resolve():
            break

    log(f"  [+] Discovered {len(image_files)} image files on disk.")
    file2path: Dict[str, Path] = {}
    file2case: Dict[int, Path] = {}
    for p in image_files:
        file2path[p.name.lower()] = p
        file2path[p.stem.lower()] = p
        c_num = parse_case_number_from_filename(p.name)
        if c_num is not None:
            file2case[c_num] = p

    # 2. Split Manifest Verification
    log("\n[*] Establishing Canonical Source-Training Split Manifest...")
    split_manifest_data = {
        "train_cases": sorted(list(CANONICAL_TRAIN_CASES)),
        "val_cases": sorted(list(CANONICAL_VAL_CASES)),
        "scanner_3_cases_excluded": True,
        "disjoint": True,
    }
    split_manifest_path = output_dir / "source_split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest_data, f, indent=2)

    log(f"  [+] 80 Training Cases (Scanners 1 & 2): {len(CANONICAL_TRAIN_CASES)}")
    log(f"  [+] 20 Held-Out Validation Cases:     {len(CANONICAL_VAL_CASES)}")
    log(f"  [+] Target Scanner 3 (Cases 101–150): STRICTLY PRESERVED AND UNTOUCHED.")

    # 3. Harvest Balanced Repaired Patches
    log("\n[*] Harvesting Balanced Patches with Corrected Coordinate Offsets...")
    ann_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        ann_by_img[a["image_id"]].append(a)

    train_patches_pos: List[Dict[str, Any]] = []
    train_patches_neg: List[Dict[str, Any]] = []
    val_patches_pos: List[Dict[str, Any]] = []
    val_patches_neg: List[Dict[str, Any]] = []

    pos_harvest_count = defaultdict(int)
    neg_harvest_count = defaultdict(int)

    for im in coco.get("images", []):
        fn = im["file_name"]
        c_num = parse_case_number_from_filename(fn)
        if c_num is None or c_num > 100:
            continue  # Strictly source cases 1..100 only

        img_path = file2path.get(Path(fn).name.lower()) or file2path.get(Path(fn).stem.lower()) or file2case.get(c_num)
        if not img_path or not img_path.is_file():
            continue

        im_anns = ann_by_img.get(im["id"], [])
        is_val = c_num in CANONICAL_VAL_CASES

        # Extract Ground Truth Mitoses with Safe Coordinates
        gt_mitoses = []
        for a in im_anns:
            if a.get("category_id") == CATEGORY_MITOTIC_FIGURE:
                abox = a.get("bbox", [])
                x1, y1, x2, y2, cx, cy = parse_midog_annotation_box(abox)
                if x2 > x1 and y2 > y1:
                    gt_mitoses.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": cx, "cy": cy})

        try:
            with Image.open(img_path) as pil_im:
                iw, ih = pil_im.size
        except Exception:
            continue

        # 3a. Positive Patches (Centered on Mitoses)
        for m in gt_mitoses:
            if is_val and pos_harvest_count[c_num] >= args.max_pos_patches_per_case:
                break
            if not is_val and pos_harvest_count[c_num] >= args.max_pos_patches_per_case:
                break

            cx, cy = m["cx"], m["cy"]
            px = max(0, min(iw - DEFAULT_PATCH_SIZE, int(cx - DEFAULT_PATCH_SIZE / 2.0)))
            py = max(0, min(ih - DEFAULT_PATCH_SIZE, int(cy - DEFAULT_PATCH_SIZE / 2.0)))

            crop = extract_patch_lazy(img_path, (px, py, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
            if crop is None:
                continue

            valid, t_frac, var_val, _ = check_patch_quality(crop)
            if not valid:
                continue

            # Compute patch-local bounding boxes
            patch_local_boxes: List[List[float]] = []
            for gm in gt_mitoses:
                if px <= gm["cx"] < px + DEFAULT_PATCH_SIZE and py <= gm["cy"] < py + DEFAULT_PATCH_SIZE:
                    lx1 = max(0.0, float(gm["x1"] - px))
                    ly1 = max(0.0, float(gm["y1"] - py))
                    lx2 = min(float(DEFAULT_PATCH_SIZE), float(gm["x2"] - px))
                    ly2 = min(float(DEFAULT_PATCH_SIZE), float(gm["y2"] - py))
                    if (lx2 - lx1) >= 10.0 and (ly2 - ly1) >= 10.0:
                        patch_local_boxes.append([lx1, ly1, lx2, ly2])

            if len(patch_local_boxes) == 0:
                continue

            item = {
                "case_id": c_num,
                "crop": crop,
                "boxes": patch_local_boxes,
                "labels": [1] * len(patch_local_boxes),
            }
            if is_val:
                val_patches_pos.append(item)
            else:
                train_patches_pos.append(item)
            pos_harvest_count[c_num] += 1

        # 3b. Negative Patches (Tissue Regions without Any Mitoses)
        grid_steps_y = range(0, max(1, ih - DEFAULT_PATCH_SIZE + 1), 1024)
        grid_steps_x = range(0, max(1, iw - DEFAULT_PATCH_SIZE + 1), 1024)
        for gy in grid_steps_y:
            for gx in grid_steps_x:
                if is_val and neg_harvest_count[c_num] >= args.max_neg_patches_per_case:
                    break
                if not is_val and neg_harvest_count[c_num] >= args.max_neg_patches_per_case:
                    break

                # Ensure NO mitosis is anywhere near this window
                has_mitosis = any(
                    (gx - 50) <= gm["cx"] < (gx + DEFAULT_PATCH_SIZE + 50) and
                    (gy - 50) <= gm["cy"] < (gy + DEFAULT_PATCH_SIZE + 50)
                    for gm in gt_mitoses
                )
                if has_mitosis:
                    continue

                crop = extract_patch_lazy(img_path, (gx, gy, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE))
                if crop is None:
                    continue

                valid, t_frac, var_val, _ = check_patch_quality(crop)
                if not valid:
                    continue

                item = {
                    "case_id": c_num,
                    "crop": crop,
                    "boxes": [],  # Clean empty target
                    "labels": [],
                }
                if is_val:
                    val_patches_neg.append(item)
                else:
                    train_patches_neg.append(item)
                neg_harvest_count[c_num] += 1

    train_pool = train_patches_pos + train_patches_neg
    val_pool = val_patches_pos + val_patches_neg

    log(f"  [+] Harvested Training Pool:   {len(train_pool)} patches ({len(train_patches_pos)} pos, {len(train_patches_neg)} neg)")
    log(f"  [+] Harvested Validation Pool: {len(val_pool)} patches ({len(val_patches_pos)} pos, {len(val_patches_neg)} neg)")

    # 4. Multi-Seed Training (Seeds 42, 7, 123)
    TRAIN_SEEDS = [42, 7, 123]
    checkpoint_hashes: Dict[str, Any] = {}
    training_summaries: Dict[str, Any] = {}

    for seed in TRAIN_SEEDS:
        log("\n" + "=" * 80)
        log(f"[*] TRAINING SOURCE DETECTOR: SEED {seed}")
        log("=" * 80)

        seed_all(seed)
        detector = build_detector(num_classes=2, pretrained_backbone=(not args.no_pretrained)).to(device)

        params = [p for p in detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

        # Deterministic shuffle for this seed
        rng = np.random.RandomState(seed)
        seed_train_pool = copy.copy(train_pool)
        rng.shuffle(seed_train_pool)

        ckpt_path = output_dir / "checkpoints" / f"source_fasterrcnn_s{seed}.pt"
        best_val_f1 = -1.0
        seed_history: List[Dict[str, Any]] = []

        for ep in range(1, args.epochs + 1):
            t_ep = time.perf_counter()
            detector.train()
            ep_losses = []

            for b_idx in range(0, len(seed_train_pool), args.batch_size):
                batch = seed_train_pool[b_idx : b_idx + args.batch_size]
                imgs, tgts = [], []

                for item in batch:
                    arr = np.array(item["crop"]).transpose(2, 0, 1) / 255.0
                    imgs.append(torch.from_numpy(arr).float().to(device))

                    boxes = item["boxes"]
                    if len(boxes) > 0:
                        tgts.append({
                            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                            "labels": torch.ones(len(boxes), dtype=torch.int64, device=device),
                        })
                    else:
                        # NATIVE TORCHVISION ZERO-TARGET HANDLING (REPAIRED)
                        tgts.append({
                            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        })

                optimizer.zero_grad()
                loss_dict = detector(imgs, tgts)
                total_loss = sum(l for l in loss_dict.values())
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
                optimizer.step()
                ep_losses.append(float(total_loss.item()))

            scheduler.step()
            mean_loss = float(np.mean(ep_losses))

            # Validation evaluation at prespecified tau=0.10
            detector.eval()
            val_tp, val_fp, val_fn = 0, 0, 0
            val_total_gt = 0
            val_box_widths = []

            with torch.no_grad():
                for item in val_pool:
                    arr = np.array(item["crop"]).transpose(2, 0, 1) / 255.0
                    x_t = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                    out = detector(x_t)[0]

                    p_boxes = out["boxes"].cpu().numpy()
                    p_scores = out["scores"].cpu().numpy()
                    p_labels = out["labels"].cpu().numpy()

                    # Filter predictions at tau=0.10
                    keep = (p_scores >= 0.10) & (p_labels == 1)
                    cand_boxes = p_boxes[keep]
                    cand_scores = p_scores[keep]

                    for b in cand_boxes:
                        val_box_widths.append(float(b[2] - b[0]))

                    pred_cents = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in cand_boxes]
                    gt_cents = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in item["boxes"]]
                    val_total_gt += len(gt_cents)

                    res = evaluate_case_centroid_matching(pred_cents, list(cand_scores), gt_cents)
                    val_tp += res["tp"]
                    val_fp += res["fp"]
                    val_fn += res["fn"]

            val_prec = float(val_tp / (val_tp + val_fp)) if (val_tp + val_fp) > 0 else 0.0
            val_rec = float(val_tp / val_total_gt) if val_total_gt > 0 else 0.0
            val_f1 = float(2.0 * val_prec * val_rec / (val_prec + val_rec)) if (val_prec + val_rec) > 0 else 0.0
            mean_w = float(np.mean(val_box_widths)) if val_box_widths else 0.0
            ep_time = time.perf_counter() - t_ep

            log(f"  [Ep {ep:02d}/{args.epochs:02d}] Loss: {mean_loss:.4f} | Val F1 (@0.10): {val_f1:.4f} (Rec: {val_rec:.4f}, Prec: {val_prec:.4f}) | Mean Box Width: {mean_w:.1f} px [{ep_time:.1f}s]")

            seed_history.append({
                "epoch": ep,
                "loss": round(mean_loss, 4),
                "val_f1": round(val_f1, 4),
                "val_recall": round(val_rec, 4),
                "val_precision": round(val_prec, 4),
                "mean_box_width": round(mean_w, 1),
            })

            # Checkpoint selection on Val F1
            if val_f1 > best_val_f1 or not ckpt_path.exists():
                best_val_f1 = val_f1
                torch.save(detector.state_dict(), ckpt_path)

        ckpt_hash = compute_sha256(ckpt_path)
        log(f"\n[+] Saved Seed {seed} Checkpoint: {ckpt_path.name}")
        log(f"    Size: {ckpt_path.stat().st_size:,} bytes")
        log(f"    SHA-256: {ckpt_hash}")

        checkpoint_hashes[f"seed_{seed}"] = {
            "filename": ckpt_path.name,
            "path": str(ckpt_path),
            "size_bytes": ckpt_path.stat().st_size,
            "sha256": ckpt_hash,
            "best_val_f1": best_val_f1,
        }
        training_summaries[f"seed_{seed}"] = {
            "seed": seed,
            "best_val_f1": best_val_f1,
            "history": seed_history,
        }

    # 5. Export Manifests & Packaging
    log("\n" + "=" * 80)
    log("[*] EXPORTING REPAIRED CHECKPOINT HASHES & PACKAGING ARCHIVE")
    log("=" * 80)

    hashes_path = output_dir / "checkpoint_hashes.json"
    with open(hashes_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_hashes, f, indent=2)

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(training_summaries, f, indent=2)

    with open(output_dir / "execution.log", "w", encoding="utf-8") as f:
        f.write("\n".join(EXECUTION_LOG_BUFFER) + "\n")

    # Final Zip Packaging
    zip_path = Path(args.zip_name).resolve()
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                fp = Path(root) / f
                arcname = fp.relative_to(output_dir).as_posix()
                zf.write(fp, arcname=arcname)

    log(f"[PASS] Checkpoint Archive Packaged: {zip_path.name} ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    log(f"[+] Total Stage H3.1-Repaired Execution Finished in {time.perf_counter() - t_start:.1f}s.")
    log("\n" + "=" * 80)
    log("MANDATORY NEW CHECKPOINT HASHES FOR H3.1-R4b AUDIT:")
    log("=" * 80)
    for s in TRAIN_SEEDS:
        h = checkpoint_hashes[f'seed_{s}']
        log(f"Seed {s}: {h['sha256']}")
    log("=" * 80)


if __name__ == "__main__":
    main()
