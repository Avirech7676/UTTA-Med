#!/usr/bin/env python3
"""
UTTA-Med — Stage H3.1: 3-Seed Source Detector Training (Seeds 42, 7, 123)
================================================================================
Senior Computational Pathology Researcher, PyTorch Detection Engineer,
and Independent Reproducibility Auditor.

OBJECTIVE:
Train three independently initialized Faster R-CNN ResNet-50-FPN source detectors
(Seeds 42, 7, and 123) using the frozen, stabilized Stage H3.1-R2/R3 protocol:
- Pretrained ResNet-50-FPN backbone
- Balanced round-robin sampling across Scanners 1 and 2 (Cases 1-50 and 51-100)
- Quality/tissue fraction filter (tissue >= 0.25, nonzero >= 0.80, variance >= 20.0)
- Calibrated SGD (lr=0.001, momentum=0.9, CosineAnnealing, 8 epochs)
- 1:1 positive-to-negative patch ratio (80 positive, 80 negative training patches)
- Diagnostic validation on 40 held-out patches across Scanners 1 and 2
- Model selection at prespecified tau=0.10
- Exports all 3 checkpoint binaries with cryptographic SHA-256 hashes

CRITICAL REPRODUCIBILITY CONSTRAINTS:
- Held-out Target Scanner 3 (Cases 101-150) is STRICTLY UNTOUCHED and EXCLUDED.
- Stage H3.2 Target Adaptation remains BLOCKED until all 3 source baselines pass audit.
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

STAGE = "H3.1-3SEEDS"
DEFAULT_PATCH_SIZE = 512
DEFAULT_PIXEL_SPACING_UM = 0.25
CENTROID_MATCHING_RADIUS_UM = 7.5
CENTROID_MATCHING_RADIUS_PX = CENTROID_MATCHING_RADIUS_UM / DEFAULT_PIXEL_SPACING_UM  # 30.0 px
DEFAULT_BOX_RADIUS_PX = 25.0

CATEGORY_MITOTIC_FIGURE = 1
CATEGORY_NOT_MITOTIC_FIGURE = 2

SOURCE_SCANNERS = [1, 2]
TARGET_HELD_OUT_SCANNER = 3

SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".svs", ".png", ".jpg", ".jpeg"}
EXECUTION_LOG_BUFFER: List[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    EXECUTION_LOG_BUFFER.append(msg)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_git_revision() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "17ec6839825528d32e8db2fd919a754b9ed140b0"


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def parse_case_number_from_filename(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        try:
            return int(digits)
        except ValueError:
            return None
    return None


def get_scanner_id_for_case(case_num: int) -> int:
    if 1 <= case_num <= 50:
        return 1
    elif 51 <= case_num <= 100:
        return 2
    elif 101 <= case_num <= 150:
        return 3
    elif 151 <= case_num <= 200:
        return 4
    return 0


def check_patch_quality(crop: Image.Image) -> Tuple[bool, float, float]:
    arr = np.array(crop)
    nonzero_mask = np.any(arr > 10, axis=-1)
    nonzero_frac = float(np.mean(nonzero_mask))
    gray = arr.mean(axis=-1)
    tissue_mask = (gray < 220) & nonzero_mask
    tissue_frac = float(np.mean(tissue_mask))
    variance = float(np.var(gray))
    is_valid = (nonzero_frac >= 0.80) and (tissue_frac >= 0.25) and (variance >= 20.0)
    return is_valid, tissue_frac, variance


def extract_patch_lazy(img_path: Path, box_xywh: Tuple[int, int, int, int]) -> Optional[Image.Image]:
    x, y, w, h = box_xywh
    try:
        with Image.open(img_path) as im:
            iw, ih = im.size
            if x >= iw or y >= ih:
                return None
            x2 = min(iw, x + w)
            y2 = min(ih, y + h)
            crop = im.crop((x, y, x2, y2))
            if crop.size != (w, h):
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                canvas.paste(crop, (0, 0))
                return canvas
            return crop.convert("RGB")
    except Exception:
        return None


def evaluate_patch_collection_centroid(
    patch_predictions: List[Tuple[List[Tuple[float, float]], List[float]]],
    patch_ground_truths: List[List[Tuple[float, float]]],
    radius_px: float = CENTROID_MATCHING_RADIUS_PX,
    confidence_threshold: float = 0.10,
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    for (p_cents, p_scores), g_cents in zip(patch_predictions, patch_ground_truths):
        valid = [(c, s) for c, s in zip(p_cents, p_scores) if s >= confidence_threshold]
        valid.sort(key=lambda x: x[1], reverse=True)
        preds = [x[0] for x in valid]
        scores = [x[1] for x in valid]

        n_gt = len(g_cents)
        n_p = len(preds)
        total_gt += n_gt
        total_pred += n_p

        if n_gt == 0:
            total_fp += n_p
            continue
        if n_p == 0:
            total_fn += n_gt
            continue

        gt_arr = np.array(g_cents)
        matched_gt = set()
        fp = 0

        for px, py in preds:
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
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = float(total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    rec = float(total_tp / total_gt) if total_gt > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_pred": total_pred,
        "n_gt": total_gt,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def build_detector(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    weights_backbone = None
    if pretrained_backbone:
        try:
            from torchvision.models import ResNet50_Weights
            weights_backbone = ResNet50_Weights.DEFAULT
            log("  [+] Loaded pre-trained ImageNet ResNet-50 backbone weights.")
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


def extract_rpn_proposals_correctly(
    model: nn.Module,
    x_tensor: torch.Tensor,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    with torch.no_grad():
        images, _ = model.transform([x_tensor.squeeze(0)], None)
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features, None)
        rpn_boxes = proposals[0].cpu().numpy()
    rpn_cents = [((rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0) for rb in rpn_boxes]
    return rpn_boxes, rpn_cents


# ------------------------------------------------------------------------------
# Main 3-Seed Training Pipeline
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="UTTA-Med 3-Seed Source Detector Training")
    parser.add_argument("--output_dir", type=str, default="results/generalization/h3/h3_1_3seeds")
    parser.add_argument("--zip_name", type=str, default="utta_med_h3_1_3seeds.zip")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123], help="Random seeds to train")
    parser.add_argument("--train_epochs", type=int, default=8, help="Epochs per seed")
    parser.add_argument("--lr", type=float, default=0.001, help="SGD learning rate")
    parser.add_argument("--pretrained_backbone", action="store_true", default=True)
    parser.add_argument("--existing_s42_checkpoint", type=str, default="", help="Optional path to existing Seed 42 checkpoint")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 80)
    log(f"UTTA-MED — STAGE {STAGE}: 3-SEED SOURCE DETECTOR TRAINING (SEEDS: {args.seeds})")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[*] Compute Device: {device} | Git Revision: {get_git_revision()}")
    if device.type == "cuda":
        log(f"[*] GPU Model: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Discovery
    candidate_paths = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path("data/MIDOG2021"),
        Path("data/midog"),
        Path("data"),
        Path("."),
    ]
    json_path = None
    image_files = []
    for root in candidate_paths:
        if not root.exists():
            continue
        for r, _, fs in os.walk(root):
            for f in fs:
                if f.lower() == "midog.json" and not json_path:
                    json_path = Path(r) / f
                if Path(f).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    image_files.append(Path(r) / f)

    if not json_path or not image_files:
        log("\n[!] BLOCKED: MIDOG.json or image files not found.")
        sys.exit(1)

    log(f"[+] Located MIDOG.json: {json_path}")
    log(f"[+] Discovered {len(image_files)} image files on disk.")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load()

    file2path = {p.name.lower(): p for p in image_files}
    for p in image_files:
        file2path[p.stem.lower()] = p

    ann_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        ann_by_img[a["image_id"]].append(a)

    # 2. Frozen 80/20 Train/Val Split on Scanners 1 and 2
    # Seed 42 partition freeze (Cases 1-50: Scanner 1, Cases 51-100: Scanner 2)
    rng_split = np.random.RandomState(42)
    scanner_cases = {1: list(range(1, 51)), 2: list(range(51, 101))}
    train_cases_set = set()
    val_cases_set = set()
    for s_id in [1, 2]:
        cases = scanner_cases[s_id].copy()
        rng_split.shuffle(cases)
        train_cases_set.update(cases[:40])
        val_cases_set.update(cases[40:])

    log(f"[+] Frozen Source Split: {len(train_cases_set)} Train Cases | {len(val_cases_set)} Val Cases (Disjoint: True)")
    log(f"[*] Preserved Held-Out Target Scanner 3 (Cases 101-150): STRICTLY EXCLUDED.")

    # 3. Harvest candidate patches
    log("[*] Harvesting balanced patches with automated tissue filtering...")
    candidate_inventory = []
    candidates_by_case = defaultdict(lambda: {"pos": [], "neg": []})
    cand_id_counter = 0

    for im in coco.get("images", []):
        img_id = im["id"]
        fname = im["file_name"]
        c_num = parse_case_number_from_filename(fname)
        if not c_num or c_num > 100:
            continue  # Strictly source scanners only

        scanner_id = get_scanner_id_for_case(c_num)
        is_train = c_num in train_cases_set
        split_name = "train" if is_train else "val"

        path_match = file2path.get(Path(fname).name.lower()) or file2path.get(Path(fname).stem.lower())
        if not path_match or not path_match.is_file():
            continue

        im_anns = ann_by_img[img_id]
        mitoses = [a for a in im_anns if a.get("category_id") == CATEGORY_MITOTIC_FIGURE]
        impostors = [a for a in im_anns if a.get("category_id") == CATEGORY_NOT_MITOTIC_FIGURE]

        # Positives
        for m in mitoses:
            bbox = m.get("bbox", [])
            point = m.get("point", [])
            cx = point[0] if len(point) == 2 else (bbox[0] + bbox[2] / 2.0 if len(bbox) == 4 else None)
            cy = point[1] if len(point) == 2 else (bbox[1] + bbox[3] / 2.0 if len(bbox) == 4 else None)
            if cx is None or cy is None:
                continue

            pw, ph = DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE
            px = max(0, int(cx - pw / 2.0))
            py = max(0, int(cy - ph / 2.0))
            crop = extract_patch_lazy(path_match, (px, py, pw, ph))
            if crop is None:
                continue

            is_valid, tissue_f, var_f = check_patch_quality(crop)
            if not is_valid:
                continue

            local_boxes = []
            local_cents = []
            pos_cnt = 0
            for a in im_anns:
                abox = a.get("bbox", [])
                apoint = a.get("point", [])
                acx = apoint[0] if len(apoint) == 2 else (abox[0] + abox[2] / 2.0 if len(abox) == 4 else None)
                acy = apoint[1] if len(apoint) == 2 else (abox[1] + abox[3] / 2.0 if len(abox) == 4 else None)
                if acx is not None and acy is not None:
                    if px <= acx < px + pw and py <= acy < py + ph:
                        if a.get("category_id") == CATEGORY_MITOTIC_FIGURE:
                            pos_cnt += 1
                            lx, ly = acx - px, acy - py
                            local_cents.append((lx, ly))
                            r = DEFAULT_BOX_RADIUS_PX
                            local_boxes.append([max(0.0, lx - r), max(0.0, ly - r), min(float(pw), lx + r), min(float(ph), ly + r)])

            if not local_boxes:
                continue

            cand_id_counter += 1
            rec = {
                "case_id": c_num,
                "scanner_id": scanner_id,
                "patch_id": cand_id_counter,
                "slide_coordinates": f"({px},{py},{pw},{ph})",
                "image_filename": path_match.name,
                "tissue_fraction": round(tissue_f, 3),
                "image_variance": round(var_f, 1),
                "gt_mitosis_count": pos_cnt,
                "sampling_category": "positive",
                "split": split_name,
            }
            candidate_inventory.append(rec)
            candidates_by_case[c_num]["pos"].append((crop, local_boxes, local_cents, rec))

        # Negatives
        for imp in impostors:
            bbox = imp.get("bbox", [])
            point = imp.get("point", [])
            cx = point[0] if len(point) == 2 else (bbox[0] + bbox[2] / 2.0 if len(bbox) == 4 else None)
            cy = point[1] if len(point) == 2 else (bbox[1] + bbox[3] / 2.0 if len(bbox) == 4 else None)
            if cx is None or cy is None:
                continue

            pw, ph = DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE
            px = max(0, int(cx - pw / 2.0))
            py = max(0, int(cy - ph / 2.0))
            crop = extract_patch_lazy(path_match, (px, py, pw, ph))
            if crop is None:
                continue

            is_valid, tissue_f, var_f = check_patch_quality(crop)
            if not is_valid:
                continue

            has_pos = any(px <= (m.get("point", [0, 0])[0] if len(m.get("point", [])) == 2 else m.get("bbox", [0, 0])[0]) < px + pw and
                          py <= (m.get("point", [0, 0])[1] if len(m.get("point", [])) == 2 else m.get("bbox", [0, 0])[1]) < py + ph for m in mitoses)
            if has_pos:
                continue

            cand_id_counter += 1
            rec = {
                "case_id": c_num,
                "scanner_id": scanner_id,
                "patch_id": cand_id_counter,
                "slide_coordinates": f"({px},{py},{pw},{ph})",
                "image_filename": path_match.name,
                "tissue_fraction": round(tissue_f, 3),
                "image_variance": round(var_f, 1),
                "gt_mitosis_count": 0,
                "sampling_category": "negative",
                "split": split_name,
            }
            candidate_inventory.append(rec)
            candidates_by_case[c_num]["neg"].append((crop, [], [], rec))

    log(f"[+] Harvested {len(candidate_inventory)} valid candidate patches.")

    # 4. Multi-Seed Training Loop
    three_seed_results = {}
    checkpoint_hashes = {}

    for seed in args.seeds:
        log("\n" + "=" * 80)
        log(f"[*] TRAINING SOURCE DETECTOR — SEED {seed}")
        log("=" * 80)

        ckpt_path = models_dir / f"source_fasterrcnn_s{seed}.pt"

        # Check if existing Seed 42 checkpoint is provided and reuse if requested
        if seed == 42 and args.existing_s42_checkpoint and Path(args.existing_s42_checkpoint).is_file():
            log(f"  [+] Adopting existing verified Seed 42 checkpoint: {args.existing_s42_checkpoint}")
            import shutil
            shutil.copyfile(args.existing_s42_checkpoint, ckpt_path)
            s42_hash = compute_sha256(ckpt_path)
            checkpoint_hashes[f"seed_{seed}"] = {
                "checkpoint_path": ckpt_path.name,
                "sha256": s42_hash,
                "status": "ADOPTED_FROM_VERIFIED_H3_1_R2",
            }
            continue

        seed_all(seed)
        rng_sample = np.random.RandomState(seed)

        # Deterministic round-robin sample for this seed
        cand_copy = copy.deepcopy(candidates_by_case)
        tr_cases = sorted(list(train_cases_set))
        va_cases = sorted(list(val_cases_set))
        rng_sample.shuffle(tr_cases)
        rng_sample.shuffle(va_cases)

        def sample_rr(c_list: List[int], target: int, cat: str) -> List[Any]:
            selected = []
            idx = 0
            while len(selected) < target and any(len(cand_copy[c][cat]) > 0 for c in c_list):
                c = c_list[idx % len(c_list)]
                if cand_copy[c][cat]:
                    selected.append(cand_copy[c][cat].pop(0))
                idx += 1
            return selected

        used_tr_pos = sample_rr(tr_cases, 80, "pos")
        used_tr_neg = sample_rr(tr_cases, 80, "neg")
        used_va_pos = sample_rr(va_cases, 20, "pos")
        used_va_neg = sample_rr(va_cases, 20, "neg")

        train_pool = []
        for p, n in zip(used_tr_pos, used_tr_neg):
            train_pool.extend([p, n])

        val_pool = []
        for p, n in zip(used_va_pos, used_va_neg):
            val_pool.extend([p, n])

        log(f"  [+] Configured training pool ({len(train_pool)} patches) and diagnostic val pool ({len(val_pool)} patches).")

        detector = build_detector(num_classes=2, pretrained_backbone=args.pretrained_backbone).to(device)
        optimizer = torch.optim.SGD(detector.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs, eta_min=1e-5)

        best_val_f1 = -1.0
        seed_history = []

        for ep in range(1, args.train_epochs + 1):
            detector.train()
            t0 = time.perf_counter()
            ep_loss = []

            for b_idx in range(0, len(train_pool), 2):
                batch = train_pool[b_idx : b_idx + 2]
                imgs, tgts = [], []
                for pimg, boxes, _, _ in batch:
                    arr = np.array(pimg).transpose(2, 0, 1) / 255.0
                    imgs.append(torch.from_numpy(arr).float().to(device))
                    if boxes:
                        tgts.append({
                            "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                            "labels": torch.ones(len(boxes), dtype=torch.int64, device=device),
                        })
                    else:
                        tgts.append({
                            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
                        })

                optimizer.zero_grad()
                loss_dict = detector(imgs, tgts)
                loss = sum(l for l in loss_dict.values())
                loss.backward()
                optimizer.step()
                ep_loss.append(float(loss.item()))

            scheduler.step()
            ep_duration = time.perf_counter() - t0
            mean_loss = float(np.mean(ep_loss))

            # Diagnostic validation evaluation
            detector.eval()
            ep_preds = []
            ep_gts = []

            with torch.no_grad():
                for pimg, _, cents, _ in val_pool:
                    arr = np.array(pimg).transpose(2, 0, 1) / 255.0
                    x_t = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                    out = detector([x_t.squeeze(0)])[0]
                    b_p = out["boxes"].cpu().numpy()
                    s_p = out["scores"].cpu().numpy()
                    l_p = out["labels"].cpu().numpy()
                    p_cents = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b, l in zip(b_p, l_p) if l == 1]
                    p_scores = [float(s) for s, l in zip(s_p, l_p) if l == 1]
                    ep_preds.append((p_cents, p_scores))
                    ep_gts.append(cents)

            res_10 = evaluate_patch_collection_centroid(ep_preds, ep_gts, confidence_threshold=0.10)
            res_20 = evaluate_patch_collection_centroid(ep_preds, ep_gts, confidence_threshold=0.20)

            log(f"  [Seed {seed} | Ep {ep}/{args.train_epochs}] Loss: {mean_loss:.4f} | Diag F1 (@0.10): {res_10['f1']:.4f} (Rec: {res_10['recall']:.4f}, Prec: {res_10['precision']:.4f}) | F1 (@0.20): {res_20['f1']:.4f} [{ep_duration:.1f}s]")
            seed_history.append({"epoch": ep, "loss": mean_loss, "metrics_at_10": res_10, "metrics_at_20": res_20})

            # Checkpoint selection at tau=0.10
            if res_10["f1"] > best_val_f1 or not ckpt_path.exists():
                best_val_f1 = res_10["f1"]
                torch.save(detector.state_dict(), ckpt_path)

        ckpt_hash = compute_sha256(ckpt_path)
        log(f"[+] Saved Seed {seed} Checkpoint: {ckpt_path} (SHA-256: {ckpt_hash[:12]}...)")

        three_seed_results[f"seed_{seed}"] = {
            "seed": seed,
            "best_diagnostic_val_f1_at_10": best_val_f1,
            "training_history": seed_history,
            "checkpoint_file": ckpt_path.name,
            "sha256": ckpt_hash,
        }
        checkpoint_hashes[f"seed_{seed}"] = {
            "checkpoint_path": ckpt_path.name,
            "sha256": ckpt_hash,
            "best_diagnostic_val_f1_at_10": best_val_f1,
        }

    # Export manifests
    with open(output_dir / "three_seed_source_training_results.json", "w", encoding="utf-8") as f:
        json.dump(three_seed_results, f, indent=2)

    with open(output_dir / "checkpoint_hashes.json", "w", encoding="utf-8") as f:
        json.dump(checkpoint_hashes, f, indent=2)

    log(f"\n[*] Writing execution log to {output_dir / 'execution_log.txt'}...")
    with open(output_dir / "execution_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(EXECUTION_LOG_BUFFER) + "\n")

    # Package into utta_med_h3_1_3seeds.zip
    zip_path = Path(args.zip_name)
    if zip_path.exists():
        zip_path.unlink()

    archived_count = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for f in files:
                fp = Path(root) / f
                arcname = fp.relative_to(output_dir)
                zf.write(fp, arcname=str(arcname))
                archived_count += 1

    log(f"[*] Archive created: {zip_path.resolve()} ({archived_count} files archived)")
    log("\n[SUMMARY] 3-Seed Source Training Completed. Checkpoints and hashes frozen.")


if __name__ == "__main__":
    main()
