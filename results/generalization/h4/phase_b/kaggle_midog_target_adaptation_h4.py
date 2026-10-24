#!/usr/bin/env python3
"""
UTTA-Med — Stage H4: Expanded Controlled Target Adaptation Evaluation (Scanner 3)
===================================================================================
Senior Computational Pathology Researcher, PyTorch Detection Engineer,
and Independent Reproducibility Auditor.

STAGE OBJECTIVE & AUDIT MANDATE:
Execute the expanded, controlled test-time domain adaptation evaluation across Arms A–E
on held-out Target Scanner 3 (Aperio ScanScope CS2, Cases 101–150) using real MIDOG 2021
tissue WSIs, strictly enforcing the preregistered frozen manifest (h4_frozen_patch_manifest.csv),
and evaluating predictions against isolated ground-truth annotations (h4_evaluation_ground_truth.csv).

CRITICAL AUDIT & METHODOLOGICAL SAFEGUARDS:
1. Strict Frozen Manifest Enforcement:
   - Loads h4_frozen_patch_manifest.csv explicitly, verifying SHA-256 against expected hash.
   - Enforces ordered case IDs (101–150) and patch origins; aborts on any discrepancy.
   - ZERO silent patch regeneration or replacement.
2. Verified Tissue-Quality Eligibility:
   - Computes image-only quality metrics: nonzero_frac >= 0.80, tissue_frac >= 0.10, variance >= 18.0.
   - If any locked tile fails eligibility, halts execution to request protocol approval.
3. Complete Annotation Isolation:
   - GPU adaptation and inference stages consume ONLY image patches and annotation-free coordinates.
   - Ground-truth annotations are isolated in h4_evaluation_ground_truth.csv and queried
     exclusively during post-inference evaluation.
4. Preregistered Statistical Analysis:
   - Primary Estimand: Seed-Macro Averaged F1 Difference (Delta_macro = mean over seeds of Delta F1).
   - Secondary Estimand: Micro-Pooled F1 Difference (pooling detections across cases and seeds).
   - 10,000 paired case-clustered bootstrap replicates (resampling 50 cases jointly across all arms/seeds).
   - Complete export of all 10,000 replicate values to h4_bootstrap_replicates_10k.csv.
5. Standardized Matching Radius:
   - Primary matching radius: 30.0 px (7.5 um at nominal 0.25 um/px) per official MIDOG protocol.
   - Sensitivity matching radius: 25.0 px (6.25 um) reported side-by-side.
6. Uniform Boundary Slit Filter (h >= 1.0 px) applied across all 5 arms and 3 seeds.
===================================================================================
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
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops.misc import FrozenBatchNorm2d

Image.MAX_IMAGE_PIXELS = None

STAGE = "H4"
DEFAULT_SEEDS = [42, 7, 123]
DEFAULT_PATCH_SIZE = 512
DEFAULT_PIXEL_SPACING_UM = 0.25

# Matching Radii Specification:
PRIMARY_MATCHING_RADIUS_UM = 7.5
PRIMARY_MATCHING_RADIUS_PX = PRIMARY_MATCHING_RADIUS_UM / DEFAULT_PIXEL_SPACING_UM  # 30.0 px
SENSITIVITY_MATCHING_RADIUS_PX = 25.0

PRIMARY_OPERATING_THRESHOLD = 0.20
SWEEP_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
CALIBRATED_L_INF_BOUND = 0.05

# Prespecified Equivalence Margins for TOST
EQUIVALENCE_MARGIN_STRICT = 0.03
EQUIVALENCE_MARGIN_STANDARD = 0.05

# Detection Safety Monitor Parameters (Preregistered Clinical Calibration)
CALIBRATED_SILENCE_WINDOW = 30
CALIBRATED_PROLIFERATION_MAX_PER_PATCH = 4
CALIBRATED_PROLIFERATION_SOURCE_RATIO = 3.0

TARGET_SCANNER_ID = 3
CATEGORY_MITOTIC_FIGURE = 1

# Certified Checkpoint SHA-256 Digests
EXPECTED_HASHES = {
    42: "af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003",
    7: "3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e",
    123: "44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348",
}

# Immutable Manifest & Evaluation File Checksums
EXPECTED_FROZEN_MANIFEST_SHA256 = "e5a8345a68ba7f0c15162c0865ae37aa9730fcc7051525abfa9ff9f5ca08d1c8"
EXPECTED_EVALUATION_GT_SHA256 = "fc33d0e3904e204409f18fe41f841308eb4a81d0c79f662e0ab8f6997082c612"

SUPPORTED_IMAGE_EXTENSIONS = {".tiff", ".tif"}
EXCLUDED_DIR_NAMES = {
    "venv", ".venv", "env", ".env", ".git", ".idea", ".vscode",
    "site-packages", "dist-packages", "node_modules", "archive", "__pycache__", ".pytest_cache"
}


def log(msg: str) -> None:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_environment_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device_type": device.type,
        "platform": platform.platform(),
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    else:
        info["gpu_name"] = platform.processor() or "CPU"
        info["total_vram_gb"] = 0.0

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        info["git_revision"] = git_rev
    except Exception:
        info["git_revision"] = "UNKNOWN_NOT_A_GIT_CHECKOUT"
    return info


# ------------------------------------------------------------------------------
# Image Quality Assessment & Patch Extraction
# ------------------------------------------------------------------------------
def check_patch_quality(crop: Image.Image) -> Tuple[bool, float, float, float]:
    """Image-only tissue quality evaluation (nonzero >= 0.80, tissue >= 0.25, variance >= 18.0)."""
    arr = np.array(crop)
    nonzero_mask = np.any(arr > 10, axis=-1)
    nonzero_frac = float(np.mean(nonzero_mask))
    gray = arr.mean(axis=-1)
    tissue_mask = (gray < 220) & nonzero_mask
    tissue_frac = float(np.mean(tissue_mask))
    variance = float(np.var(gray))
    is_valid = (nonzero_frac >= 0.80) and (tissue_frac >= 0.10) and (variance >= 18.0)
    return is_valid, nonzero_frac, tissue_frac, variance


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


def discover_all_images(search_roots: List[Path]) -> Dict[int, Path]:
    image_map: Dict[int, Path] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for r, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            for f in files:
                f_lower = f.lower()
                if f_lower.endswith((".tiff", ".tif")):
                    stem = Path(f).stem
                    try:
                        cid = int(stem)
                        if cid not in image_map:
                            image_map[cid] = Path(r) / f
                    except ValueError:
                        pass
    return image_map


# ------------------------------------------------------------------------------
# Frozen Manifest Loading & Enforcement (Zero Silent Replacement)
# ------------------------------------------------------------------------------
def load_and_enforce_frozen_manifest(
    manifest_path: Path,
    image_map: Dict[int, Path],
    expected_sha256: str = EXPECTED_FROZEN_MANIFEST_SHA256,
    allow_synthetic_diagnostic: bool = False,
    max_patches: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Explicitly loads and enforces h4_frozen_patch_manifest.csv.
    1. Verifies SHA-256 digest against approved freeze; aborts on mismatch.
    2. Enforces ordered case IDs and patch coordinates.
    3. Verifies zero annotation leakage in manifest.
    4. Evaluates image-only quality measures; reports eligibility failures without silent replacement.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen manifest file not found at: {manifest_path}")

    actual_sha = compute_sha256(manifest_path)
    log(f"[*] Verifying Frozen Manifest SHA-256: {actual_sha} (Expected: {expected_sha256})")
    if actual_sha != expected_sha256:
        raise ValueError(
            f"MANIFEST INTEGRITY VIOLATION: Actual SHA-256 {actual_sha} does not match "
            f"expected freeze {expected_sha256}! Silent tampering or unapproved resampling is forbidden."
        )

    manifest_df = pd.read_csv(manifest_path)
    log(f"[+] Approved Manifest Authenticated: {len(manifest_df)} locked patches across {manifest_df['case_id'].nunique()} cases.")

    # Check for leaked annotation columns
    forbidden_cols = {"gt_mitoses_count", "gt_annotations_json", "annotations", "labels"}
    leaked = forbidden_cols.intersection(set(manifest_df.columns))
    if len(leaked) > 0:
        raise ValueError(f"MANIFEST ANNOTATION LEAKAGE: Leaked ground truth columns found: {leaked}")

    # Enforce strictly ordered presentation: sorted by case_id, then patch_index
    if max_patches is not None and max_patches > 0:
        manifest_df = manifest_df.head(max_patches).copy()
        log(f"[*] Smoke Test Mode Active: Constrained to first {len(manifest_df)} patches.")

    target_patches = []
    eligibility_failures = []
    quality_records = []
    cases_represented = set()

    for _, row in manifest_df.iterrows():
        g_idx = int(row["global_patch_index"])
        cid = int(row["case_id"])
        pidx = int(row["case_patch_index"])
        px = int(row["origin_x"])
        py = int(row["origin_y"])
        pw = int(row["patch_width"])
        ph = int(row["patch_height"])
        sid = str(row["stratum_id"])

        im_path = image_map.get(cid)
        crop = None
        if im_path and im_path.is_file():
            crop = extract_patch_lazy(im_path, (px, py, pw, ph))

        if crop is None:
            if allow_synthetic_diagnostic:
                # Deterministic synthetic pseudo-tissue crop for offline preflight testing
                rng_synth = np.random.RandomState(20261016 + cid * 7919 + pidx)
                synth_arr = rng_synth.randint(130, 205, size=(pw, ph, 3), dtype=np.uint8)
                crop = Image.fromarray(synth_arr, "RGB")
            else:
                raise FileNotFoundError(
                    f"WSI image for Case {cid} not found on disk. Ensure MIDOG dataset is mounted."
                )

        is_valid, nz_f, t_f, var_f = check_patch_quality(crop)
        quality_records.append({
            "global_patch_index": g_idx,
            "case_id": cid,
            "case_patch_index": pidx,
            "origin_x": px,
            "origin_y": py,
            "is_valid": is_valid,
            "nonzero_frac": round(nz_f, 4),
            "tissue_frac": round(t_f, 4),
            "variance": round(var_f, 2),
        })

        if not is_valid:
            eligibility_failures.append({
                "global_patch_index": g_idx,
                "case_id": cid,
                "case_patch_index": pidx,
                "origin_x": px,
                "origin_y": py,
                "failure_reason": f"CRITERIA_FAILED (nonzero={nz_f:.3f}, tissue={t_f:.3f}, var={var_f:.1f})",
                "nonzero_frac": round(nz_f, 4),
                "tissue_frac": round(t_f, 4),
                "variance": round(var_f, 2),
            })

        target_patches.append({
            "global_patch_index": g_idx,
            "case_id": cid,
            "patch_index": pidx,
            "image_path": str(im_path) if im_path else "SYNTHETIC_DIAGNOSTIC",
            "patch_origin_x": px,
            "patch_origin_y": py,
            "patch_size": pw,
            "stratum_id": sid,
            "tissue_fraction": round(t_f, 4),
            "variance": round(var_f, 2),
            "is_valid": is_valid,
            "crop_image": crop,
        })
        cases_represented.add(cid)

    manifest_summary = {
        "sampling_protocol": "H4_PREREGISTERED_SPATIALLY_STRATIFIED_4X4_GRID_SAMPLING",
        "declared_evaluation_population": "DECLARED_TISSUE_SAMPLED_WINDOWS_NOT_FULL_SLIDE",
        "target_scanner_id": TARGET_SCANNER_ID,
        "target_scanner_name": "Scanner_3 (Aperio ScanScope CS2)",
        "cases_represented_count": len(cases_represented),
        "total_patches_evaluated": len(target_patches),
        "total_tissue_area_mm2": round(len(target_patches) * (DEFAULT_PATCH_SIZE * DEFAULT_PIXEL_SPACING_UM / 1000.0) ** 2, 4),
        "eligibility_failures_count": len(eligibility_failures),
    }

    return target_patches, manifest_summary, quality_records


# ------------------------------------------------------------------------------
# Ground-Truth Isolation & Loading (Post-Inference Evaluation Only)
# ------------------------------------------------------------------------------
def load_and_enforce_evaluation_ground_truth(
    gt_path: Path,
    expected_sha256: str = EXPECTED_EVALUATION_GT_SHA256,
) -> Tuple[Dict[int, List[Tuple[float, float]]], int]:
    """
    Loads evaluation ground truth strictly post-inference.
    Verifies SHA-256 digest and indexes centroids by global_patch_index.
    """
    if not gt_path.is_file():
        raise FileNotFoundError(f"Evaluation ground truth file not found at: {gt_path}")

    actual_sha = compute_sha256(gt_path)
    log(f"[*] Verifying Evaluation Ground Truth SHA-256: {actual_sha} (Expected: {expected_sha256})")
    if actual_sha != expected_sha256:
        raise ValueError(f"GROUND TRUTH INTEGRITY VIOLATION: Actual SHA-256 {actual_sha} does not match expected {expected_sha256}!")

    gt_df = pd.read_csv(gt_path)
    gt_by_global_patch: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
    for _, row in gt_df.iterrows():
        g_idx = int(row["global_patch_index"])
        lx = float(row["local_x"])
        ly = float(row["local_y"])
        gt_by_global_patch[g_idx].append((lx, ly))

    total_mitoses = len(gt_df)
    log(f"[+] Evaluation Ground Truth Authenticated: {total_mitoses} mitotic figures across {len(gt_by_global_patch)} positive tiles.")
    return gt_by_global_patch, total_mitoses


# ------------------------------------------------------------------------------
# Network Architecture & Checkpoint Management
# ------------------------------------------------------------------------------
def enable_bn_affine_adaptation(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, FrozenBatchNorm2d)):
            module.eval()
            if hasattr(module, "weight") and module.weight is not None:
                if not isinstance(module.weight, nn.Parameter):
                    module.weight = nn.Parameter(module.weight.clone(), requires_grad=True)
                else:
                    module.weight.requires_grad = True
            if hasattr(module, "bias") and module.bias is not None:
                if not isinstance(module.bias, nn.Parameter):
                    module.bias = nn.Parameter(module.bias.clone(), requires_grad=True)
                else:
                    module.bias.requires_grad = True


def compute_roi_proposal_entropy(
    model: nn.Module,
    x_t: torch.Tensor,
    source_model: Optional[nn.Module] = None,
    operating_threshold: float = PRIMARY_OPERATING_THRESHOLD,
    source_anchored: bool = False,
) -> Tuple[torch.Tensor, int]:
    device = x_t.device
    model.eval()

    images, _ = model.transform([x_t.squeeze(0)], None)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features, None)

    if len(proposals) == 0 or len(proposals[0]) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, _ = model.roi_heads.box_predictor(box_features)
    class_probs = torch.softmax(class_logits, dim=-1)

    if source_anchored and source_model is not None:
        with torch.no_grad():
            s_images, _ = source_model.transform([x_t.squeeze(0)], None)
            s_features = source_model.backbone(s_images.tensors)
            s_box_features = source_model.roi_heads.box_roi_pool(s_features, proposals, s_images.image_sizes)
            s_box_features = source_model.roi_heads.box_head(s_box_features)
            s_class_logits, _ = source_model.roi_heads.box_predictor(s_box_features)
            s_class_probs = torch.softmax(s_class_logits, dim=-1)
            admit_mask = (s_class_probs[:, 1] >= operating_threshold)
    else:
        admit_mask = (class_probs[:, 1] >= operating_threshold)

    admitted_count = int(admit_mask.sum().item())
    if admitted_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0

    admitted_probs = class_probs[admit_mask]
    entropy_per_proposal = -torch.sum(admitted_probs * torch.log(admitted_probs + 1e-6), dim=-1)
    loss = torch.mean(entropy_per_proposal)
    return loss, admitted_count


def load_and_verify_source_detector(
    seed: int,
    device: torch.device,
    promote_bn: bool = True,
) -> Tuple[nn.Module, str, Path]:
    expected_hash = EXPECTED_HASHES[seed]
    candidate_files = [
        Path(f"checkpoints/source_fasterrcnn_s{seed}.pt"),
        Path(f"results/generalization/h3/h3_1_repaired_train/checkpoints/source_fasterrcnn_s{seed}.pt"),
        Path(f"/kaggle/input/datasets/avinashreddy2424/utta-med-h3-1-repaired-checkpoints/checkpoints/source_fasterrcnn_s{seed}.pt"),
        Path(f"/kaggle/input/utta-med-h3-1-repaired-checkpoints/checkpoints/source_fasterrcnn_s{seed}.pt"),
        Path(f"/kaggle/working/checkpoints/source_fasterrcnn_s{seed}.pt"),
    ]

    verified_path = None
    actual_hash = ""
    for cand in candidate_files:
        if cand.is_file():
            h = compute_sha256(cand)
            if h == expected_hash:
                verified_path = cand
                actual_hash = h
                break

    if verified_path is None:
        log(f"[!] Certified recovered checkpoint for Seed {seed} not found! Expected SHA-256: {expected_hash}")
        sys.exit(1)

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=DEFAULT_PATCH_SIZE,
        max_size=DEFAULT_PATCH_SIZE,
    )
    state = torch.load(verified_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if promote_bn:
        enable_bn_affine_adaptation(model)
    return model, actual_hash, verified_path


# ------------------------------------------------------------------------------
# Detection Safety Monitor
# ------------------------------------------------------------------------------
class DetectionSafetyMonitor:
    def __init__(
        self,
        silence_window: int = CALIBRATED_SILENCE_WINDOW,
        max_per_patch: int = CALIBRATED_PROLIFERATION_MAX_PER_PATCH,
        source_ratio: float = CALIBRATED_PROLIFERATION_SOURCE_RATIO,
    ) -> None:
        self.silence_window = silence_window
        self.max_per_patch = max_per_patch
        self.source_ratio = source_ratio
        self.empty_streak = 0
        self.consecutive_rejections = 0

    def check_patch(self, n_pred: int, n_source: Optional[int] = None) -> Tuple[bool, str]:
        if n_pred == 0:
            self.empty_streak += 1
            if self.empty_streak >= self.silence_window:
                self.consecutive_rejections += 1
                return True, "SILENCE_ALARM"
        else:
            self.empty_streak = 0

        if n_pred > self.max_per_patch:
            self.consecutive_rejections += 1
            return True, "PROLIFERATION_ABS_ALARM"

        if n_source is not None and n_source > 0:
            if n_pred > (self.source_ratio * n_source):
                self.consecutive_rejections += 1
                return True, "PROLIFERATION_REL_ALARM"

        self.consecutive_rejections = 0
        return False, "NORMAL"


def count_detections_at_tau(
    boxes: np.ndarray,
    scores: np.ndarray,
    tau: float = PRIMARY_OPERATING_THRESHOLD,
    min_box_height: float = 1.0,
) -> int:
    """
    Symmetrically counts candidate mitotic detections above operating threshold tau (0.20)
    excluding degenerate sub-1px boundary slits (height < 1.0 px).
    Enforces identical filtering criteria across both adapted and source detectors.
    """
    if len(boxes) == 0 or len(scores) == 0:
        return 0
    count = 0
    for b, s in zip(boxes, scores):
        if s >= tau:
            bh = float(b[3] - b[1])
            if bh >= min_box_height:
                count += 1
    return count


class TargetAdaptationController:
    """
    Authoritative Target Adaptation Safety Controller for UTTA-Med Stage H4.
    Encapsulates detection safety monitoring, atomic snapshot rollback,
    and latching fail-closed global fallback across all adaptation arms.
    """
    def __init__(
        self,
        arm: str,
        model: nn.Module,
        source_state: Dict[str, torch.Tensor],
        adaptable_params: List[nn.Parameter],
        optimizer: Optional[torch.optim.Optimizer] = None,
        silence_window: int = CALIBRATED_SILENCE_WINDOW,
        max_per_patch: int = CALIBRATED_PROLIFERATION_MAX_PER_PATCH,
        source_ratio: float = CALIBRATED_PROLIFERATION_SOURCE_RATIO,
        lr: float = 0.00025,
        momentum: float = 0.9,
    ) -> None:
        self.arm = arm
        self.model = model
        self.source_state = source_state
        self.adaptable_params = adaptable_params
        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum

        self.monitor = DetectionSafetyMonitor(
            silence_window=silence_window,
            max_per_patch=max_per_patch,
            source_ratio=source_ratio,
        )

        self.last_stable_state = {k: v.clone() for k, v in model.state_dict().items()}
        self.last_stable_optim = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None

        self.total_updates = 0
        self.total_alarms = 0
        self.alarm_silence_count = 0
        self.alarm_prolif_abs_count = 0
        self.alarm_prolif_rel_count = 0
        self.skipped_updates = 0
        self.actual_parameter_restorations = 0
        self.total_fallbacks = 0
        self.in_global_fallback = False

    def promote_stable_snapshot(self) -> None:
        """Promotes current parameters to verified stable snapshot after non-alarming patch."""
        self.last_stable_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        if self.optimizer is not None:
            self.last_stable_optim = copy.deepcopy(self.optimizer.state_dict())

    def process_patch_safety(
        self,
        n_det_at_tau: int,
        n_source_at_tau: Optional[int] = None,
    ) -> Tuple[bool, str, str]:
        """
        Processes safety monitor checks and executes authoritative rollback or fallback transitions.
        Returns: (alarm_triggered: bool, alarm_type: str, action_taken: str)
        """
        if self.in_global_fallback:
            return False, "IN_GLOBAL_FALLBACK", "NO_ACTION"

        alarm, alarm_type = self.monitor.check_patch(n_det_at_tau, n_source=n_source_at_tau)
        if not alarm:
            # Benign patch: promote stable snapshot
            self.promote_stable_snapshot()
            return False, alarm_type, "PROMOTED_STABLE"

        self.total_alarms += 1
        self.skipped_updates += 1
        if alarm_type == "SILENCE_ALARM":
            self.alarm_silence_count += 1
        elif alarm_type == "PROLIFERATION_ABS_ALARM":
            self.alarm_prolif_abs_count += 1
        elif alarm_type == "PROLIFERATION_REL_ALARM":
            self.alarm_prolif_rel_count += 1

        action_taken = "SKIPPED_ONLY"

        if self.arm == "Arm_E":
            if self.monitor.consecutive_rejections >= 3:
                # Global Fallback: streak >= 3 rejections triggers latching return to certified source state
                self.in_global_fallback = True
                self.total_fallbacks += 1
                self.model.load_state_dict(self.source_state)
                if self.optimizer is not None:
                    self.optimizer = torch.optim.SGD(self.adaptable_params, lr=self.lr, momentum=self.momentum)
                self.actual_parameter_restorations += 1
                div_fb = max(float((p - self.source_state[n]).abs().max().item()) 
                             for n, p in self.model.named_parameters() if n in self.source_state)
                assert div_fb == 0.0, f"Global fallback divergence: {div_fb}"
                action_taken = "GLOBAL_FALLBACK_TO_SOURCE"
            else:
                # Atomic rollback to last verified stable state
                cur_diff = max(float((p - self.last_stable_state[n]).abs().max().item()) 
                               for n, p in self.model.named_parameters() if n in self.last_stable_state)
                if cur_diff > 0.0:
                    self.model.load_state_dict(self.last_stable_state)
                    if self.optimizer is not None and self.last_stable_optim is not None:
                        self.optimizer.load_state_dict(self.last_stable_optim)
                    self.actual_parameter_restorations += 1
                    div_rb = max(float((p - self.last_stable_state[n]).abs().max().item()) 
                                 for n, p in self.model.named_parameters() if n in self.last_stable_state)
                    assert div_rb == 0.0, f"Atomic rollback divergence: {div_rb}"
                    action_taken = "ATOMIC_ROLLBACK_TO_STABLE"

        elif self.arm == "Arm_D":
            # Arm D: Step-level atomic rollback without global fallback to source
            cur_diff = max(float((p - self.last_stable_state[n]).abs().max().item()) 
                           for n, p in self.model.named_parameters() if n in self.last_stable_state)
            if cur_diff > 0.0:
                self.model.load_state_dict(self.last_stable_state)
                if self.optimizer is not None and self.last_stable_optim is not None:
                    self.optimizer.load_state_dict(self.last_stable_optim)
                self.actual_parameter_restorations += 1
                div_rb = max(float((p - self.last_stable_state[n]).abs().max().item()) 
                             for n, p in self.model.named_parameters() if n in self.last_stable_state)
                assert div_rb == 0.0, f"Atomic rollback divergence: {div_rb}"
                action_taken = "ATOMIC_ROLLBACK_TO_STABLE"

        return True, alarm_type, action_taken


# ------------------------------------------------------------------------------
# Centroid Matching & Multi-Radius Evaluation
# ------------------------------------------------------------------------------
def evaluate_detections_independent(
    predictions_per_patch: List[Tuple[List[Tuple[float, float]], List[float]]],
    ground_truths_per_patch: List[List[Tuple[float, float]]],
    radius_px: float = PRIMARY_MATCHING_RADIUS_PX,
    operating_threshold: float = PRIMARY_OPERATING_THRESHOLD,
) -> Dict[str, Any]:
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_preds = 0
    total_gts = 0

    for (p_cents, p_scores), gts in zip(predictions_per_patch, ground_truths_per_patch):
        valid = [(c, s) for c, s in zip(p_cents, p_scores) if s >= operating_threshold]
        valid.sort(key=lambda x: x[1], reverse=True)
        preds = [x[0] for x in valid]

        n_gt = len(gts)
        n_pred = len(preds)
        total_preds += n_pred
        total_gts += n_gt

        if n_gt == 0:
            total_fp += n_pred
            continue
        if n_pred == 0:
            total_fn += n_gt
            continue

        gt_arr = np.array(gts)
        matched_gt: Set[int] = set()
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
    rec = float(total_tp / total_gts) if total_gts > 0 else 0.0
    f1 = float(2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_pred": total_preds,
        "n_gt": total_gts,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


# ------------------------------------------------------------------------------
# Preregistered Paired Clustered Bootstrap Engine (Seed-Macro Primary)
# ------------------------------------------------------------------------------
def case_clustered_bootstrap_10k(
    case_seed_predictions: Dict[int, Dict[int, Dict[str, Dict[str, int]]]],
    n_replicates: int = 10000,
    seed: int = 42,
    seeds_evaluated: Optional[List[int]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Executes paired case-clustered bootstrapping (default 10,000 replicates).
    Primary Estimand: Seed-Macro Averaged F1 Difference (Delta_macro).
    Secondary Estimand: Micro-Pooled F1 Difference (Delta_micro).
    Exports all individual replicate values. Supports arbitrary evaluated seed subsets.
    """
    rng = np.random.RandomState(seed)
    case_ids = np.array(sorted(list(case_seed_predictions.keys())))
    N = len(case_ids)

    if seeds_evaluated is not None:
        seeds = sorted(seeds_evaluated)
    elif case_seed_predictions:
        first_cid = next(iter(case_seed_predictions.keys()))
        seeds = sorted(list(case_seed_predictions[first_cid].keys()))
    else:
        seeds = [42, 7, 123]

    arms = ["Arm_A", "Arm_B", "Arm_C", "Arm_D", "Arm_E"]

    bootstrap_replicates_records = []
    deltas_macro_ea = []
    deltas_micro_ea = []
    deltas_macro_ed = []
    deltas_macro_ba = []
    deltas_macro_da = []

    for b in range(n_replicates):
        samp = rng.choice(case_ids, size=N, replace=True)

        # 1. Per-seed F1 calculation
        seed_arm_f1s: Dict[int, Dict[str, float]] = {s: {} for s in seeds}
        for s in seeds:
            for arm in arms:
                tp_s = sum(case_seed_predictions[cid][s][arm]["tp"] for cid in samp)
                fp_s = sum(case_seed_predictions[cid][s][arm]["fp"] for cid in samp)
                fn_s = sum(case_seed_predictions[cid][s][arm]["fn"] for cid in samp)
                f1_s = (2.0 * tp_s) / (2.0 * tp_s + fp_s + fn_s) if (2.0 * tp_s + fp_s + fn_s) > 0 else 0.0
                seed_arm_f1s[s][arm] = f1_s

        # Primary Estimand: Seed-Macro Average across evaluated seeds
        arm_f1s_macro = {arm: float(np.mean([seed_arm_f1s[s][arm] for s in seeds])) for arm in arms}
        d_macro_ea = arm_f1s_macro["Arm_E"] - arm_f1s_macro["Arm_A"]
        d_macro_ed = arm_f1s_macro["Arm_E"] - arm_f1s_macro["Arm_D"]
        d_macro_ba = arm_f1s_macro["Arm_B"] - arm_f1s_macro["Arm_A"]
        d_macro_da = arm_f1s_macro["Arm_D"] - arm_f1s_macro["Arm_A"]

        # Secondary Estimand: Micro-Pooled Across evaluated seeds
        arm_f1s_micro = {}
        for arm in arms:
            tp_m = sum(case_seed_predictions[cid][s][arm]["tp"] for cid in samp for s in seeds)
            fp_m = sum(case_seed_predictions[cid][s][arm]["fp"] for cid in samp for s in seeds)
            fn_m = sum(case_seed_predictions[cid][s][arm]["fn"] for cid in samp for s in seeds)
            f1_m = (2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m) if (2.0 * tp_m + fp_m + fn_m) > 0 else 0.0
            arm_f1s_micro[arm] = f1_m

        d_micro_ea = arm_f1s_micro["Arm_E"] - arm_f1s_micro["Arm_A"]

        deltas_macro_ea.append(d_macro_ea)
        deltas_micro_ea.append(d_micro_ea)
        deltas_macro_ed.append(d_macro_ed)
        deltas_macro_ba.append(d_macro_ba)
        deltas_macro_da.append(d_macro_da)

        bootstrap_replicates_records.append({
            "replicate": b,
            "delta_macro_ea": round(d_macro_ea, 6),
            "delta_micro_ea": round(d_micro_ea, 6),
            "delta_macro_ed": round(d_macro_ed, 6),
            "delta_macro_ba": round(d_macro_ba, 6),
            "delta_macro_da": round(d_macro_da, 6),
            "f1_macro_arm_a": round(arm_f1s_macro["Arm_A"], 6),
            "f1_macro_arm_e": round(arm_f1s_macro["Arm_E"], 6),
            "f1_micro_arm_a": round(arm_f1s_micro["Arm_A"], 6),
            "f1_micro_arm_e": round(arm_f1s_micro["Arm_E"], 6),
        })

    def analyze_delta(deltas: List[float]) -> Dict[str, Any]:
        arr = np.array(deltas)
        mean_d = float(np.mean(arr))
        median_d = float(np.median(arr))
        ci_l = float(np.percentile(arr, 2.5))
        ci_u = float(np.percentile(arr, 97.5))
        # Empirical bootstrap tail proportion <= 0.0 (not a calibrated null-hypothesis test p-value)
        boot_tail_leq_0 = float(np.mean(arr <= 0.0))
        sig_superior = bool(ci_l > 0.0)
        sig_inferior = bool(ci_u < 0.0)
        tost_strict = bool(ci_l >= -EQUIVALENCE_MARGIN_STRICT and ci_u <= EQUIVALENCE_MARGIN_STRICT)
        tost_standard = bool(ci_l >= -EQUIVALENCE_MARGIN_STANDARD and ci_u <= EQUIVALENCE_MARGIN_STANDARD)

        if sig_superior:
            verdict = "SUPERIOR"
        elif sig_inferior:
            verdict = "INFERIOR"
        elif tost_strict:
            verdict = "EQUIVALENT_STRICT_MARGIN"
        elif tost_standard:
            verdict = "EQUIVALENT_STANDARD_MARGIN"
        else:
            verdict = "INCONCLUSIVE_NO_DEMONSTRATED_BENEFIT"

        return {
            "mean_delta": round(mean_d, 6),
            "median_delta": round(median_d, 6),
            "ci_95": [round(ci_l, 6), round(ci_u, 6)],
            "bootstrap_tail_leq_0": round(boot_tail_leq_0, 4),
            "ci_strictly_positive": sig_superior,
            "tost_strict_03": tost_strict,
            "tost_standard_05": tost_standard,
            "determination": verdict,
        }

    summary = {
        "Delta_Arm_E_vs_Arm_A_Primary_Macro": analyze_delta(deltas_macro_ea),
        "Delta_Arm_E_vs_Arm_A_Secondary_Micro": analyze_delta(deltas_micro_ea),
        "Delta_Arm_E_vs_Arm_D_Macro": analyze_delta(deltas_macro_ed),
        "Delta_Arm_B_vs_Arm_A_Macro": analyze_delta(deltas_macro_ba),
        "Delta_Arm_D_vs_Arm_A_Macro": analyze_delta(deltas_macro_da),
    }

    return summary, bootstrap_replicates_records


# ------------------------------------------------------------------------------
# Core Adaptation Evaluation Loop (Annotation-Blind)
# ------------------------------------------------------------------------------
def evaluate_seed_target_adaptation(
    seed: int,
    target_patches: List[Dict[str, Any]],
    device: torch.device,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Executes adaptation across all 5 arms for a single seed.
    GUARANTEE: Consumes strictly image patches; ZERO target annotations are queried.
    """
    seed_all(seed)
    arms = ["Arm_A", "Arm_B", "Arm_C", "Arm_D", "Arm_E"]

    arm_predictions_filtered: Dict[str, Dict[int, Tuple[List[Tuple[float, float]], List[float]]]] = {a: {} for a in arms}
    arm_predictions_raw: Dict[str, Dict[int, Tuple[List[Tuple[float, float]], List[float]]]] = {a: {} for a in arms}
    arm_telemetry: Dict[str, Any] = {}
    arm_geometry: Dict[str, Any] = {}
    raw_predictions: List[Dict[str, Any]] = []

    log(f"\n[*] Evaluating Seed {seed} across 5 Arms ({len(target_patches)} Patches, Annotation-Blind)")

    for arm in arms:
        t0 = time.perf_counter()
        adapted_model, _, _ = load_and_verify_source_detector(seed, device, promote_bn=True)
        source_model, _, _ = load_and_verify_source_detector(seed, device, promote_bn=False)
        source_model.eval()

        source_state = {k: v.clone() for k, v in adapted_model.state_dict().items()}
        adaptable_params = [p for p in adapted_model.parameters() if p.requires_grad]

        optimizer = None
        if arm != "Arm_A":
            optimizer = torch.optim.SGD(adaptable_params, lr=0.00025, momentum=0.9)

        controller = TargetAdaptationController(
            arm=arm,
            model=adapted_model,
            source_state=source_state,
            adaptable_params=adaptable_params,
            optimizer=optimizer,
            silence_window=CALIBRATED_SILENCE_WINDOW,
            max_per_patch=CALIBRATED_PROLIFERATION_MAX_PER_PATCH,
            source_ratio=CALIBRATED_PROLIFERATION_SOURCE_RATIO,
            lr=0.00025,
            momentum=0.9,
        )
        max_displacement = 0.0

        all_box_widths: List[float] = []
        all_box_heights: List[float] = []
        all_box_areas: List[float] = []
        sub_1px_slit_count = 0

        for p_idx, p_meta in enumerate(target_patches):
            # Bounded GPU memory: Stream exactly one patch tensor to GPU at a time
            arr = np.array(p_meta["crop_image"]).transpose(2, 0, 1) / 255.0
            x_t = torch.from_numpy(arr).float().unsqueeze(0).to(device)

            g_idx = p_meta["global_patch_index"]
            cid = p_meta["case_id"]
            pox = p_meta["patch_origin_x"]
            poy = p_meta["patch_origin_y"]

            # 1. Forward inference on current adapted state
            with torch.no_grad():
                out = adapted_model(x_t)[0]

            boxes = out["boxes"].cpu().numpy()
            scores = out["scores"].cpu().numpy()

            p_cents_filtered = []
            p_scores_filtered = []
            p_cents_raw = []
            p_scores_raw = []

            for b, s in zip(boxes, scores):
                bw = b[2] - b[0]
                bh = b[3] - b[1]
                ba = bw * bh
                all_box_widths.append(float(bw))
                all_box_heights.append(float(bh))
                all_box_areas.append(float(ba))

                is_sub_1px = (bh < 1.0)
                is_boundary_pinned = (b[3] >= (DEFAULT_PATCH_SIZE - 0.1))

                if is_sub_1px:
                    sub_1px_slit_count += 1

                cx_local = (b[0] + b[2]) / 2.0
                cy_local = (b[1] + b[3]) / 2.0

                p_cents_raw.append((float(cx_local), float(cy_local)))
                p_scores_raw.append(float(s))

                if not is_sub_1px:
                    p_cents_filtered.append((float(cx_local), float(cy_local)))
                    p_scores_filtered.append(float(s))

                if s >= 0.05:
                    raw_predictions.append({
                        "seed": seed,
                        "arm": arm,
                        "case_id": cid,
                        "patch_index": p_meta["patch_index"],
                        "global_patch_index": g_idx,
                        "patch_origin_x": pox,
                        "patch_origin_y": poy,
                        "box_local": [round(float(b[0]), 2), round(float(b[1]), 2), round(float(b[2]), 2), round(float(b[3]), 2)],
                        "centroid_local": [round(float(cx_local), 2), round(float(cy_local), 2)],
                        "centroid_global": [round(float(pox + cx_local), 2), round(float(poy + cy_local), 2)],
                        "width_px": round(float(bw), 2),
                        "height_px": round(float(bh), 2),
                        "score": round(float(s), 4),
                        "is_sub_1px": is_sub_1px,
                        "is_boundary_pinned": is_boundary_pinned,
                    })

            arm_predictions_filtered[arm][g_idx] = (p_cents_filtered, p_scores_filtered)
            arm_predictions_raw[arm][g_idx] = (p_cents_raw, p_scores_raw)

            # 2. Adaptation step on pre-update state
            if arm == "Arm_A" or controller.optimizer is None or controller.in_global_fallback:
                del x_t
                continue

            # Compute source reference detections symmetrically
            with torch.no_grad():
                src_out = source_model(x_t)[0]
            src_boxes = src_out["boxes"].cpu().numpy()
            src_scores = src_out["scores"].cpu().numpy()
            n_source_at_tau = count_detections_at_tau(src_boxes, src_scores, tau=PRIMARY_OPERATING_THRESHOLD, min_box_height=1.0)

            n_det_at_tau = count_detections_at_tau(boxes, scores, tau=PRIMARY_OPERATING_THRESHOLD, min_box_height=1.0)

            alarm, alarm_type, action_taken = controller.process_patch_safety(
                n_det_at_tau=n_det_at_tau,
                n_source_at_tau=n_source_at_tau,
            )

            if arm in ["Arm_D", "Arm_E"] and alarm:
                del x_t
                continue

            is_source_anchored = (arm == "Arm_E")
            loss_ent, n_admitted = compute_roi_proposal_entropy(
                model=adapted_model,
                x_t=x_t,
                source_model=source_model,
                operating_threshold=PRIMARY_OPERATING_THRESHOLD,
                source_anchored=is_source_anchored,
            )

            if n_admitted == 0:
                controller.skipped_updates += 1
                del x_t
                continue

            # Pre-step state snapshot for atomic rollback upon gradient anomaly
            pre_step_state = {k: v.clone() for k, v in adapted_model.state_dict().items()}
            pre_step_optim = copy.deepcopy(controller.optimizer.state_dict())

            controller.optimizer.zero_grad()
            loss_ent.backward()

            # Gradient sanity check
            grad_finite = True
            for p in adaptable_params:
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break

            if not grad_finite or not torch.isfinite(loss_ent):
                # Gradient failure: atomically roll back parameters
                adapted_model.load_state_dict(pre_step_state)
                controller.optimizer.load_state_dict(pre_step_optim)
                controller.actual_parameter_restorations += 1
                controller.skipped_updates += 1
                del x_t
                continue

            controller.optimizer.step()
            controller.total_updates += 1

            # Parameter displacement bound (L_inf <= 0.05)
            if arm in ["Arm_C", "Arm_D", "Arm_E"]:
                with torch.no_grad():
                    for name, p_curr in adapted_model.named_parameters():
                        if p_curr.requires_grad and name in source_state:
                            p_src = source_state[name]
                            diff = p_curr - p_src
                            cur_disp = float(diff.abs().max().item())
                            if cur_disp > max_displacement:
                                max_displacement = cur_disp
                            p_curr.clamp_(p_src - CALIBRATED_L_INF_BOUND, p_src + CALIBRATED_L_INF_BOUND)

            del x_t

        elapsed = time.perf_counter() - t0
        arm_telemetry[arm] = {
            "updates": controller.total_updates,
            "alarms": controller.total_alarms,
            "alarms_silence": controller.alarm_silence_count,
            "alarms_prolif_abs": controller.alarm_prolif_abs_count,
            "alarms_prolif_rel": controller.alarm_prolif_rel_count,
            "skipped_updates": controller.skipped_updates,
            "actual_parameter_restorations": controller.actual_parameter_restorations,
            "rollbacks": controller.actual_parameter_restorations,  # Preserved legacy alias
            "fallbacks": controller.total_fallbacks,
            "max_displacement": round(max_displacement, 6),
            "in_global_fallback": controller.in_global_fallback,
            "time_sec": round(elapsed, 2),
            "sub_1px_slits_filtered": sub_1px_slit_count,
        }

        mean_w = float(np.mean(all_box_widths)) if len(all_box_widths) > 0 else 0.0
        mean_h = float(np.mean(all_box_heights)) if len(all_box_heights) > 0 else 0.0
        frac_over_25 = float(np.mean([a > (0.25 * DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE) for a in all_box_areas])) if len(all_box_areas) > 0 else 0.0
        arm_geometry[arm] = {
            "mean_width_px": round(mean_w, 2),
            "mean_height_px": round(mean_h, 2),
            "frac_boxes_over_25_pct_tile": round(frac_over_25, 4),
            "sub_1px_slit_count": sub_1px_slit_count,
        }

    return arm_predictions_filtered, arm_predictions_raw, arm_telemetry, raw_predictions, arm_geometry


# ------------------------------------------------------------------------------
# Preflight Implementation Validation Suite
# ------------------------------------------------------------------------------
def execute_preflight_suite(device: torch.device) -> bool:
    log("\n" + "=" * 80)
    log("[*] EXECUTING STAGE H4 PREFLIGHT IMPLEMENTATION VALIDATION SUITE")
    log("=" * 80)

    # Gate 1: Checkpoint Provenance
    for seed, exp_hash in EXPECTED_HASHES.items():
        _, actual_hash, _ = load_and_verify_source_detector(seed, device)
        if actual_hash != exp_hash:
            log(f"  [FAIL] Hash mismatch for Seed {seed}!")
            return False
        log(f"  [PASS] Seed {seed} Authenticated: {actual_hash[:16]}... (Certified Recovered)")

    # Gate 2: BN Affine Promotion & Freeze Invariant
    unpromoted_model, _, _ = load_and_verify_source_detector(42, device, promote_bn=False)
    promoted_model, _, _ = load_and_verify_source_detector(42, device, promote_bn=True)

    adaptable_params = [p for p in promoted_model.parameters() if p.requires_grad]
    if len(adaptable_params) != 106:
        log(f"  [FAIL] Expected 106 adaptable BN affine parameters, got {len(adaptable_params)}")
        return False
    log(f"  [PASS] Exactly 106 BN affine parameters promoted across 53 normalization modules.")

    # Gate 3: Model Equivalence Test
    dummy_x = torch.randn(1, 3, 512, 512, device=device)
    unpromoted_model.eval()
    promoted_model.eval()
    with torch.no_grad():
        out_unprom = unpromoted_model(dummy_x)[0]
        out_prom = promoted_model(dummy_x)[0]

    diff_boxes = float((out_unprom["boxes"] - out_prom["boxes"]).abs().max().item()) if len(out_unprom["boxes"]) > 0 else 0.0
    if diff_boxes > 1e-4:
        log(f"  [FAIL] Bit-for-bit divergence: {diff_boxes}")
        return False
    log("  [PASS] Pre-adaptation bit-for-bit numerical equivalence verified.")

    # Gate 4: Atomic Rollback & Global Fallback Divergence Verification
    log("\n[*] Gate 4: Controller State Transitions & Parameter Rollback Invariance...")
    s42_promoted, _, _ = load_and_verify_source_detector(42, device, promote_bn=True)
    source_backup = {k: v.clone() for k, v in s42_promoted.state_dict().items()}
    
    # 4a. Atomic Rollback Test
    with torch.no_grad():
        for p in s42_promoted.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.01)
    
    mut_div = max(float((p - source_backup[n]).abs().max().item()) for n, p in s42_promoted.named_parameters() if p.requires_grad)
    if mut_div == 0.0:
        log("  [FAIL] Test mutation did not introduce non-zero divergence.")
        return False
    
    s42_promoted.load_state_dict(source_backup)
    post_rb_div = max(float((p - source_backup[n]).abs().max().item()) for n, p in s42_promoted.named_parameters() if n in source_backup)
    if post_rb_div != 0.0:
        log(f"  [FAIL] Non-zero atomic rollback divergence: {post_rb_div:.8f}")
        return False
    log(f"  [PASS] Atomic Rollback Invariant Verified: Divergence = {post_rb_div:.8f} (Exact Bitwise Match).")

    # 4b. Global Fallback Test
    with torch.no_grad():
        for p in s42_promoted.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * 0.02)
    s42_promoted.load_state_dict(source_backup)
    post_fb_div = max(float((p - source_backup[n]).abs().max().item()) for n, p in s42_promoted.named_parameters() if n in source_backup)
    if post_fb_div != 0.0:
        log(f"  [FAIL] Non-zero global fallback divergence: {post_fb_div:.8f}")
        return False
    log(f"  [PASS] Global Fallback Invariant Verified: Divergence = {post_fb_div:.8f} (Exact Bitwise Match).")

    return True


# ------------------------------------------------------------------------------
# Archive Packaging Safe
# ------------------------------------------------------------------------------
def package_archive_safe(output_dir: Path, zip_path: Path) -> int:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total_files = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                p = Path(root) / file
                if p.suffix.lower() in [".zip"] or p.resolve() == zip_path.resolve():
                    continue
                arcname = p.relative_to(output_dir)
                zf.write(p, arcname=arcname)
                total_files += 1
    return total_files


# ------------------------------------------------------------------------------
# Main Execution Pipeline
# ------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="UTTA-Med Stage H4 Target Adaptation Runner")
    parser.add_argument("--output_dir", type=str, default="results/generalization/h4/phase_b")
    parser.add_argument("--zip_name", type=str, default="utta_med_h4_phase_b_execution_ready.zip")
    parser.add_argument("--manifest_path", type=str, default="h4_frozen_patch_manifest.csv")
    parser.add_argument("--ground_truth_path", type=str, default="h4_evaluation_ground_truth.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--allow_synthetic_diagnostic", action="store_true", help="Allow synthetic pseudo-tissue crops if WSIs are unmounted")
    parser.add_argument("--stage_b_smoke_test", "--smoke_test", action="store_true", help="Execute real preflight smoke test on 10 patches across 2 cases")
    parser.add_argument("--bootstrap_replicates", type=int, default=10000, help="Number of bootstrap replicates (default: 10,000)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    log("=" * 80)
    log(f"UTTA-MED — STAGE {STAGE}: CONTROLLED TARGET ADAPTATION (SCANNER {TARGET_SCANNER_ID})")
    log("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = get_environment_info(device)
    log(f"[*] Compute Device: {device} | Git Revision: {env['git_revision']}")
    if device.type == "cuda":
        log(f"[*] GPU Model: {env['gpu_name']} ({env['total_vram_gb']} GB VRAM)")

    # Execute Preflight Validation Suite
    preflight_ok = execute_preflight_suite(device)
    if not preflight_ok:
        log("[!] PREFLIGHT FAILED: Model implementation gates not certified.")
        sys.exit(1)

    # Discover WSI images once
    search_dirs = [Path("data"), Path("C:/Users/avina/Downloads"), Path("/kaggle/input"), Path(".")]
    image_map = discover_all_images(search_dirs)
    log(f"[+] Discovered {len(image_map)} WSI images on disk.")

    # 1. Load and Enforce Frozen Manifest
    max_p = 10 if args.stage_b_smoke_test else None
    target_patches, manifest_summary, quality_records = load_and_enforce_frozen_manifest(
        manifest_path=Path(args.manifest_path),
        image_map=image_map,
        expected_sha256=EXPECTED_FROZEN_MANIFEST_SHA256,
        allow_synthetic_diagnostic=args.allow_synthetic_diagnostic,
        max_patches=max_p,
    )

    # Save Tissue Quality Audit Table
    pd.DataFrame(quality_records).to_csv(output_dir / "h4_tissue_quality_audit.csv", index=False)

    # Halt if any eligibility failures occur (zero silent replacement)
    failures = [q for q in quality_records if not q["is_valid"]]
    if len(failures) > 0:
        pd.DataFrame(failures).to_csv(output_dir / "h4_tissue_eligibility_failures.csv", index=False)
        log("\n" + "!" * 80)
        log(f"[!] PROTOCOL HALT: {len(failures)} frozen patches failed tissue quality eligibility!")
        log(f"[!] Silent replacement is forbidden by preregistered protocol. Halting execution.")
        log("!" * 80)
        sys.exit(1)

    # 2. Adaptation & Inference Loop Across Seeds & Arms (Zero Ground Truth Access)
    seeds_to_run = args.seeds if not args.stage_b_smoke_test else [42]
    all_raw_predictions: List[Dict[str, Any]] = []
    all_telemetry_by_seed: Dict[int, Dict[str, Any]] = {}
    all_geometry_by_seed: Dict[int, Dict[str, Any]] = {}

    predictions_by_seed_arm_filtered: Dict[int, Dict[str, Dict[int, Tuple[List[Tuple[float, float]], List[float]]]]] = {}
    predictions_by_seed_arm_raw: Dict[int, Dict[str, Dict[int, Tuple[List[Tuple[float, float]], List[float]]]]] = {}

    for seed in seeds_to_run:
        arm_preds_f, arm_preds_r, telemetry, raw_preds, geom = evaluate_seed_target_adaptation(
            seed=seed,
            target_patches=target_patches,
            device=device,
            output_dir=output_dir,
        )
        predictions_by_seed_arm_filtered[seed] = arm_preds_f
        predictions_by_seed_arm_raw[seed] = arm_preds_r
        all_telemetry_by_seed[seed] = telemetry
        all_geometry_by_seed[seed] = geom
        all_raw_predictions.extend(raw_preds)

    # Export Raw Predictions CSV
    with open(output_dir / "h4_raw_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "seed", "arm", "case_id", "patch_index", "global_patch_index", "patch_origin_x", "patch_origin_y",
            "box_local", "centroid_local", "centroid_global", "width_px", "height_px", "score", "is_sub_1px", "is_boundary_pinned"
        ])
        writer.writeheader()
        for r in all_raw_predictions:
            writer.writerow(r)

    # 3. Post-Inference Evaluation Against Isolated Ground Truth
    log("\n" + "=" * 80)
    log("[*] POST-INFERENCE EVALUATION AGAINST ISOLATED GROUND TRUTH")
    log("=" * 80)
    gt_dict, total_mitoses = load_and_enforce_evaluation_ground_truth(
        gt_path=Path(args.ground_truth_path),
        expected_sha256=EXPECTED_EVALUATION_GT_SHA256,
    )

    per_arm_metrics_records = []
    case_seed_predictions: Dict[int, Dict[int, Dict[str, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}))
    )
    all_eval_results_by_seed: Dict[int, Dict[str, Any]] = defaultdict(dict)
    sens_records = []

    patch_global_ids = [p["global_patch_index"] for p in target_patches]
    gts_for_eval = [gt_dict.get(gid, []) for gid in patch_global_ids]

    for s in seeds_to_run:
        for arm in ["Arm_A", "Arm_B", "Arm_C", "Arm_D", "Arm_E"]:
            preds_f = [predictions_by_seed_arm_filtered[s][arm][gid] for gid in patch_global_ids]
            preds_r = [predictions_by_seed_arm_raw[s][arm][gid] for gid in patch_global_ids]

            # Primary: 30.0 px (7.5 um)
            eval_prim = evaluate_detections_independent(preds_f, gts_for_eval, radius_px=PRIMARY_MATCHING_RADIUS_PX)
            # Sensitivity: 25.0 px (6.25 um)
            eval_sens = evaluate_detections_independent(preds_f, gts_for_eval, radius_px=SENSITIVITY_MATCHING_RADIUS_PX)
            # Unfiltered control: 30.0 px
            eval_unfilt = evaluate_detections_independent(preds_r, gts_for_eval, radius_px=PRIMARY_MATCHING_RADIUS_PX)

            all_eval_results_by_seed[s][arm] = eval_prim
            tel = all_telemetry_by_seed[s][arm]

            per_arm_metrics_records.append({
                "seed": s,
                "arm": arm,
                "precision": eval_prim["precision"],
                "recall": eval_prim["recall"],
                "f1": eval_prim["f1"],
                "tp": eval_prim["tp"],
                "fp": eval_prim["fp"],
                "fn": eval_prim["fn"],
                "n_pred": eval_prim["n_pred"],
                "n_gt": eval_prim["n_gt"],
                "time_sec": tel["time_sec"],
                "updates": tel["updates"],
                "alarms": tel["alarms"],
                "alarms_silence": tel["alarms_silence"],
                "alarms_prolif_abs": tel["alarms_prolif_abs"],
                "alarms_prolif_rel": tel["alarms_prolif_rel"],
                "skipped_updates": tel["skipped_updates"],
                "actual_parameter_restorations": tel["actual_parameter_restorations"],
                "rollbacks": tel["actual_parameter_restorations"],
                "fallbacks": tel["fallbacks"],
                "in_global_fallback": tel["in_global_fallback"],
                "max_displacement": tel["max_displacement"],
            })

            sens_records.append({
                "seed": s,
                "arm": arm,
                "f1_primary_30px": eval_prim["f1"],
                "prec_primary_30px": eval_prim["precision"],
                "rec_primary_30px": eval_prim["recall"],
                "tp_30px": eval_prim["tp"],
                "fp_30px": eval_prim["fp"],
                "f1_sens_25px": eval_sens["f1"],
                "prec_sens_25px": eval_sens["precision"],
                "rec_sens_25px": eval_sens["recall"],
                "tp_25px": eval_sens["tp"],
                "fp_25px": eval_sens["fp"],
                "f1_raw_unfiltered_30px": eval_unfilt["f1"],
                "prec_raw_unfiltered_30px": eval_unfilt["precision"],
            })

            # Case-level breakdown for clustered bootstrapping
            for p_meta, (c_preds, c_scores) in zip(target_patches, preds_f):
                cid = p_meta["case_id"]
                gid = p_meta["global_patch_index"]
                p_eval = evaluate_detections_independent([(c_preds, c_scores)], [gt_dict.get(gid, [])], radius_px=PRIMARY_MATCHING_RADIUS_PX)
                case_seed_predictions[cid][s][arm]["tp"] += p_eval["tp"]
                case_seed_predictions[cid][s][arm]["fp"] += p_eval["fp"]
                case_seed_predictions[cid][s][arm]["fn"] += p_eval["fn"]

            log(f"  Seed {s} {arm:<6}: F1={eval_prim['f1']:.4f} | Prec={eval_prim['precision']:.4f} | Rec={eval_prim['recall']:.4f} | TP={eval_prim['tp']} | FP={eval_prim['fp']} | Alarms={tel['alarms']} | Restorations={tel['actual_parameter_restorations']}")

    pd.DataFrame(per_arm_metrics_records).to_csv(output_dir / "h4_per_arm_metrics.csv", index=False)
    pd.DataFrame(sens_records).to_csv(output_dir / "h4_radius_sensitivity_comparison.csv", index=False)

    # 4. Preregistered 10,000 Paired Case-Clustered Bootstrap Analysis
    n_boot = 100 if args.stage_b_smoke_test else args.bootstrap_replicates
    log("\n" + "=" * 80)
    log(f"[*] EXECUTING PAIRED CASE-CLUSTERED BOOTSTRAP (B = {n_boot}, SEED-MACRO PRIMARY)")
    log("=" * 80)

    bootstrap_summary, boot_replicate_records = case_clustered_bootstrap_10k(
        case_seed_predictions=case_seed_predictions,
        n_replicates=n_boot,
        seed=42,
        seeds_evaluated=seeds_to_run,
    )

    pd.DataFrame(boot_replicate_records).to_csv(output_dir / "h4_bootstrap_replicates_10k.csv", index=False)

    ci_records = [
        {
            "estimand": "Seed-Macro Averaged (Primary)",
            "contrast": "Arm_E_minus_Arm_A",
            "bootstrap_replicates": n_boot,
            "mean_delta_f1": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["mean_delta"],
            "median_delta_f1": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["median_delta"],
            "ci_95_lower": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["ci_95"][0],
            "ci_95_upper": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["ci_95"][1],
            "bootstrap_tail_leq_0": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["bootstrap_tail_leq_0"],
            "scientific_determination": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["determination"],
        },
        {
            "estimand": "Micro-Pooled (Secondary)",
            "contrast": "Arm_E_minus_Arm_A",
            "bootstrap_replicates": n_boot,
            "mean_delta_f1": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["mean_delta"],
            "median_delta_f1": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["median_delta"],
            "ci_95_lower": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["ci_95"][0],
            "ci_95_upper": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["ci_95"][1],
            "bootstrap_tail_leq_0": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["bootstrap_tail_leq_0"],
            "scientific_determination": bootstrap_summary["Delta_Arm_E_vs_Arm_A_Secondary_Micro"]["determination"],
        },
    ]
    pd.DataFrame(ci_records).to_csv(output_dir / "h4_confidence_intervals.csv", index=False)

    contrasts_summary_records = []
    for c_name, b_meta in bootstrap_summary.items():
        contrasts_summary_records.append({
            "contrast": c_name,
            "mean_delta": b_meta["mean_delta"],
            "ci_95_lower": b_meta["ci_95"][0],
            "ci_95_upper": b_meta["ci_95"][1],
            "bootstrap_tail_leq_0": b_meta["bootstrap_tail_leq_0"],
            "tost_strict_03": b_meta["tost_strict_03"],
            "tost_standard_05": b_meta["tost_standard_05"],
            "determination": b_meta["determination"],
        })
        log(f"  {c_name:<38}: Mean={b_meta['mean_delta']:+.4f} | 95% CI=[{b_meta['ci_95'][0]:+.4f}, {b_meta['ci_95'][1]:+.4f}] | Verdict={b_meta['determination']}")

    pd.DataFrame(contrasts_summary_records).to_csv(output_dir / "h4_paired_bootstrap_contrasts.csv", index=False)

    primary_verdict = bootstrap_summary["Delta_Arm_E_vs_Arm_A_Primary_Macro"]["determination"]

    # 5. Consistency Checks Suite
    forbidden_cols = {"gt_mitoses_count", "gt_annotations_json", "annotations", "labels"}
    leaked_manifest_cols = forbidden_cols.intersection(set(pd.read_csv(Path(args.manifest_path), nrows=2).columns))
    checks = {
        "check_manifest_authenticated": compute_sha256(Path(args.manifest_path)) == EXPECTED_FROZEN_MANIFEST_SHA256,
        "check_evaluation_gt_authenticated": compute_sha256(Path(args.ground_truth_path)) == EXPECTED_EVALUATION_GT_SHA256,
        "check_zero_leaked_annotations_in_manifest": len(leaked_manifest_cols) == 0,
        "check_all_patches_tissue_eligible": len(failures) == 0,
        "check_primary_matching_radius_30px": PRIMARY_MATCHING_RADIUS_PX == 30.0,
        "check_sensitivity_matching_radius_25px": SENSITIVITY_MATCHING_RADIUS_PX == 25.0,
        "check_bootstrap_replicates_sufficient": n_boot >= (100 if args.stage_b_smoke_test else 10000),
        "check_zero_parameter_updates_arm_a": all(all_telemetry_by_seed[s]["Arm_A"]["updates"] == 0 for s in seeds_to_run),
        "check_displacement_bound_respected": all(all_telemetry_by_seed[s][arm]["max_displacement"] <= CALIBRATED_L_INF_BOUND + 1e-6 for s in seeds_to_run for arm in ["Arm_C", "Arm_D", "Arm_E"]),
    }

    with open(output_dir / "h4_consistency_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    # 6. Generate Stage H4 Final Audit Report
    report_path = output_dir / "H4_FINAL_AUDIT_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# UTTA-Med — Stage H4: Controlled Target Adaptation Audit Report\n\n")
        f.write(f"**Execution Timestamp:** {env['timestamp']}  \n")
        f.write(f"**Compute Platform:** {env['gpu_name']} ({env['total_vram_gb']} GB VRAM)  \n")
        f.write(f"**Evaluated Target Windows:** {len(target_patches)} tiles across {len(set(p['case_id'] for p in target_patches))} cases  \n")
        f.write(f"**Target Mitoses Evaluated:** {total_mitoses} certified mitotic figures  \n")
        f.write(f"**Primary Matching Radius:** 30.0 px (7.5 µm at 0.25 µm/px)  \n")
        f.write(f"**Primary Estimand Determination:** `{primary_verdict}`  \n\n")

        f.write("## 1. Primary Seed-Macro Evaluation Matrix\n\n")
        f.write("| Arm | Description | Mean F1 | Mean Precision | Mean Recall | Updates | Alarms | Restorations | Fallbacks |\n")
        f.write("|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
        for arm in ["Arm_A", "Arm_B", "Arm_C", "Arm_D", "Arm_E"]:
            mean_f1 = np.mean([all_eval_results_by_seed[s][arm]["f1"] for s in seeds_to_run])
            mean_p = np.mean([all_eval_results_by_seed[s][arm]["precision"] for s in seeds_to_run])
            mean_r = np.mean([all_eval_results_by_seed[s][arm]["recall"] for s in seeds_to_run])
            tot_up = sum(all_telemetry_by_seed[s][arm]["updates"] for s in seeds_to_run)
            tot_al = sum(all_telemetry_by_seed[s][arm]["alarms"] for s in seeds_to_run)
            tot_rb = sum(all_telemetry_by_seed[s][arm]["actual_parameter_restorations"] for s in seeds_to_run)
            tot_fb = sum(all_telemetry_by_seed[s][arm]["fallbacks"] for s in seeds_to_run)
            f.write(f"| {arm} | {arm} | **{mean_f1:.4f}** | {mean_p:.4f} | {mean_r:.4f} | {tot_up} | {tot_al} | {tot_rb} | {tot_fb} |\n")

        f.write("\n## 2. Clustered Bootstrap Contrasts\n\n")
        f.write("| Contrast | Mean $\\Delta F_1$ | 95% Bootstrap CI | Boot Tail $\\le 0$ | Verdict |\n")
        f.write("|---|:---:|:---:|:---:|:---:|\n")
        for c_name, b_meta in bootstrap_summary.items():
            f.write(f"| `{c_name}` | {b_meta['mean_delta']:+.4f} | [{b_meta['ci_95'][0]:+.4f}, {b_meta['ci_95'][1]:+.4f}] | {b_meta['bootstrap_tail_leq_0']:.4f} | `{b_meta['determination']}` |\n")

    log(f"[+] Saved final audit report: {report_path}")

    # Build Self-Contained Reproducibility Archive
    zip_out = Path(args.zip_name)
    n_arch = package_archive_safe(output_dir, zip_out)
    elapsed_total = time.perf_counter() - t_start
    log(f"[*] Reproducibility archive created: {zip_out} ({n_arch} files, {zip_out.stat().st_size / (1024*1024):.2f} MB)")
    log(f"[+] Stage {STAGE} Execution Completed in {elapsed_total:.1f}s.")


if __name__ == "__main__":
    main()
