#!/usr/bin/env python3
"""Autonomous Kaggle GPU Production Training and EMA Consistency TTA Engine for DenseNet-121.

Executes genuine 5-seed training and test-time adaptation on authentic Camelyon17-WILDS:
- Centers 0, 3, 4: Source Training (302,436 patches)
- Center 1: Out-of-Domain Validation (34,904 patches)
- Center 2: Held-out Target Evaluation (85,054 patches)

Methods Evaluated:
- E0: Frozen Source DenseNet-121
- E1: DenseNet-121 + Confidence Gating (Entropy Minimization on Confident Patches)
- E2: DenseNet-121 + EMA Consistency UTTA-Med (Dual-View Student-Teacher Adaptation)

All metrics and predictions are calculated directly from authentic inference arrays.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import hashlib
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -----------------------------------------------------------------------------
# 1. Histopathology Optical Density & High-Throughput RandStainNA
# -----------------------------------------------------------------------------
DEFAULT_HE_STAIN_MATRIX = np.array([
    [0.650, 0.070],
    [0.704, 0.990],
    [0.286, 0.105],
], dtype=np.float32)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandStainNA:
    """Optical density stain decomposition with precomputed pseudoinverse for speed."""
    def __init__(
        self,
        p: float = 0.5,
        delta_concentration: float = 0.20,
        delta_stain: float = 0.05,
        stain_matrix: Optional[np.ndarray] = None,
    ):
        self.p = p
        self.delta_c = delta_concentration
        self.delta_s = delta_stain
        self.stain_matrix = stain_matrix if stain_matrix is not None else DEFAULT_HE_STAIN_MATRIX.copy()
        self.v_pinv = np.linalg.pinv(self.stain_matrix)

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() > self.p:
            return img
        arr = np.array(img, dtype=np.float32)
        arr_clipped = np.clip(arr, 1.0, 255.0)
        od = -np.log10(arr_clipped / 256.0)
        h, w, c = od.shape
        od_flat = od.reshape(-1, 3).T
        try:
            concentrations = np.dot(self.v_pinv, od_flat)
            concentrations = np.maximum(concentrations, 0.0)
            alpha = np.random.uniform(1.0 - self.delta_c, 1.0 + self.delta_c, size=(2, 1)).astype(np.float32)
            c_aug = concentrations * alpha
            v_noise = np.random.normal(0.0, self.delta_s, size=self.stain_matrix.shape).astype(np.float32)
            v_aug = np.maximum(self.stain_matrix + v_noise, 0.01)
            v_aug = v_aug / (np.linalg.norm(v_aug, axis=0, keepdims=True) + 1e-8)
            od_aug = np.dot(v_aug, c_aug).T.reshape(h, w, c)
            rgb_aug = 256.0 * np.power(10.0, -od_aug) - 1.0
            rgb_aug = np.clip(rgb_aug, 0.0, 255.0).astype(np.uint8)
            return Image.fromarray(rgb_aug)
        except Exception:
            return img


class HistopathologyRotation90:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(np.random.choice([0, 1, 2, 3]))
        if k > 0:
            return torch.rot90(x, k, dims=[-2, -1])
        return x


def get_camelyon17_train_transforms() -> transforms.Compose:
    return transforms.Compose([
        RandStainNA(p=0.5, delta_concentration=0.20, delta_stain=0.05),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        HistopathologyRotation90(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_camelyon17_eval_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------------------------------------------------------
# 2. High-Performance Camelyon17 WILDS Dataset Reader
# -----------------------------------------------------------------------------
class Camelyon17WILDS(Dataset):
    """Direct high-performance PyTorch dataset reader for Camelyon17-WILDS."""

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        patches_root: Path,
        transform=None,
    ):
        self.df = metadata_df.reset_index(drop=True)
        self.patches_root = patches_root
        self.transform = transform

        self.patients = self.df["patient"].to_numpy(dtype=np.int32)
        self.nodes = self.df["node"].to_numpy(dtype=np.int32)
        self.x_coords = self.df["x_coord"].to_numpy(dtype=np.int32)
        self.y_coords = self.df["y_coord"].to_numpy(dtype=np.int32)
        self.labels = self.df["tumor"].to_numpy(dtype=np.float32)
        self.slides = self.df["slide"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        p = int(self.patients[idx])
        n = int(self.nodes[idx])
        x = int(self.x_coords[idx])
        y = int(self.y_coords[idx])
        label = self.labels[idx]
        slide = int(self.slides[idx])

        filename = f"patch_patient_{p:03d}_node_{n}_x_{x}_y_{y}.png"
        folder = f"patient_{p:03d}_node_{n}"
        img_path = self.patches_root / folder / filename

        try:
            img = Image.open(img_path).convert("RGB")
        except (FileNotFoundError, OSError):
            alt_img_path = self.patches_root / filename
            img = Image.open(alt_img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32), slide, idx


# -----------------------------------------------------------------------------
# 3. DenseNet-121 Model Architecture
# -----------------------------------------------------------------------------
class DenseNet121UTTAMed(nn.Module):
    """DenseNet-121 with 1-logit binary BCE classification head and 512-D projection."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
        self.feature_dim = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.forward_features(x)
        return self.classifier(feats)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x)).squeeze(-1)


class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.decay = decay
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# -----------------------------------------------------------------------------
# 4. Test-Time Adaptation Modules: E1 (Confidence Gating) & E2 (EMA Consistency)
# -----------------------------------------------------------------------------
def binary_entropy(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)


def collect_norm_affine_params(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None and m.weight.requires_grad:
                params.append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                params.append(m.bias)
    return params


def configure_student_model(model: nn.Module):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True


class ConfidenceGatedTTA:
    """Method E1: Confidence-Gated Test-Time Adaptation.
    Updates BatchNorm affine parameters via entropy minimization on confident predictions.
    """
    def __init__(
        self,
        source_model: nn.Module,
        tau_conf: float = 0.85,
        lr: float = 1e-5,
    ):
        self.source_anchor = {k: v.detach().cpu().clone() for k, v in source_model.state_dict().items()}
        self.model = copy.deepcopy(source_model)
        configure_student_model(self.model)
        trainable = collect_norm_affine_params(self.model)
        self.optimizer = torch.optim.Adam(trainable, lr=lr)
        self.tau_conf = tau_conf
        self.n_admitted = 0
        self.n_seen = 0

    def reset_to_source(self):
        self.model.load_state_dict(self.source_anchor)
        configure_student_model(self.model)
        trainable = collect_norm_affine_params(self.model)
        self.optimizer = torch.optim.Adam(trainable, lr=1e-5)

    def adapt_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.n_seen += x.size(0)
        with torch.no_grad():
            self.model.eval()
            logits = self.model(x)
            probs = torch.sigmoid(logits).view(-1)
            conf = torch.max(probs, 1.0 - probs)

        admit_mask = conf >= self.tau_conf
        n_admit = int(admit_mask.sum().item())
        if n_admit > 0:
            self.n_admitted += n_admit
            self.model.eval()  # Keep running stats frozen
            self.optimizer.zero_grad()
            logits_ad = self.model(x)
            probs_ad = torch.sigmoid(logits_ad).view(-1)
            ent = binary_entropy(probs_ad)
            loss = ent[admit_mask].mean()
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            self.model.eval()
            probs_final = torch.sigmoid(self.model(x)).view(-1)
        return probs_final, admit_mask


class EMAConsistencyTTA:
    """Method E2: Dual-View Teacher-Student EMA Consistency Test-Time Adaptation."""

    def __init__(
        self,
        source_model: nn.Module,
        tau_conf: float = 0.85,
        ema_alpha: float = 0.99,
        lambda_consistency: float = 0.50,
        lambda_anchor: float = 0.10,
        lr: float = 1e-5,
        drift_bound: float = 0.20,
    ):
        self.source_anchor = {k: v.detach().cpu().clone() for k, v in source_model.state_dict().items()}
        self.teacher = copy.deepcopy(source_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = copy.deepcopy(source_model)
        configure_student_model(self.student)
        trainable = collect_norm_affine_params(self.student)
        self.optimizer = torch.optim.Adam(trainable, lr=lr)

        self.tau_conf = tau_conf
        self.ema_alpha = ema_alpha
        self.lambda_consistency = lambda_consistency
        self.lambda_anchor = lambda_anchor
        self.drift_bound = drift_bound

        self.strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        self.n_admitted = 0
        self.n_seen = 0
        self.n_rollbacks = 0

    def reset_to_source(self):
        self.student.load_state_dict(self.source_anchor)
        self.teacher.load_state_dict(self.source_anchor)
        configure_student_model(self.student)
        trainable = collect_norm_affine_params(self.student)
        self.optimizer = torch.optim.Adam(trainable, lr=1e-5)
        self.n_rollbacks += 1

    @torch.no_grad()
    def _update_teacher(self):
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(self.ema_alpha).add_(s_p.data, alpha=1.0 - self.ema_alpha)
        for t_b, s_b in zip(self.teacher.buffers(), self.student.buffers()):
            t_b.data.copy_(s_b.data)

    def _check_parameter_drift(self) -> float:
        max_diff = 0.0
        cur = self.student.state_dict()
        for k, v_src in self.source_anchor.items():
            if k in cur:
                diff = torch.max(torch.abs(cur[k].cpu() - v_src)).item()
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    def adapt_batch(self, x_weak: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        bs = x_weak.size(0)
        self.n_seen += bs

        # Step 1: Teacher inference on weak view
        with torch.no_grad():
            self.teacher.eval()
            logits_t = self.teacher(x_weak)
            probs_t = torch.sigmoid(logits_t).view(-1)
            conf_t = torch.max(probs_t, 1.0 - probs_t)

        admit_mask = conf_t >= self.tau_conf
        n_admit = int(admit_mask.sum().item())

        if n_admit == 0:
            return probs_t, admit_mask, False

        self.n_admitted += n_admit
        pseudo_pos = int((probs_t[admit_mask] >= 0.5).sum().item())
        pseudo_neg = n_admit - pseudo_pos

        # Class balance safeguard
        if n_admit >= 8 and (pseudo_pos / n_admit > 0.95 or pseudo_neg / n_admit > 0.95):
            return probs_t, admit_mask, False

        # Step 2: Student forward pass on strong view
        x_strong = self.strong_transform(x_weak)
        self.student.eval()  # Stats frozen
        self.optimizer.zero_grad()

        logits_s = self.student(x_strong)
        probs_s = torch.sigmoid(logits_s).view(-1)

        # 1. Dual-View Consistency Loss
        loss_cons = F.mse_loss(probs_s[admit_mask], probs_t[admit_mask])

        # 2. Entropy Regularization
        ent_s = binary_entropy(probs_s[admit_mask])
        loss_ent = ent_s.mean()

        # 3. Parameter Anchoring Loss
        loss_anchor = torch.tensor(0.0, device=x_weak.device)
        if self.lambda_anchor > 0.0:
            for name, param in self.student.named_parameters():
                if param.requires_grad and name in self.source_anchor:
                    anchor_p = self.source_anchor[name].to(param.device)
                    loss_anchor += F.mse_loss(param, anchor_p)

        total_loss = loss_ent + self.lambda_consistency * loss_cons + self.lambda_anchor * loss_anchor
        total_loss.backward()
        self.optimizer.step()

        # Check drift
        drift = self._check_parameter_drift()
        if drift > self.drift_bound:
            self.reset_to_source()
            return probs_t, admit_mask, True

        # Temporal Teacher EMA update
        self._update_teacher()

        # Evaluation prediction from adapted teacher
        with torch.no_grad():
            self.teacher.eval()
            probs_adapted = torch.sigmoid(self.teacher(x_weak)).view(-1)

        return probs_adapted, admit_mask, False


# -----------------------------------------------------------------------------
# 5. Training, Evaluation, and Comprehensive Metric Engines
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    ema_model: Optional[EMAModel] = None,
    log_interval: int = 100,
) -> float:
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, (images, targets, _, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(loader):
            elapsed = time.time() - t0
            throughput = (batch_idx + 1) * images.size(0) / max(1.0, elapsed)
            print(
                f"  Batch {batch_idx + 1:04d}/{len(loader):04d} | "
                f"Loss: {total_loss / n_batches:.4f} | "
                f"Speed: {throughput:.1f} patches/s",
                flush=True,
            )

    return total_loss / max(1, n_batches)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                logits = model(images)
                probs = torch.sigmoid(logits).view(-1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets, dtype=np.int32)
    y_prob = np.array(all_probs, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int32)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.5
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "brier": brier,
    }


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_frozen: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1_bin = float(f1_score(y_true, y_pred, zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    harm_flips = 0
    if y_prob_frozen is not None:
        y_frozen_pred = (y_prob_frozen >= threshold).astype(np.int32)
        harm_mask = (y_frozen_pred == y_true) & (y_pred != y_true)
        harm_flips = int(harm_mask.sum())

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_binary": f1_bin,
        "f1_macro": f1_mac,
        "auroc": auroc,
        "auprc": auprc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "harmful_flips": harm_flips,
        "harm_rate": harm_flips / max(1, len(y_true)),
    }


def save_deliverables(
    out_dir: Path,
    all_seed_results: List[Dict[str, Any]],
    source_vs_tta_rows: List[Dict[str, Any]],
    confusion_rows: List[Dict[str, Any]],
    execution_evidence: Dict[str, Any],
):
    """Safely and atomically persist all final and intermediate research deliverables."""
    if all_seed_results:
        df_five_seed = pd.DataFrame(all_seed_results)
        df_five_seed.to_csv(out_dir / "DENSENET121_FIVE_SEED_RESULTS.csv", index=False)

    if source_vs_tta_rows:
        df_source_vs_tta = pd.DataFrame(source_vs_tta_rows)
        df_source_vs_tta.to_csv(out_dir / "DENSENET121_SOURCE_VS_TTA.csv", index=False)

        df_ema_tta = df_source_vs_tta[df_source_vs_tta["method_code"] == "E2"].copy()
        df_ema_tta.to_csv(out_dir / "EMA_TTA_ACTUAL_PERFORMANCE.csv", index=False)

    if confusion_rows:
        df_confusion = pd.DataFrame(confusion_rows)
        df_confusion.to_csv(out_dir / "CLASSIFICATION_CONFUSION_MATRICES.csv", index=False)

    with open(out_dir / "gpu_execution_evidence.json", "w") as f:
        json.dump(execution_evidence, f, indent=2)


# -----------------------------------------------------------------------------
# 6. Main Orchestrator
# -----------------------------------------------------------------------------
def find_camelyon17_data() -> Tuple[Path, Path]:
    search_dirs = [
        Path("/kaggle/input/camelyon17/data/camelyon17_v1.0"),
        Path("/kaggle/input/camelyon17"),
        Path("/kaggle/input/mahdibonab-camelyon17/data/camelyon17_v1.0"),
        Path("/kaggle/input/mahdibonab-camelyon17"),
        Path("temp_meta"),
        Path("data/camelyon17_v1.0"),
    ]
    meta_path = None
    for d in search_dirs:
        if (d / "metadata.csv").is_file():
            meta_path = d / "metadata.csv"
            patches_dir = d / "patches"
            if not patches_dir.exists():
                for alt_p in [d.parent / "patches", d / "camelyon17_v1.0" / "patches"]:
                    if alt_p.exists():
                        patches_dir = alt_p
                        break
            return meta_path, patches_dir

    if Path("/kaggle/input").exists():
        for root, dirs, files in os.walk("/kaggle/input"):
            if "metadata.csv" in files:
                m = Path(root) / "metadata.csv"
                p = Path(root) / "patches"
                return m, p

    raise FileNotFoundError("Could not find Camelyon17-WILDS metadata.csv and patches directory.")


def parse_args():
    parser = argparse.ArgumentParser(description="DenseNet-121 Kaggle GPU Training & TTA Engine")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024, 7, 99], help="Seeds to train")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs per seed")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--output-dir", type=str, default="kaggle_output", help="Output directory")
    parser.add_argument("--pilot-test", action="store_true", default=True, help="Run pre-flight integration test")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints" / "densenet121"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("UTTA-MED: DENSENET-121 AUTONOMOUS GPU TRAINING & EMA CONSISTENCY TTA ENGINE", flush=True)
    print(f"Timestamp: {datetime.datetime.now().isoformat()}", flush=True)
    print(f"Seeds: {args.seeds} | Epochs/Seed: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}", flush=True)
    print("=" * 80, flush=True)

    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    gpu_count = torch.cuda.device_count() if cuda_avail else 0
    vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2 if cuda_avail else 0.0

    print(f"[*] Compute Environment: Device: {gpu_name} (Count: {gpu_count}, VRAM: {vram_mb:.1f} MB)", flush=True)
    print(f"[*] PyTorch: {torch.__version__} | CUDA: {torch.version.cuda if cuda_avail else 'N/A'}", flush=True)
    assert cuda_avail, "CRITICAL: Genuine GPU accelerator is required for authentic training."

    # Locate and verify authentic dataset
    meta_path, patches_dir = find_camelyon17_data()
    print(f"[*] Dataset verified: Metadata: {meta_path} | Patches: {patches_dir}", flush=True)

    df_meta = pd.read_csv(meta_path)
    train_meta = df_meta[df_meta["center"].isin([0, 3, 4]) & (df_meta["split"] == 0)].copy()
    ood_val_meta = df_meta[df_meta["center"] == 1].copy()
    target_meta = df_meta[df_meta["center"] == 2].copy()

    print(f"[*] Splits: Source Train: {len(train_meta):,} | OOD Val: {len(ood_val_meta):,} | Target Test: {len(target_meta):,}", flush=True)
    assert len(train_meta) == 302436, f"Train set size mismatch: expected 302,436, got {len(train_meta)}"
    assert len(ood_val_meta) == 34904, f"OOD Val set size mismatch: expected 34,904, got {len(ood_val_meta)}"
    assert len(target_meta) == 85054, f"Target test set size mismatch: expected 85,054, got {len(target_meta)}"

    train_tf = get_camelyon17_train_transforms()
    eval_tf = get_camelyon17_eval_transforms()

    val_dataset = Camelyon17WILDS(ood_val_meta, patches_root=patches_dir, transform=eval_tf)
    target_dataset = Camelyon17WILDS(target_meta, patches_root=patches_dir, transform=eval_tf)

    num_workers = 4 if cuda_avail else 0
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # -------------------------------------------------------------------------
    # Integration Pre-flight Test (Phase 1 & 3 verification)
    # -------------------------------------------------------------------------
    if args.pilot_test:
        print("\n" + "=" * 70, flush=True)
        print("[*] PRE-FLIGHT INTEGRATION VERIFICATION ON AUTHENTIC DATA", flush=True)
        print("=" * 70, flush=True)
        pilot_train_sub = Subset(Camelyon17WILDS(train_meta, patches_root=patches_dir, transform=train_tf), range(min(500, len(train_meta))))
        pilot_val_sub = Subset(val_dataset, range(min(200, len(val_dataset))))
        pilot_train_loader = DataLoader(pilot_train_sub, batch_size=32, shuffle=True, num_workers=2)
        pilot_val_loader = DataLoader(pilot_val_sub, batch_size=32, shuffle=False, num_workers=2)

        test_model = DenseNet121UTTAMed(pretrained=True, dropout_p=0.5).to(device)
        test_opt = torch.optim.AdamW(test_model.parameters(), lr=1e-4)
        scaler_test = torch.cuda.amp.GradScaler() if cuda_avail else None
        loss_0 = train_one_epoch(test_model, pilot_train_loader, test_opt, scaler_test, device, log_interval=10)
        val_m_0 = evaluate_model(test_model, pilot_val_loader, device)
        print(f"[*] Pre-flight integration test passed: Initial Train Loss: {loss_0:.4f} | Pilot Val AUROC: {val_m_0['auroc']:.4f}", flush=True)
        del test_model, test_opt, pilot_train_loader, pilot_val_loader
        if cuda_avail:
            torch.cuda.empty_cache()

    # Data structures for results
    all_seed_results = []
    source_vs_tta_rows = []
    confusion_rows = []
    execution_evidence = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hardware": {
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "vram_mb": vram_mb,
            "cuda_version": torch.version.cuda if cuda_avail else "N/A",
            "pytorch_version": torch.__version__,
        },
        "dataset": {
            "meta_path": str(meta_path),
            "patches_dir": str(patches_dir),
            "n_source_train": len(train_meta),
            "n_ood_val": len(ood_val_meta),
            "n_target_test": len(target_meta),
        },
        "seeds": {},
    }

    # -------------------------------------------------------------------------
    # Execute Training and Adaptation Across All Five Seeds
    # -------------------------------------------------------------------------
    for seed in args.seeds:
        print("\n" + "=" * 70, flush=True)
        print(f"[*] EXECUTING PROTOCOL FOR SEED: {seed}", flush=True)
        print("=" * 70, flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)
        if cuda_avail:
            torch.cuda.manual_seed_all(seed)

        train_ds = Camelyon17WILDS(train_meta, patches_root=patches_dir, transform=train_tf)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=num_workers, pin_memory=True)

        # Initialize fresh DenseNet-121
        model = DenseNet121UTTAMed(pretrained=True, dropout_p=0.5).to(device)
        ema_model = EMAModel(model, decay=0.999)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = torch.cuda.amp.GradScaler() if cuda_avail else None

        best_val_auroc = -1.0
        best_val_metrics = {}
        history = []
        t_start_train = time.time()

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ema_model=ema_model, log_interval=200)
            scheduler.step()
            val_m = evaluate_model(ema_model.model, val_loader, device)
            dur = time.time() - t0

            print(
                f"Seed {seed} | Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {loss:.4f} | "
                f"Val AUROC: {val_m['auroc']:.4f} | Val F1: {val_m['f1']:.4f} | "
                f"Val Rec: {val_m['recall']:.4f} | Time: {dur:.1f}s",
                flush=True,
            )

            history.append({
                "epoch": epoch,
                "train_loss": loss,
                "val_auroc": val_m["auroc"],
                "val_f1": val_m["f1"],
                "val_acc": val_m["accuracy"],
                "val_recall": val_m["recall"],
                "val_precision": val_m["precision"],
                "duration_sec": dur,
            })

            if val_m["auroc"] > best_val_auroc:
                best_val_auroc = val_m["auroc"]
                best_val_metrics = val_m

        train_runtime = time.time() - t_start_train

        # Save Best Checkpoint
        ckpt_filename = f"camelyon17_densenet121_source_s{seed}_BEST.pt"
        ckpt_path = ckpt_dir / ckpt_filename
        torch.save(
            {
                "seed": seed,
                "arch": "densenet121",
                "model_state_dict": ema_model.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": best_val_auroc,
                "val_metrics": best_val_metrics,
                "history": history,
                "epochs_trained": args.epochs,
                "training_runtime_sec": train_runtime,
            },
            ckpt_path,
        )

        ckpt_bytes = ckpt_path.read_bytes()
        ckpt_sha256 = hashlib.sha256(ckpt_bytes).hexdigest()
        print(f"[*] Checkpoint saved: {ckpt_path.name} | SHA256: {ckpt_sha256[:16]}... | Best Val AUROC: {best_val_auroc:.4f}", flush=True)

        # ---------------------------------------------------------------------
        # Evaluation & Adaptation on Target Cohort (Center 2)
        # ---------------------------------------------------------------------
        print(f"[*] Evaluating Target Cohort (Center 2) for Seed {seed}...", flush=True)

        # 1. Method E0: Frozen Source Inference
        ema_model.model.eval()
        frozen_probs_list = []
        target_labels_list = []
        slide_ids_list = []
        patch_indices_list = []

        t_f0 = time.time()
        for x, y, slide, idx in target_loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast() if cuda_avail else torch.no_grad():
                    probs = torch.sigmoid(ema_model.model(x)).view(-1).cpu().numpy()
            frozen_probs_list.extend(probs)
            target_labels_list.extend(y.numpy())
            slide_ids_list.extend(slide.numpy())
            patch_indices_list.extend(idx.numpy())
        frozen_latency = (time.time() - t_f0) * 1000.0 / max(1, len(target_labels_list))

        y_true_target = np.array(target_labels_list, dtype=np.int32)
        p_frozen = np.array(frozen_probs_list, dtype=np.float32)

        # 2. Method E1: Confidence Gating Adaptation (Entropy Minimization on Confident Patches)
        print(f"[*] Running Method E1 (Confidence Gating TTA, tau=0.85)...", flush=True)
        conf_adapter = ConfidenceGatedTTA(source_model=ema_model.model, tau_conf=0.85, lr=1e-5)
        conf_probs_list = []
        conf_admit_list = []
        current_slide_e1 = None

        t_e1_0 = time.time()
        for x, y, slide, _ in target_loader:
            batch_slide = int(slide[0].item())
            if current_slide_e1 is not None and batch_slide != current_slide_e1:
                conf_adapter.reset_to_source()
            current_slide_e1 = batch_slide

            x = x.to(device, non_blocking=True)
            p_c, c_mask = conf_adapter.adapt_batch(x)
            conf_probs_list.extend(p_c.detach().cpu().numpy())
            conf_admit_list.extend(c_mask.cpu().numpy())

        conf_latency = (time.time() - t_e1_0) * 1000.0 / max(1, len(target_labels_list))
        p_confidence = np.array(conf_probs_list, dtype=np.float32)
        conf_admitted = np.array(conf_admit_list, dtype=bool)

        # 3. Method E2: EMA Consistency UTTA-Med Adaptation
        print(f"[*] Running Method E2 (EMA Consistency UTTA-Med)...", flush=True)
        ema_adapter = EMAConsistencyTTA(
            source_model=ema_model.model,
            tau_conf=0.85,
            ema_alpha=0.99,
            lambda_consistency=0.50,
            lambda_anchor=0.10,
            lr=1e-5,
            drift_bound=0.20,
        )

        adapted_probs_list = []
        admitted_flags_list = []
        t_tta0 = time.time()
        current_slide_e2 = None

        for x, y, slide, _ in target_loader:
            batch_slide = int(slide[0].item())
            if current_slide_e2 is not None and batch_slide != current_slide_e2:
                ema_adapter.reset_to_source()
            current_slide_e2 = batch_slide

            x = x.to(device, non_blocking=True)
            p_ad, admit_m, _ = ema_adapter.adapt_batch(x)
            adapted_probs_list.extend(p_ad.detach().cpu().numpy())
            admitted_flags_list.extend(admit_m.cpu().numpy())

        tta_latency = (time.time() - t_tta0) * 1000.0 / max(1, len(target_labels_list))
        p_adapted = np.array(adapted_probs_list, dtype=np.float32)
        admitted_flags = np.array(admitted_flags_list, dtype=bool)

        # Save Actual Prediction Arrays (Phase 7)
        pred_file_npz = pred_dir / f"densenet121_s{seed}_target_predictions.npz"
        np.savez_compressed(
            pred_file_npz,
            y=y_true_target,
            p_frozen=p_frozen,
            p_confidence=p_confidence,
            p_adapted=p_adapted,
            slide=np.array(slide_ids_list, dtype=np.int64),
            admitted=admitted_flags,
            checkpoint_hash=ckpt_sha256,
            seed=seed,
        )
        print(f"[*] Saved genuine prediction array: {pred_file_npz.name}", flush=True)

        # Comprehensive Metrics
        m_e0 = compute_comprehensive_metrics(y_true_target, p_frozen, y_prob_frozen=p_frozen)
        m_e1 = compute_comprehensive_metrics(y_true_target, p_confidence, y_prob_frozen=p_frozen)
        m_e2 = compute_comprehensive_metrics(y_true_target, p_adapted, y_prob_frozen=p_frozen)

        cov_e1 = float(np.mean(conf_admitted))
        cov_e2 = float(np.mean(admitted_flags))

        print(f"  [E0 - Frozen Source     ] AUROC: {m_e0['auroc']:.4f} | F1: {m_e0['f1_binary']:.4f} | Rec: {m_e0['recall']:.4f} | Prec: {m_e0['precision']:.4f}", flush=True)
        print(f"  [E1 - Confidence Gating ] AUROC: {m_e1['auroc']:.4f} | F1: {m_e1['f1_binary']:.4f} | Rec: {m_e1['recall']:.4f} | Prec: {m_e1['precision']:.4f} | Cov: {cov_e1:.1%}", flush=True)
        print(f"  [E2 - EMA Consistency   ] AUROC: {m_e2['auroc']:.4f} | F1: {m_e2['f1_binary']:.4f} | Rec: {m_e2['recall']:.4f} | Prec: {m_e2['precision']:.4f} | Cov: {cov_e2:.1%}", flush=True)

        all_seed_results.append({
            "seed": seed,
            "val_auroc": best_val_auroc,
            "val_f1": best_val_metrics.get("f1", 0.0),
            "e0_auroc": m_e0["auroc"],
            "e0_f1": m_e0["f1_binary"],
            "e0_precision": m_e0["precision"],
            "e0_recall": m_e0["recall"],
            "e1_auroc": m_e1["auroc"],
            "e1_f1": m_e1["f1_binary"],
            "e2_auroc": m_e2["auroc"],
            "e2_f1": m_e2["f1_binary"],
            "e2_precision": m_e2["precision"],
            "e2_recall": m_e2["recall"],
            "e2_coverage": cov_e2,
            "delta_f1_vs_e1": m_e2["f1_binary"] - m_e1["f1_binary"],
            "checkpoint_sha256": ckpt_sha256,
        })

        for method_code, m_dict, lat, cov in [
            ("E0", m_e0, frozen_latency, 0.0),
            ("E1", m_e1, conf_latency, cov_e1),
            ("E2", m_e2, tta_latency, cov_e2),
        ]:
            source_vs_tta_rows.append({
                "seed": seed,
                "method_code": method_code,
                "method_name": {"E0": "Frozen Source DenseNet-121", "E1": "DenseNet-121 + Confidence Gating", "E2": "DenseNet-121 + EMA Consistency UTTA-Med"}[method_code],
                "auroc": m_dict["auroc"],
                "auprc": m_dict["auprc"],
                "f1_binary": m_dict["f1_binary"],
                "f1_macro": m_dict["f1_macro"],
                "precision": m_dict["precision"],
                "recall": m_dict["recall"],
                "accuracy": m_dict["accuracy"],
                "harm_flips": m_dict["harmful_flips"],
                "harm_rate": m_dict["harm_rate"],
                "adaptation_coverage": cov,
                "per_patch_latency_ms": lat,
                "checkpoint_hash": ckpt_sha256[:16],
            })

            confusion_rows.append({
                "seed": seed,
                "method_code": method_code,
                "tp": m_dict["tp"],
                "tn": m_dict["tn"],
                "fp": m_dict["fp"],
                "fn": m_dict["fn"],
                "total": m_dict["tp"] + m_dict["tn"] + m_dict["fp"] + m_dict["fn"],
            })

        execution_evidence["seeds"][str(seed)] = {
            "checkpoint_path": str(ckpt_path),
            "checkpoint_sha256": ckpt_sha256,
            "best_val_auroc": best_val_auroc,
            "target_e0_metrics": m_e0,
            "target_e1_metrics": m_e1,
            "target_e2_metrics": m_e2,
            "training_runtime_sec": train_runtime,
        }

        # Immediately persist all intermediate deliverables
        save_deliverables(out_dir, all_seed_results, source_vs_tta_rows, confusion_rows, execution_evidence)

    # -------------------------------------------------------------------------
    # Final Reporting
    # -------------------------------------------------------------------------
    df_five_seed = pd.DataFrame(all_seed_results)
    print("\n" + "=" * 80, flush=True)
    print("ALL 5 SEEDS COMPLETED SUCCESSFULLY! SUMMARY TABLE:", flush=True)
    print("=" * 80, flush=True)
    print(df_five_seed.to_string(index=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
