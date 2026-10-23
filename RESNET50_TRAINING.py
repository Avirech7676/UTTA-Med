#!/usr/bin/env python3
"""ResNet-50 Pretrained Supervised Training Pipeline (Model B).

Architecture:
- Pretrained ResNet-50 backbone (ImageNet-1K V2 weights)
- 2048-D bottleneck features
- Modular projection head: Linear(2048 -> 512) -> ReLU -> Dropout(p=0.5) -> Linear(512 -> 1)
- Binary 1-logit BCE loss
- RandStainNA optical density stain decomposition & concentration perturbation
- Center-preserving augmentations (90-deg rotations, flips)
- AdamW optimizer (lr=1e-4, weight_decay=1e-2)
- Linear warmup + Cosine Annealing scheduler
- Gradient clipping (max_norm = 1.0)
- Exponential Moving Average (EMA) weights
- Mixed-precision training via torch.cuda.amp
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.randstainna import RandStainNA
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

OFFICIAL_DROPOUT_P = 0.5
OFFICIAL_IMAGE_SIZE = 96


class ResNet50UTTAMed(nn.Module):
    """ResNet-50 with 1-logit binary BCE classification head and 512-D bottleneck projection."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_p: float = OFFICIAL_DROPOUT_P,
        num_classes: int = 1,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes),
        )
        self.dropout_p = dropout_p
        self.num_classes = num_classes

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


class HistopathologyRotation90:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = np.random.choice([0, 1, 2, 3])
        if k > 0:
            return torch.rot90(x, k, dims=[-2, -1])
        return x


def get_training_transforms(image_size: int = OFFICIAL_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandStainNA(p=0.5, delta_concentration=0.20, delta_stain=0.05),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        HistopathologyRotation90(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(image_size: int = OFFICIAL_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SyntheticCamelyonDataset(Dataset):
    def __init__(self, num_samples: int = 256, image_size: int = 96, transform=None, seed: int = 42):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        rng = torch.Generator().manual_seed(seed)
        base = torch.tensor([0.80, 0.48, 0.72]).view(3, 1, 1)
        noise = torch.randn(num_samples, 3, image_size, image_size, generator=rng) * 0.08
        self.images = torch.clamp(base + noise, 0.0, 1.0)
        self.labels = torch.randint(0, 2, (num_samples, 1), generator=rng).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        if self.transform is not None:
            pil_img = transforms.ToPILImage()(img)
            img = self.transform(pil_img)
        return img, self.labels[idx]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    ema_model: Optional[EMAModel] = None,
    max_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

        if ema_model is not None:
            ema_model.update(model)

        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits).view(-1).cpu().numpy()
        all_probs.extend(probs)
        all_targets.extend(targets.view(-1).cpu().numpy())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train Pretrained ResNet-50 (Model B)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (32 for 2048-D memory)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="AdamW weight decay")
    parser.add_argument("--data-root", type=str, default=None, help="Path to Camelyon17 dataset")
    parser.add_argument("--output-dir", type=str, default="checkpoints/resnet50", help="Output directory")
    parser.add_argument("--smoke", action="store_true", help="Run quick CPU smoke test")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("UTTA-Med: Pretrained ResNet-50 (Model B) Supervised Training Pipeline")
    print(f"Seed: {args.seed} | Epochs: {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print("=" * 70)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    print(f"Active Device: {device}")

    train_tf = get_training_transforms()
    eval_tf = get_eval_transforms()

    if args.smoke or args.data_root is None:
        print("[Notice] Using synthetic Camelyon17 validation stream for pipeline verification.")
        train_ds = SyntheticCamelyonDataset(num_samples=128 if args.smoke else 1024, transform=train_tf, seed=args.seed)
        val_ds = SyntheticCamelyonDataset(num_samples=64 if args.smoke else 256, transform=eval_tf, seed=args.seed + 1)
    else:
        from src.data.camelyon17 import create_camelyon17_datasets
        train_ds, val_ds, _ = create_camelyon17_datasets(
            data_root=args.data_root,
            train_transform=train_tf,
            eval_transform=eval_tf,
            source_centers=[0, 3, 4],
            val_center=1,
            target_center=2,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = ResNet50UTTAMed(pretrained=not args.smoke, dropout_p=OFFICIAL_DROPOUT_P).to(device)
    ema_model = EMAModel(model, decay=0.999)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_auroc = -1.0
    best_checkpoint_path = out_dir / f"camelyon17_resnet50_source_s{args.seed}_BEST.pt"

    for epoch in range(1, (2 if args.smoke else args.epochs) + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, ema_model=ema_model)
        scheduler.step()

        val_metrics = evaluate_model(ema_model.model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_metrics['auroc']:.4f} | Val F1: {val_metrics['f1']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | Time: {elapsed:.1f}s"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            torch.save(
                {
                    "epoch": epoch,
                    "seed": args.seed,
                    "model_state_dict": ema_model.model.state_dict(),
                    "val_auroc": best_val_auroc,
                    "val_f1": val_metrics["f1"],
                    "val_metrics": val_metrics,
                    "arch": "resnet50",
                    "pipeline": "ResNet50_AdamW_RandStainNA_EMA",
                },
                best_checkpoint_path,
            )
            print(f" -> Saved new best checkpoint to {best_checkpoint_path}")

    print("=" * 70)
    print(f"Training Complete. Best Validation AUROC: {best_val_auroc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
