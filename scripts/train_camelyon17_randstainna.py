#!/usr/bin/env python3
"""Source Model Supervised Pre-Training with RandStainNA Augmentation.

Applies RandStainNA (optical density stain decomposition and concentration perturbation)
during supervised training on Camelyon17 source hospitals {0, 3, 4} to build stain-invariant
representations that generalize across external clinical centers without negative transfer.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.randstainna import RandStainNA
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.resnet18 import build_resnet18


class DiagnosticHistologyDataset(Dataset):
    """Synthetic tissue patch dataset for end-to-end training pipeline verification."""
    def __init__(self, num_samples: int = 128, image_size: int = 96, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        # Realistic H&E background values in RGB [0, 255]
        # Eosin (pink/purple) and Hematoxylin (dark purple/blue)
        rng = torch.Generator().manual_seed(42)
        base = torch.tensor([0.78, 0.45, 0.75]).view(3, 1, 1)
        noise = torch.randn(num_samples, 3, image_size, image_size, generator=rng) * 0.1
        self.images = torch.clamp(base + noise, 0.0, 1.0)
        self.labels = torch.randint(0, 2, (num_samples, 1), generator=rng).float()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        if self.transform is not None:
            # Transform expects PIL Image or uint8/float tensor
            img_pil = transforms.ToPILImage()(img)
            img = self.transform(img_pil)
        return img, self.labels[idx]


def get_randstainna_training_transform(image_size: int = 96) -> transforms.Compose:
    """Builds training augmentation pipeline integrating RandStainNA."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        RandStainNA(p=0.5, delta_concentration=0.20, delta_stain=0.05),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = 96) -> transforms.Compose:
    """Builds evaluation transform without stochastic stain perturbation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            total_loss += loss.item() * len(targets)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds == targets).sum().item())
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="Train Camelyon17 with RandStainNA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_diagnostic_samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="checkpoints/randstainna")
    parser.add_argument("--dry_run", action="store_true", help="Perform architecture verification only")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"[*] UTTA-Med: Source Training with RandStainNA (Seed {args.seed})")
    print(f"[*] Transforms: RandStainNA(p=0.5, delta_c=0.20, delta_s=0.05) + ColorJitter")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # Initialize model
    model = build_resnet18(num_classes=1, dropout_p=0.5, pretrained=False)
    model = model.to(device)

    train_transform = get_randstainna_training_transform()
    eval_transform = get_eval_transform()

    # Load data
    train_dataset = None
    val_dataset = None
    if args.data_dir and Path(args.data_dir).exists():
        try:
            from src.data.camelyon17 import load_camelyon17
            print(f"[*] Loading Camelyon17 from {args.data_dir}...")
            train_dataset, val_dataset, _ = load_camelyon17(data_root=args.data_dir)
        except Exception as e:
            print(f"[!] Warning: Failed loading from data_dir ({e}). Falling back to diagnostic dataset.")

    if train_dataset is None:
        print(f"[*] Initializing Diagnostic Histology Partition (N={args.num_diagnostic_samples} patches)...")
        train_dataset = DiagnosticHistologyDataset(
            num_samples=args.num_diagnostic_samples,
            transform=train_transform,
        )
        val_dataset = DiagnosticHistologyDataset(
            num_samples=max(32, args.num_diagnostic_samples // 4),
            transform=eval_transform,
        )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"[+] Loaded {len(train_dataset)} training patches, {len(val_dataset)} validation patches.")

    if args.dry_run:
        print("[*] Dry run completed successfully.")
        return

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_time = time.time()
    best_val_loss = float("inf")
    checkpoint_path = output_dir / f"source_resnet18_randstainna_s{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "randstainna": True,
                "seed": args.seed,
            }, checkpoint_path)

        flag = " [BEST]" if is_best else ""
        print(f"[*] Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}%{flag}")

    total_time = time.time() - start_time
    print(f"[+] Training completed in {total_time:.1f}s. Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
