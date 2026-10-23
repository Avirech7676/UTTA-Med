"""UTTA-Med Training Throughput Benchmark.

Authoritative source-training configuration:
- Backbone: ResNet-18 (ImageNet pretrained, dropout p=0.5)
- Output: 1-logit binary classification
- Loss: BCEWithLogitsLoss
- Optimizer: Adam (lr=1e-4, weight_decay=1e-4)
- Source training centers: {0, 3, 4} (canonical Camelyon17 train split)
- Max epochs: 20, Patience: 5
- Hardware benchmark: Supports CPU, CUDA GPU (with torch.cuda.synchronize() and optional AMP),
  and configurable batch sizes (16, 32, 64, 128).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.camelyon17 import create_camelyon17_datasets
from src.data.paths import resolve_parquet_root
from src.data.transforms import get_camelyon17_train_transform
from src.models.resnet18 import (
    OFFICIAL_DROPOUT_P,
    OFFICIAL_NUM_CLASSES,
    build_resnet18,
)
from src.utils.reproducibility import set_seed


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput on CPU or GPU"
    )
    parser.add_argument("--batches", type=int, default=100, help="Number of measured batches")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup batches")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP) on CUDA",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0 if sys.platform == "win32" else 2,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for Camelyon17 data",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Authoritative learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Authoritative weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=20,
        help="Authoritative maximum epochs (default: 20)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Authoritative early-stopping patience (default: 5)",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable loading ImageNet pre-trained weights for benchmark",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    set_seed(42)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_cuda = (device.type == "cuda") and torch.cuda.is_available()
    use_amp = use_cuda and not args.no_amp

    if args.data_root:
        data_root = resolve_parquet_root(args.data_root)
    else:
        data_root = resolve_parquet_root()

    # Load official source training dataset (centers {0, 3, 4}) with canonical train transform
    source_train_ds, _, _, _ = create_camelyon17_datasets(
        root=data_root,
        train_transform=get_camelyon17_train_transform(),
    )
    dataset = source_train_ds

    required_samples = (args.batches + args.warmup) * args.batch_size
    required_samples = min(required_samples, len(dataset))

    subset = Subset(dataset, range(required_samples))

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    pretrained = not args.no_pretrained
    try:
        model = build_resnet18(
            num_classes=OFFICIAL_NUM_CLASSES,
            pretrained=pretrained,
            dropout_p=OFFICIAL_DROPOUT_P,
        ).to(device)
    except Exception as e:
        print(f"[INFO] Initializing without pretraining weights: {e}")
        model = build_resnet18(
            num_classes=OFFICIAL_NUM_CLASSES,
            pretrained=False,
            dropout_p=OFFICIAL_DROPOUT_P,
        ).to(device)

    model.train()

    # Authoritative 1-logit BCE loss and Adam optimizer (lr=1e-4, wd=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("=" * 60)
    print("UTTA-Med Training Throughput Benchmark")
    print("=" * 60)
    print(f"Backbone      : ResNet-18 (dropout_p={OFFICIAL_DROPOUT_P}, num_classes={OFFICIAL_NUM_CLASSES})")
    print(f"Criterion     : BCEWithLogitsLoss (1-logit binary)")
    print(f"Optimizer     : Adam (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Protocol      : max_epochs={args.max_epochs}, patience={args.patience}")
    print(f"Device        : {device} ({torch.cuda.get_device_name(0) if use_cuda else 'CPU'})")
    print(f"AMP enabled   : {use_amp}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Warmup        : {args.warmup} batches")
    print(f"Measured      : {args.batches} batches")
    print(f"num_workers   : {args.num_workers}")
    print(f"Data root     : {data_root}")
    print()

    iterator = iter(loader)

    # Warmup
    for _ in range(args.warmup):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].view(-1, 1).float().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Synchronize CUDA before starting timer to eliminate launch overhead
    if use_cuda:
        torch.cuda.synchronize()

    measured_batches = 0
    measured_samples = 0

    start = time.perf_counter()

    while measured_batches < args.batches:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].view(-1, 1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        measured_batches += 1
        measured_samples += images.size(0)

    # Synchronize CUDA before stopping timer to capture actual execution completion
    if use_cuda:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    batch_time = elapsed / max(measured_batches, 1)
    batches_per_sec = measured_batches / elapsed
    samples_per_sec = measured_samples / elapsed

    train_samples = 302_436
    batches_per_epoch = train_samples / args.batch_size

    epoch_seconds = train_samples / samples_per_sec
    full_run_seconds = epoch_seconds * args.max_epochs

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Measured time          : {elapsed:.2f} s")
    print(f"Measured samples       : {measured_samples}")
    print(f"Average batch time     : {batch_time:.4f} s")
    print(f"Batches / second       : {batches_per_sec:.4f}")
    print(f"Samples / second       : {samples_per_sec:.2f}")
    print()
    print(f"Batches / epoch        : {batches_per_epoch:.1f}")
    print(f"Estimated epoch time   : {epoch_seconds / 60:.2f} min")
    print(f"Estimated full run ({args.max_epochs} ep): {full_run_seconds / 3600:.2f} hours")
    print("=" * 60)


if __name__ == "__main__":
    main()
