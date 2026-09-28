"""UTTA-Med Training Throughput Benchmark.

Supports CPU, CUDA GPU (with torch.cuda.synchronize() and optional AMP),
and configurable batch sizes (16, 32, 64, 128).
"""

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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from src.data.camelyon17 import Camelyon17Dataset
from src.data.transforms import get_camelyon17_train_transform
from src.models.resnet import build_resnet
from src.utils.reproducibility import set_seed


def parse_args():
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
    return parser.parse_args()


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
        data_root = Path(args.data_root)
    else:
        env_root = os.environ.get("CAMELYON17_ROOT")
        if env_root and Path(env_root).exists():
            data_root = Path(env_root)
        elif Path("/content/Camelyon17-Data/data").exists():
            data_root = Path("/content/Camelyon17-Data/data")
        elif Path(r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data").exists():
            data_root = Path(r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data\data")
        else:
            data_root = Path("data")

    dataset = Camelyon17Dataset(
        root=data_root,
        split="train",
        transform=get_camelyon17_train_transform(96),
        centers=[0, 3, 4],
    )

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

    model = build_resnet(
        num_classes=1,
        pretrained=False,
        dropout_p=0.5,
    ).to(device)

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print("=" * 60)
    print("UTTA-Med Training Throughput Benchmark")
    print("=" * 60)
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

        images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
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
        labels = batch[1].to(device, non_blocking=True)

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
    ten_epoch_seconds = epoch_seconds * 10

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Measured time        : {elapsed:.2f} s")
    print(f"Measured samples     : {measured_samples}")
    print(f"Average batch time   : {batch_time:.4f} s")
    print(f"Batches / second     : {batches_per_sec:.4f}")
    print(f"Samples / second     : {samples_per_sec:.2f}")
    print()
    print(f"Batches / epoch      : {batches_per_epoch:.1f}")
    print(f"Estimated epoch time : {epoch_seconds / 60:.2f} min")
    print(f"Estimated 10 epochs  : {ten_epoch_seconds / 3600:.2f} hours")
    print("=" * 60)


if __name__ == "__main__":
    main()
