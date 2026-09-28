"""Official Source ResNet-18 Training Script for UTTA-Med.

Authoritative Frozen Protocol:
- Config: configs/FINAL/source_resnet18.yaml
- Backbone: ResNet-18 (ImageNet-1K pretrained)
- Classification: 1-logit BCE with sigmoid
- Regularization: MC-Dropout (dropout_p = 0.5)
- Optimizer: Adam (lr = 1e-4, weight_decay = 1e-4)
- Early Stopping: OOD Validation (Center 1) AUROC, patience = 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.camelyon17 import create_camelyon17_datasets
from src.data.paths import resolve_parquet_root
from src.data.transforms import get_camelyon17_eval_transform, get_camelyon17_train_transform
from src.models.resnet18 import ResNet18UTTAMed, build_resnet18 as build_resnet
from src.training.losses import get_loss_function
from src.training.trainer import SourceTrainer
from src.utils.reproducibility import set_seed, seed_worker


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "git_commit_unknown"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train source ResNet-18 model on Camelyon17"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/FINAL/source_resnet18.yaml",
        help="Path to YAML training configuration",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run Phase 2.4 CPU smoke test (1024 train samples, 256 val samples, 1 epoch, batch size 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save experiment artifacts",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override dataset root path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda or cpu)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP) on CUDA",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine experiment mode & parameters
    seed = args.seed if args.seed is not None else config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    if args.smoke:
        print("=" * 70)
        print("UTTA-Med: Phase 2.4 CPU Smoke Training")
        print("=" * 70)
        epochs = 1
        batch_size = 16
        train_samples_count = 1024
        val_samples_count = 256
        exp_dir = Path("experiments/smoke")
        is_smoke = True
    else:
        epochs = args.epochs if args.epochs is not None else config.get("training", {}).get("epochs", 20)
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else config.get("training", {}).get("batch_size", 64)
        )
        train_samples_count = None
        val_samples_count = None
        exp_name = config.get("experiment", {}).get("name", "source_resnet18")
        exp_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path("experiments") / f"{exp_name}_s{seed}_v1"
        )
        is_smoke = False

    lr = args.lr if args.lr is not None else config.get("training", {}).get("learning_rate", 1e-4)
    weight_decay = config.get("training", {}).get("weight_decay", 1e-4)
    num_classes = config.get("model", {}).get("num_classes", 1)
    dropout_p = config.get("model", {}).get("dropout_p", 0.5)
    pretrained = config.get("model", {}).get("pretrained", True)

    if args.data_root:
        data_root = args.data_root
    elif Path("/content/Camelyon17-Data/data").exists():
        data_root = "/content/Camelyon17-Data/data"
    elif "dataset" in config and "root" in config["dataset"]:
        data_root = config["dataset"]["root"]
    else:
        data_root = resolve_parquet_root(PROJECT_ROOT)

    exp_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = exp_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Artifacts output directory: {exp_dir}")
    print(f"Seed: {seed}")
    print(f"Batch size: {batch_size} | Epochs: {epochs}")

    # Build transforms
    train_transform = get_camelyon17_train_transform()
    eval_transform = get_camelyon17_eval_transform()

    print(f"Loading Camelyon17 dataset from: {data_root}")
    source_train_ds, id_val_ds, ood_val_ds, ood_test_ds = create_camelyon17_datasets(
        root=data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
        return_metadata=True,
    )

    # Subsetting for smoke testing
    g = torch.Generator().manual_seed(seed)
    if is_smoke:
        print(f"Preparing deterministic subset: {train_samples_count} train, {val_samples_count} val samples...")
        # Permute deterministically to ensure balanced class sampling across slides
        train_indices = torch.randperm(len(source_train_ds), generator=g)[:train_samples_count].tolist()
        val_indices = torch.randperm(len(ood_val_ds), generator=g)[:val_samples_count].tolist()

        train_ds = Subset(source_train_ds, train_indices)
        val_ds = Subset(ood_val_ds, val_indices)
    else:
        train_ds = source_train_ds
        val_ds = ood_val_ds

    # DataLoaders (num_workers=0 safe for Windows parquet loading)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    val_eval_batch_size = config.get("evaluation", {}).get("batch_size", batch_size)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size if is_smoke else val_eval_batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
    )

    # Model instantiation
    print("Instantiating ResNet-18 model...")
    model = build_resnet(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
    )

    # Optimizer & Criterion (Frozen protocol specifies Adam)
    opt_name = config.get("training", {}).get("optimizer", "adam").lower()
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = get_loss_function(
        loss_name="cross_entropy" if num_classes >= 2 else "bce",
        num_classes=num_classes,
    )

    compute_cfg = config.get("compute", {})
    requested_device = args.device if args.device is not None else compute_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    use_amp = bool(compute_cfg.get("amp", True) and (not args.no_amp) and (device.type == "cuda") and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device        : {device} ({gpu_name})")
    print(f"AMP enabled   : {use_amp}")

    # Explicitly update config compute section so saved config.yaml is accurate
    config["compute"] = {
        "device": str(device),
        "amp": use_amp,
    }

    patience = config.get("training", {}).get("early_stop_patience", 5)

    trainer = SourceTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_dir=exp_dir,
        patience=patience,
        metric_for_best="auroc",
        use_amp=use_amp,
    )

    print("\nStarting training loop...")
    results = trainer.train(epochs=epochs)

    # Perform final evaluation on validation set
    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(val_loader)
    print("Validation Metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")

    commit_hash = get_git_commit()

    # Save Checkpoint as checkpoint.pt in exp_dir with full reproducibility metadata
    checkpoint_path = exp_dir / "checkpoint.pt"
    trainer.save_checkpoint(
        filepath=checkpoint_path,
        epoch=results["best_epoch"],
        metrics=val_metrics,
        extra={
            "config": config,
            "seed": seed,
            "pytorch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else "None",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "batch_size": batch_size,
            "amp": use_amp,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "pretrained": pretrained,
            "dropout_p": dropout_p,
            "git_commit": commit_hash,
            "is_smoke": is_smoke,
            "timestamp": datetime.now().isoformat(),
        },
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    # Generate experiment artifacts
    print("\nGenerating required experiment artifacts...")
    # 1. config.yaml
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 2. command.txt
    with open(exp_dir / "command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # 3. git_commit.txt
    with open(exp_dir / "git_commit.txt", "w") as f:
        f.write(f"{commit_hash}\n")

    # 4. metrics.json
    metrics_payload = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "is_smoke": is_smoke,
        "non_research_smoke_test": is_smoke,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else "None",
        "amp": use_amp,
        "git_commit": commit_hash,
        "history": results["history"],
        "val_metrics": val_metrics,
        "best_epoch": results["best_epoch"],
        "best_score": results["best_score"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Phase 2 Target Verifications
    print("\n" + "=" * 70)
    print("Phase 2 Target Verification Checks")
    print("=" * 70)

    # Check 1: Loss decreases
    batch_losses = results["train_batch_losses"]
    if len(batch_losses) >= 10:
        first_quarter = np.mean(batch_losses[: len(batch_losses) // 4])
        last_quarter = np.mean(batch_losses[-len(batch_losses) // 4 :])
        loss_decreased = last_quarter < first_quarter
        print(f"Check 1: Training Loss Decrease")
        print(f"  First quarter avg loss: {first_quarter:.4f}")
        print(f"  Last quarter avg loss:  {last_quarter:.4f}")
        print(f"  Result: {'PASS (Loss decreased)' if loss_decreased else 'WARNING (Loss did not decrease)'}")
    else:
        loss_decreased = True
        print(f"Check 1: Training Loss: {batch_losses[0]:.4f} -> {batch_losses[-1]:.4f}")

    # Check 2: AUROC > ~0.5 (above random)
    auroc = val_metrics.get("auroc", float("nan"))
    above_random = not np.isnan(auroc) and auroc >= 0.50
    print(f"\nCheck 2: Performance Above Random")
    print(f"  Validation AUROC: {auroc:.4f}")
    print(f"  Result: {'PASS (AUROC >= 0.50)' if above_random else 'FAIL (AUROC < 0.50)'}")

    # Check 3: Checkpoint reload numerical equality check
    print(f"\nCheck 3: Checkpoint Reload Numerical Equality Check")
    test_tensor = torch.randn(4, 3, 96, 96).to(device)

    # Model original predictions
    model.eval()
    with torch.no_grad():
        out_original = model(test_tensor)

    # New model instance reload
    reloaded_model = build_resnet(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=False,  # Weights overwritten by checkpoint
    ).to(device)
    reloaded_model.eval()

    ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(ckpt_data["model_state_dict"])

    with torch.no_grad():
        out_reloaded = reloaded_model(test_tensor)

    max_diff = torch.max(torch.abs(out_original - out_reloaded)).item()
    is_identical = torch.allclose(out_original, out_reloaded, atol=1e-6)

    print(f"  Max absolute difference: {max_diff:.8e}")
    print(f"  Exact numerical equality (atol=1e-6): {is_identical}")
    print(f"  Result: {'PASS (Checkpoint reload verified)' if is_identical else 'FAIL (Checkpoint mismatch)'}")

    print("\n" + "=" * 70)
    print("Smoke Training Complete & All Verifications Evaluated!")
    print("=" * 70)


if __name__ == "__main__":
    main()
