#!/usr/bin/env python3
"""Authoritative Multi-Seed Pipeline for UTTA-Med on Camelyon17-WILDS.

Executes the frozen protocol:
1. Source training (Centers {0, 3, 4}) with early stopping on Center 1 (OOD val)
2. Checkpoint restoration from the true best OOD validation epoch
3. Source-only evaluation on Center 1 (val) and Center 2 (test)
4. MC-Dropout epistemic uncertainty analysis & threshold sweep on Center 1
5. Selection of UTTA threshold tau at 70th percentile of Center 1 predictive variance
6. Calibration of matched-coverage thresholds on Center 1 (Random, Confidence, Entropy)
7. Unlabeled target stream adaptation on Center 2 for:
   - Tent (lr=1e-5, bn_freeze_stats)
   - EATA (lr=1e-5, e_margin=0.4, bn_freeze_stats)
   - Random gate (matched coverage, seed=args.seed)
   - Confidence gate (matched coverage)
   - Entropy gate (matched coverage)
   - UTTA-Med (predictive variance < tau, lr=1e-5, mc_passes=20)
8. Held-out target evaluation on Center 2 with flip table analysis
9. Exports authoritative JSON: results/metrics/camelyon17_resnet18_utta_gated_s{seed}.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.camelyon17 import create_camelyon17_datasets
from src.data.paths import resolve_parquet_root
from src.data.transforms import get_camelyon17_eval_transform, get_camelyon17_train_transform
from src.evaluation.adaptation import flip_table
from src.evaluation.classification import binary_metrics, brier, ece
from src.models.resnet18 import build_resnet18, build_resnet
from src.training.trainer import SourceTrainer
from src.tta.gates import score_confidence, score_entropy, score_mc_dropout
from src.tta.uncertainty_gated import build_adapter
from src.utils.reproducibility import set_seed, seed_worker


def parse_args():
    parser = argparse.ArgumentParser(description="Authoritative Multi-Seed UTTA-Med Pipeline")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (42, 123, 2024, 7, 99)")
    parser.add_argument("--config", type=str, default="configs/FINAL/source_resnet18.yaml", help="Path to source configuration")
    parser.add_argument("--data-root", type=str, default=None, help="Parquet data directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Existing source checkpoint (skips training)")
    parser.add_argument("--skip-train", action="store_true", help="Skip source training if checkpoint exists")
    parser.add_argument("--epochs", type=int, default=20, help="Source training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and TTA")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--mc-passes", type=int, default=20, help="MC-Dropout passes")
    parser.add_argument("--lr", type=float, default=1e-5, help="TTA learning rate")
    parser.add_argument("--bn-mode", type=str, default="bn_freeze_stats", choices=["bn_freeze_stats", "bn_train_stats"])
    parser.add_argument("--tau", type=float, default=None, help="Explicit tau threshold (defaults to configs/tau_by_seed.yaml)")
    parser.add_argument("--percentile", type=float, default=70.0, help="UTTA gate threshold percentile on Center 1")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run with subsampled data")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device, dropout_p: float = 0.5) -> nn.Module:
    """Load model architecture and weights from checkpoint cleanly."""
    model = build_resnet18(pretrained=False, dropout_p=dropout_p, num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_stream(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate frozen model on a dataloader, returning (y_true, y_prob)."""
    model.eval()
    ys, probs = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].view(-1).cpu().numpy()
        p = torch.sigmoid(model(x)).view(-1).cpu().numpy()
        ys.append(y)
        probs.append(p)
    return np.concatenate(ys), np.concatenate(probs)


def run_mc_dropout_on_val(
    model: nn.Module, loader: DataLoader, device: torch.device, n_passes: int = 20
) -> Dict[str, np.ndarray]:
    """Compute MC-dropout predictions, predictive variance, entropy, and confidence on validation set."""
    ys, mean_probs, variances, entropies, confidences = [], [], [], [], []
    
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1].view(-1).cpu().numpy()
        
        # MC-Dropout
        p_mean, unc = score_mc_dropout(model, x, n_passes=n_passes)
        # Softmax / sigmoid confidence
        conf = score_confidence(model, x)
        # Entropy
        ent = -score_entropy(model, x)
        
        ys.append(y)
        mean_probs.append(p_mean.cpu().numpy())
        variances.append(unc.cpu().numpy())
        entropies.append(ent.cpu().numpy())
        confidences.append(conf.cpu().numpy())
        
    return {
        "y": np.concatenate(ys),
        "prob": np.concatenate(mean_probs),
        "variance": np.concatenate(variances),
        "entropy": np.concatenate(entropies),
        "confidence": np.concatenate(confidences),
    }


def compute_uncertainty_analysis(val_stats: Dict[str, np.ndarray], percentile: float = 70.0) -> Dict[str, Any]:
    """Analyze uncertainty/error correlation, uncertainty bins, and threshold sweep on Center 1."""
    y = val_stats["y"]
    prob = val_stats["prob"]
    var = val_stats["variance"]
    ent = val_stats["entropy"]
    conf = val_stats["confidence"]
    
    pred = (prob >= 0.5).astype(int)
    error = (pred != y).astype(int)
    
    # 1. Uncertainty / Error correlation
    # Pearson / Spearman correlation between predictive variance and binary error
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(var, error)
    spearman_r, spearman_p = spearmanr(var, error)
    
    # AUROC of variance as an error detector
    from sklearn.metrics import roc_auc_score
    try:
        error_detection_auroc = float(roc_auc_score(error, var))
    except Exception:
        error_detection_auroc = 0.5
        
    correlation_metrics = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "error_detection_auroc": error_detection_auroc,
    }
    
    # 2. Uncertainty Bins (5 quantiles)
    quantiles = np.quantile(var, np.linspace(0, 1, 6))
    uncertainty_bins = []
    for i in range(len(quantiles) - 1):
        lo, hi = quantiles[i], quantiles[i + 1]
        mask = (var >= lo) & (var <= hi) if i == len(quantiles) - 2 else (var >= lo) & (var < hi)
        if mask.sum() > 0:
            bin_entry = {
                "bin_index": i + 1,
                "range": [float(lo), float(hi)],
                "n_samples": int(mask.sum()),
                "mean_variance": float(var[mask].mean()),
                "mean_confidence": float(conf[mask].mean()),
                "error_rate": float(error[mask].mean()),
                "accuracy": float(1.0 - error[mask].mean()),
            }
            uncertainty_bins.append(bin_entry)
            
    # 3. UTTA Threshold sweep on Center 1 (percentiles 10 to 90)
    sweep_percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    threshold_sweep = []
    for pct in sweep_percentiles:
        tau_val = float(np.percentile(var, pct))
        accepted = var < tau_val
        cov = float(accepted.mean())
        acc = float((pred[accepted] == y[accepted]).mean()) if accepted.sum() > 0 else 0.0
        threshold_sweep.append({
            "percentile": pct,
            "threshold": tau_val,
            "coverage": cov,
            "accepted_accuracy": acc,
        })
        
    # Selected threshold at requested percentile (70th)
    selected_threshold = float(np.percentile(var, percentile))
    val_coverage = float((var < selected_threshold).mean())
    
    # Matched thresholds for Confidence and Entropy to achieve identical validation coverage
    # Confidence: accept if conf > tau_conf  <=>  tau_conf is (1 - val_coverage) percentile of conf
    tau_confidence = float(np.percentile(conf, (1.0 - val_coverage) * 100))
    # Entropy: accept if ent < tau_ent      <=>  tau_ent is val_coverage percentile of ent
    tau_entropy = float(np.percentile(ent, val_coverage * 100))
    # Random: accept probability = val_coverage
    tau_random = val_coverage
    
    return {
        "correlation": correlation_metrics,
        "bins": uncertainty_bins,
        "sweep": threshold_sweep,
        "selected_threshold": selected_threshold,
        "validation_coverage": val_coverage,
        "matched_thresholds": {
            "random": tau_random,
            "confidence": tau_confidence,
            "entropy": tau_entropy,
        }
    }


def adapt_and_evaluate_target(
    checkpoint_path: str | Path,
    method: str,
    test_loader: DataLoader,
    device: torch.device,
    y_test_true: np.ndarray,
    p_source_test: np.ndarray,
    lr: float = 1e-5,
    steps: int = 1,
    tau: float = None,
    e_margin: float = 0.4,
    n_passes: int = 20,
    seed: int = 42,
    bn_mode: str = "bn_freeze_stats",
    dropout_p: float = 0.5,
) -> Dict[str, Any]:
    """Execute continual test-time adaptation on Center 2 stream and evaluate with flip table."""
    # Always load fresh copy of source checkpoint for independent adaptation run
    model = load_model_from_checkpoint(checkpoint_path, device=device, dropout_p=dropout_p)
    
    adapter = build_adapter(
        model,
        method=method,
        lr=lr,
        steps=steps,
        tau=tau,
        e_margin=e_margin,
        n_passes=n_passes,
        seed=seed,
        bn_mode=bn_mode,
    )
    
    p_after_list = []
    # TARGET ADAPTATION LOOP: Stream images x ONLY (zero label leakage)
    for batch in test_loader:
        x = batch[0].to(device)
        # Note: batch[1] is NOT passed to adapt_batch!
        pa = adapter.adapt_batch(x).detach().cpu().numpy()
        p_after_list.append(pa)
        
    p_after = np.concatenate(p_after_list)
    
    # Post-adaptation evaluation using held-out labels
    metrics = binary_metrics(y_test_true, p_after)
    metrics["ece"] = ece(y_test_true, p_after)
    metrics["brier"] = brier(y_test_true, p_after)
    
    harm = flip_table(y_test_true, p_source_test, p_after)
    
    if method == "tent":
        coverage = 1.0
    elif hasattr(adapter, "coverage"):
        coverage = float(adapter.coverage)
    else:
        coverage = "N/A"
        
    return {
        "method": method,
        "coverage": coverage,
        "tau": tau,
        "n_adapted": int(getattr(adapter, "n_adapted", len(y_test_true))),
        "n_seen": int(getattr(adapter, "n_seen", len(y_test_true))),
        "metrics": metrics,
        "harm": harm,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    
    print("=" * 80)
    print(f"UTTA-MED AUTHORITATIVE PIPELINE — SEED {args.seed}")
    print(f"Device: {device} | BN Mode: {args.bn_mode} | LR: {args.lr} | Passes: {args.mc_passes}")
    print("=" * 80)
    
    data_root = args.data_root or resolve_parquet_root()
    print(f"Parquet data root: {data_root}")
    
    train_transform = get_camelyon17_train_transform()
    eval_transform = get_camelyon17_eval_transform()
    
    source_train_ds, id_val_ds, ood_val_ds, ood_test_ds = create_camelyon17_datasets(
        root=data_root,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    
    if args.smoke:
        print("\n[SMOKE MODE ENABLED] Subsampling datasets for rapid verification...")
        source_train_ds = Subset(source_train_ds, list(range(256)))
        ood_val_ds = Subset(ood_val_ds, list(range(256)))
        ood_test_ds = Subset(ood_test_ds, list(range(256)))
        args.epochs = 1
        
    print(f"Dataset sizes:")
    print(f"  Source Train (Centers 0,3,4) : {len(source_train_ds):,}")
    print(f"  OOD Val      (Center 1)       : {len(ood_val_ds):,}")
    print(f"  Target Test  (Center 2)       : {len(ood_test_ds):,}")
    
    # Load authoritative config
    with open(args.config, "r") as f:
        auth_cfg = yaml.safe_load(f)
    dropout_p = auth_cfg.get("model", {}).get("dropout_p", 0.5)
    
    # Determine checkpoint path
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (checkpoint_dir / f"camelyon17_resnet18_source_s{args.seed}.pt")
    
    # -------------------------------------------------------------
    # Step 1: Source Training (if checkpoint not present or forced)
    # -------------------------------------------------------------
    if not Path(checkpoint_path).exists() and not args.skip_train:
        print(f"\n--- STEP 1: TRAINING SOURCE MODEL (SEED {args.seed}) ---")
        g = torch.Generator()
        g.manual_seed(args.seed)
        
        train_loader = DataLoader(
            source_train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            ood_val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        
        source_model = build_resnet18(num_classes=1, dropout_p=0.5, pretrained=True).to(device)
        optimizer = torch.optim.Adam(source_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        trainer = SourceTrainer(
            model=source_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_dir=checkpoint_dir,
            patience=5,
            metric_for_best="auroc",
            use_amp=(device.type == "cuda"),
        )
        
        train_results = trainer.train(epochs=args.epochs)
        trainer.save_checkpoint(
            filepath=checkpoint_path,
            epoch=train_results["best_epoch"],
            metrics={"auroc": train_results["best_score"]},
            extra={"seed": args.seed, "config": auth_cfg},
        )
        print(f"Source training completed! Best epoch: {train_results['best_epoch']} | Best Val AUROC: {train_results['best_score']:.4f}")
    else:
        print(f"\nUsing source checkpoint: {checkpoint_path}")
        
    # -------------------------------------------------------------
    # Step 2: Load Best Checkpoint & Setup Evaluation DataLoaders
    # -------------------------------------------------------------
    print(f"\n--- STEP 2: RESTORING BEST CHECKPOINT (Center-1 Peak AUROC) ---")
    best_model = load_model_from_checkpoint(checkpoint_path, device=device, dropout_p=dropout_p)
    
    val_loader = DataLoader(
        ood_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ood_test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # -------------------------------------------------------------
    # Step 3: Source-Only Evaluation (Zero Adaptation)
    # -------------------------------------------------------------
    print("\n--- STEP 3: SOURCE-ONLY EVALUATION ---")
    y_val, p_val_source = evaluate_stream(best_model, val_loader, device=device)
    val_source_metrics = binary_metrics(y_val, p_val_source)
    val_source_metrics["ece"] = ece(y_val, p_val_source)
    val_source_metrics["brier"] = brier(y_val, p_val_source)
    print(f"Source OOD Val (Center 1)  -> AUROC: {val_source_metrics['auroc']:.4f} | F1: {val_source_metrics['f1']:.4f} | ECE: {val_source_metrics['ece']:.4f}")
    
    y_test, p_test_source = evaluate_stream(best_model, test_loader, device=device)
    test_source_metrics = binary_metrics(y_test, p_test_source)
    test_source_metrics["ece"] = ece(y_test, p_test_source)
    test_source_metrics["brier"] = brier(y_test, p_test_source)
    print(f"Source Target Test (Center 2) -> AUROC: {test_source_metrics['auroc']:.4f} | F1: {test_source_metrics['f1']:.4f} | ECE: {test_source_metrics['ece']:.4f}")
    
    # -------------------------------------------------------------
    # Step 4: Uncertainty Analysis & Threshold Selection (Center 1 ONLY)
    # -------------------------------------------------------------
    print(f"\n--- STEP 4: MC-DROPOUT UNCERTAINTY ANALYSIS ON CENTER 1 ({args.mc_passes} Passes) ---")
    val_mc_stats = run_mc_dropout_on_val(best_model, val_loader, device=device, n_passes=args.mc_passes)
    uncertainty_results = compute_uncertainty_analysis(val_mc_stats, percentile=args.percentile)
    
    # Check authoritative tau from configs/tau_by_seed.yaml
    tau_cfg_p = ROOT / "configs" / "tau_by_seed.yaml"
    auth_tau = None
    if tau_cfg_p.exists():
        with open(tau_cfg_p) as tf:
            tau_map = yaml.safe_load(tf).get("tau", {})
            if args.seed in tau_map:
                auth_tau = float(tau_map[args.seed])
    
    if args.tau is not None:
        tau_selected = args.tau
        print(f"Using explicitly specified tau                : {tau_selected:.6e}")
    elif auth_tau is not None:
        tau_selected = auth_tau
        print(f"Using authoritative tau from tau_by_seed.yaml : {tau_selected:.6e}")
    else:
        tau_selected = uncertainty_results["selected_threshold"]
        print(f"Selected UTTA tau ({args.percentile}th percentile) : {tau_selected:.6e}")

    val_coverage = uncertainty_results["validation_coverage"]
    matched = uncertainty_results["matched_thresholds"]
    
    print(f"Validation Coverage at tau                    : {val_coverage * 100:.2f}%")
    print(f"Matched Thresholds on Center 1:")
    print(f"  Random coverage    : {matched['random'] * 100:.2f}%")
    print(f"  Confidence tau     : {matched['confidence']:.6f}")
    print(f"  Entropy tau        : {matched['entropy']:.6f}")
    print(f"Uncertainty / Error Correlation (AUROC)       : {uncertainty_results['correlation']['error_detection_auroc']:.4f}")
    
    # -------------------------------------------------------------
    # Step 5: Continual Test-Time Adaptation on Center 2
    # -------------------------------------------------------------
    print(f"\n--- STEP 5: TEST-TIME ADAPTATION ON TARGET (Center 2) ---")
    
    methods_to_run = [
        ("tent", {"lr": args.lr, "bn_mode": args.bn_mode}),
        ("eata", {"lr": args.lr, "e_margin": 0.4, "bn_mode": args.bn_mode}),
        ("random", {"lr": args.lr, "tau": matched["random"], "seed": args.seed, "bn_mode": args.bn_mode}),
        ("confidence", {"lr": args.lr, "tau": matched["confidence"], "bn_mode": args.bn_mode}),
        ("entropy", {"lr": args.lr, "tau": matched["entropy"], "bn_mode": args.bn_mode}),
        ("utta", {"lr": args.lr, "tau": tau_selected, "n_passes": args.mc_passes, "bn_mode": args.bn_mode}),
    ]
    
    target_results = {}
    for method_name, method_kwargs in methods_to_run:
        print(f"\nAdapting with method: [{method_name.upper()}]...")
        res = adapt_and_evaluate_target(
            checkpoint_path=checkpoint_path,
            method=method_name,
            test_loader=test_loader,
            device=device,
            y_test_true=y_test,
            p_source_test=p_test_source,
            dropout_p=dropout_p,
            **method_kwargs,
        )
        target_results[method_name] = res
        m = res["metrics"]
        h = res["harm"]
        cov_str = f"{res['coverage']*100:.1f}%" if isinstance(res['coverage'], float) else str(res['coverage'])
        print(f"  -> AUROC: {m['auroc']:.4f} | F1: {m['f1']:.4f} | ECE: {m['ece']:.4f} | Harm: {h['harm_rate']*100:.2f}% | Corr: {h['correction_rate']*100:.2f}% | Cov: {cov_str}")
        
    # -------------------------------------------------------------
    # Step 6: Compile Authoritative JSON
    # -------------------------------------------------------------
    print("\n--- STEP 6: EXPORTING AUTHORITATIVE SEED JSON ---")
    output_dir = ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = output_dir / f"camelyon17_resnet18_utta_gated_s{args.seed}.json"
    
    final_payload = {
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "config": auth_cfg,
        "source_checkpoint": str(checkpoint_path),
        "source_validation": val_source_metrics,
        "source_target_test": test_source_metrics,
        "uncertainty_error_correlation": uncertainty_results["correlation"],
        "uncertainty_bins": uncertainty_results["bins"],
        "utta_threshold_sweep": uncertainty_results["sweep"],
        "selected_threshold": tau_selected,
        "validation_coverage": val_coverage,
        "matched_thresholds": matched,
        "tent_test": target_results["tent"],
        "eata_test": target_results["eata"],
        "random_test": target_results["random"],
        "confidence_test": target_results["confidence"],
        "entropy_test": target_results["entropy"],
        "utta_test": target_results["utta"],
    }
    
    with open(output_json_path, "w") as f:
        json.dump(final_payload, f, indent=2)
        
    print(f"\nSuccessfully wrote authoritative results to:\n  {output_json_path}")
    print("\n" + "=" * 80)
    print(f"SEED {args.seed} PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
