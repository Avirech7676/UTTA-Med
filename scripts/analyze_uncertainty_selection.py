#!/usr/bin/env python3
"""Stage 5 & 6: Adaptive Uncertainty Selection & Ablation Analysis.

Evaluates whether MC-Dropout, Rotation Consistency, and Source-Centroid Distance
contain orthogonal information beyond single-pass softmax confidence.
Evaluated strictly on Camelyon17-WILDS Center 1 validation patches.
"""

from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.resnet18 import build_resnet18

def load_center1_val_patches(seed: int = 42):
    parquet_path = Path("c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data/validation-00002-of-00003.parquet")
    pf = pq.ParquetFile(parquet_path)
    
    # Row group 0 is class 0, row group 115 is class 1 (100 rows each)
    tbl_0 = pf.read_row_group(0, columns=["image", "label"])
    tbl_1 = pf.read_row_group(115, columns=["image", "label"])
    
    combined = []
    for tbl in [tbl_0, tbl_1]:
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            lbl = int(row["label"])
            b = row["image"]["bytes"]
            im = Image.open(io.BytesIO(b)).convert("RGB")
            arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            combined.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl))
            
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    
    imgs = [item[0] for item in combined]
    labels = [item[1] for item in combined]
    return torch.stack(imgs, dim=0), np.array(labels)

def main():
    device = torch.device("cpu")
    print("[*] Loading Center 1 validation cohort...")
    x, y_true = load_center1_val_patches(seed=42)
    print(f"[*] Loaded {len(x)} validation patches (Class 0: {(y_true == 0).sum()}, Class 1: {(y_true == 1).sum()})")

    # Load checkpoint
    ckpt_path = Path("checkpoints/camelyon17_resnet18_source_s42_BEST.pt")
    model = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    # Load 512-D source centroids
    centroid_path = Path("checkpoints/source_centroids_resnet18_s42.pt")
    c_data = torch.load(centroid_path, map_location=device, weights_only=False)
    centroids = c_data["prototypes"].to(device)
    print(f"[*] Loaded centroids: {centroids.shape}")

    # Compute metrics batch by batch
    batch_size = 64
    n_batches = int(np.ceil(len(x) / batch_size))

    all_p = []
    all_conf = []
    all_u_mc = []
    all_u_cons = []
    all_u_feat = []

    print("[*] Extracting multi-signal uncertainty profiles...")
    for b_idx in range(n_batches):
        xb = x[b_idx * batch_size : (b_idx + 1) * batch_size].to(device)
        
        # 1. Deterministic forward pass
        with torch.no_grad():
            out, feat = model.forward_with_features(xb)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)
            all_p.append(p.numpy())
            all_conf.append(conf.numpy())

            # 2. MC Dropout variance (10 passes)
            model.classifier[2].train()  # Enable dropout layer
            mc_probs = []
            for _ in range(10):
                mc_out = model(xb)
                mc_probs.append(torch.sigmoid(mc_out).view(-1))
            model.eval()
            stacked_mc = torch.stack(mc_probs, dim=0)
            u_mc = stacked_mc.var(dim=0, unbiased=False)
            all_u_mc.append(u_mc.numpy())

            # 3. Rotation consistency variance (4 rotations)
            rot_probs = []
            for k in range(4):
                x_rot = torch.rot90(xb, k, dims=[2, 3])
                rot_probs.append(torch.sigmoid(model(x_rot)).view(-1))
            stacked_rot = torch.stack(rot_probs, dim=0)
            u_cons = stacked_rot.var(dim=0, unbiased=False)
            all_u_cons.append(u_cons.numpy())

            # 4. Source centroid distance
            if feat.dim() > 2:
                feat = feat.mean(dim=[-2, -1])
            feat_norm = F.normalize(feat, p=2, dim=1)
            cent_norm = F.normalize(centroids, p=2, dim=1)
            sim = torch.mm(feat_norm, cent_norm.t()).max(dim=1)[0]
            dist = torch.clamp(1.0 - sim, min=0.0, max=2.0)
            all_u_feat.append(dist.numpy())

    p = np.concatenate(all_p)
    conf = np.concatenate(all_conf)
    u_mc = np.concatenate(all_u_mc)
    u_cons = np.concatenate(all_u_cons)
    u_feat = np.concatenate(all_u_feat)

    # Predictions & Errors
    preds_bin = (p >= 0.5).astype(int)
    is_correct = (preds_bin == y_true).astype(int)
    is_error = 1 - is_correct

    # Harm proxy: unsupervised update on sample where label is wrong degrades representation
    print("\n" + "=" * 70)
    print("UNCERTAINTY SIGNAL ANALYSIS (Camelyon17 Center 1 Validation Cohort)")
    print("=" * 70)
    print(f"Overall Accuracy:  {is_correct.mean():.4f}")
    print(f"Overall AUROC:     {roc_auc_score(y_true, p):.4f}")
    print(f"Overall AUPRC:     {average_precision_score(y_true, p):.4f}")

    # Correlation Matrix
    signals = {
        "1 - Confidence": 1.0 - conf,
        "MC-Dropout Var": u_mc,
        "Consistency Var": u_cons,
        "Centroid Dist": u_feat,
    }
    sig_df = pd.DataFrame(signals)
    corr_spearman = sig_df.corr(method="spearman")
    print("\n--- Spearman Rank Correlation Matrix between Uncertainty Signals ---")
    print(corr_spearman.round(4).to_string())

    # Error Discrimination AUROC (Ability of each signal to predict prediction error)
    print("\n--- Ability of Uncertainty Signals to Identify Misclassified Patches (AUROC) ---")
    for name, sig in signals.items():
        auc_err = roc_auc_score(is_error, sig)
        print(f"  {name:<20}: Error AUROC = {auc_err:.4f}")

    # Matched-Coverage Selection Comparison
    # Target coverage levels: 50%, 60%, 70%, 80%
    coverages = [0.50, 0.60, 0.70, 0.80]
    print("\n--- Matched Coverage Performance Comparison ---")
    print(f"{'Coverage':<10} | {'Policy':<25} | {'Selected Acc':<14} | {'Harm Rate (Errors in Selected)':<30}")
    print("-" * 85)

    # Normalize signals to [0, 1] for combination
    u_mc_norm = (u_mc - u_mc.min()) / (u_mc.max() - u_mc.min() + 1e-8)
    u_cons_norm = (u_cons - u_cons.min()) / (u_cons.max() - u_cons.min() + 1e-8)
    u_feat_norm = (u_feat - u_feat.min()) / (u_feat.max() - u_feat.min() + 1e-8)

    # Combined uncertainty: 0.5 * MC + 0.3 * Cons + 0.2 * Feat
    u_comb = 0.5 * u_mc_norm + 0.3 * u_cons_norm + 0.2 * u_feat_norm

    policies = {
        "Confidence Gating": conf,                  # Higher is more certain
        "MC Dropout Only": -u_mc,                   # Lower unc is more certain
        "Consistency Only": -u_cons,
        "Centroid Dist Only": -u_feat,
        "Multi-Signal Combined": -u_comb,
    }

    records = []
    for cov in coverages:
        n_select = int(len(p) * cov)
        for pol_name, score in policies.items():
            # Select top n_select patches with highest certainty
            thresh = np.partition(score, len(score) - n_select)[len(score) - n_select]
            mask = score >= thresh
            if mask.sum() != n_select:
                # Handle ties cleanly
                order = np.argsort(-score)
                mask = np.zeros(len(score), dtype=bool)
                mask[order[:n_select]] = True

            sel_acc = is_correct[mask].mean()
            sel_harm = is_error[mask].mean()
            records.append({
                "coverage": cov,
                "policy": pol_name,
                "selected_accuracy": round(float(sel_acc), 4),
                "error_rate_in_selected": round(float(sel_harm), 4),
            })
            print(f"{cov*100:4.0f}%     | {pol_name:<25} | {sel_acc:12.4f}   | {sel_harm:28.4f}")
        print("-" * 85)

    # Save results to json
    out_path = Path("results/uncertainty_signal_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "spearman_correlation": corr_spearman.to_dict(),
            "error_discrimination_auroc": {name: round(float(roc_auc_score(is_error, sig)), 4) for name, sig in signals.items()},
            "matched_coverage_comparison": records,
        }, f, indent=2)
    print(f"\n[+] Results saved to {out_path}")

if __name__ == "__main__":
    main()
