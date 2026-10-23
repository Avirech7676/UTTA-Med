#!/usr/bin/env python3
"""Phase 3 & 4: Systematic Tuning of Confidence + Feature-Distance Gating on Center 1 Validation Split.

Rules:
1. Strict isolation: Tuning is conducted exclusively on Camelyon17-WILDS Center 1 validation patches.
2. Center 2 target-test cohort is untouched during parameter selection.
3. Evaluates error detection AUROC, false admission rate, and post-adaptation F1 across:
   - Weight combinations (w_conf, w_feat)
   - Joint admission thresholds (tau_joint)
   - Adaptation mechanisms: gradient clipping, source anchoring, confidence-weighted entropy,
     update frequency, and class balancing.
4. Identifies the optimal, defensible B7 configuration.
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.models.resnet18 import build_resnet18
from src.v2.gating.confidence_feature_gating import (
    ConfidenceFeatureGatingPolicy,
    ConfFeatureDecision,
)

def load_center1_validation_data(n_samples: int = 400, seed: int = 42):
    parquet_path = Path("c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data/validation-00002-of-00003.parquet")
    pf = pq.ParquetFile(parquet_path)
    
    # Read balanced slice from validation row groups
    tbl_0 = pf.read_row_group(0, columns=["image", "label", "slide", "patient"])
    tbl_1 = pf.read_row_group(115, columns=["image", "label", "slide", "patient"])
    
    combined = []
    for tbl in [tbl_0, tbl_1]:
        df = tbl.to_pandas()
        for _, row in df.iterrows():
            lbl = int(row["label"])
            b = row["image"]["bytes"]
            im = Image.open(io.BytesIO(b)).convert("RGB")
            arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            combined.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl, int(row["slide"]), int(row["patient"])))
            if len(combined) >= n_samples:
                break
        if len(combined) >= n_samples:
            break
            
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    imgs = torch.stack([item[0] for item in combined], dim=0)
    labels = np.array([item[1] for item in combined])
    slides = np.array([item[2] for item in combined])
    patients = np.array([item[3] for item in combined])
    return imgs, labels, slides, patients

def main():
    print("=" * 80)
    print("PHASE 3: TUNING CONFIDENCE + FEATURE-DISTANCE GATING (CENTER 1 DEV SPLIT)")
    print("=" * 80)

    device = torch.device("cpu")
    x_val, y_val, slides_val, patients_val = load_center1_validation_data(n_samples=400, seed=42)
    print(f"[*] Loaded {len(x_val)} Center 1 validation patches (Tumor: {(y_val == 1).sum()}, Normal: {(y_val == 0).sum()})")

    # Load verified ResNet-18 model (Seed 42)
    ckpt_path = ROOT / "checkpoints" / "camelyon17_resnet18_source_s42_BEST.pt"
    model = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()

    # Load 512-D centroids
    centroids_path = ROOT / "checkpoints" / "source_centroids_resnet18_s42.pt"
    centroids_data = torch.load(centroids_path, map_location=device, weights_only=False)
    centroids = centroids_data["centroids"] if "centroids" in centroids_data else centroids_data["prototypes"]
    print(f"[*] Centroids shape: {centroids.shape}")

    # Baseline predictions
    with torch.no_grad():
        logits_val = model(x_val).view(-1)
        p_val = torch.sigmoid(logits_val).numpy()
        pred_val = (p_val >= 0.5).astype(int)

    errors = (pred_val != y_val).astype(int)
    baseline_f1 = f1_score(y_val, pred_val, zero_division=0)
    baseline_auc = roc_auc_score(y_val, p_val)
    print(f"[*] Frozen Source Baseline on Val Slice: AUROC={baseline_auc:.4f}, Binary F1={baseline_f1:.4f}, Total Errors={errors.sum()}/{len(y_val)}")

    # Extract features for all validation patches
    with torch.no_grad():
        feat_val = model.forward_features(x_val).detach()

    # Precompute individual signals
    conf_val = np.maximum(p_val, 1.0 - p_val)
    s_conf = (conf_val - 0.5) * 2.0  # [0, 1]
    u_conf = 1.0 - s_conf           # [0, 1]

    feat_norm = F.normalize(feat_val, p=2, dim=-1)
    cent_norm = F.normalize(centroids, p=2, dim=-1)
    pred_cls = torch.from_numpy(pred_val).long()
    selected_centroids = cent_norm[pred_cls]
    cos_sim = torch.sum(feat_norm * selected_centroids, dim=-1).clamp(-1.0, 1.0).numpy()
    d_feat = 1.0 - cos_sim
    s_feat = np.clip(1.0 - (d_feat / 0.80), 0.0, 1.0)
    u_feat = 1.0 - s_feat

    # 1. Error Detection AUROC
    auc_err_conf = roc_auc_score(errors, u_conf)
    auc_err_feat = roc_auc_score(errors, u_feat)
    print(f"[*] Error Detection AUROC (Confidence Uncertainty): {auc_err_conf:.4f}")
    print(f"[*] Error Detection AUROC (Feature Distance Uncertainty): {auc_err_feat:.4f}")

    # 2. Sweep Weight Combinations (w_conf, w_feat) on Error Detection AUROC
    print("\n--- SWEEPING WEIGHT COMBINATIONS (w_conf, w_feat) ---")
    weight_records = []
    w_grid = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7), (0.2, 0.8), (0.0, 1.0)]

    for wc, wf in w_grid:
        u_comb = wc * u_conf + wf * u_feat
        s_comb = wc * s_conf + wf * s_feat
        auc_err = roc_auc_score(errors, u_comb)

        # Matched coverage at 60% admission
        tau_60 = np.quantile(s_comb, 0.40) # Top 60% admitted
        mask_60 = s_comb >= tau_60
        admitted_errors = errors[mask_60].sum()
        admitted_err_rate = admitted_errors / max(mask_60.sum(), 1)

        # Matched coverage at 80% admission
        tau_80 = np.quantile(s_comb, 0.20) # Top 80% admitted
        mask_80 = s_comb >= tau_80
        admitted_errors_80 = errors[mask_80].sum()
        admitted_err_rate_80 = admitted_errors_80 / max(mask_80.sum(), 1)

        rec = {
            "w_conf": wc,
            "w_feat": wf,
            "auc_error_detection": round(float(auc_err), 4),
            "err_rate_at_60_cov": round(float(admitted_err_rate), 4),
            "admitted_errors_at_60_cov": int(admitted_errors),
            "err_rate_at_80_cov": round(float(admitted_err_rate_80), 4),
            "admitted_errors_at_80_cov": int(admitted_errors_80),
        }
        weight_records.append(rec)
        print(f"  w_conf={wc:.1f}, w_feat={wf:.1f} | Error AUROC: {auc_err:.4f} | "
              f"Err Rate @ 60% Cov: {admitted_err_rate:.4f} ({admitted_errors} errs) | "
              f"Err Rate @ 80% Cov: {admitted_err_rate_80:.4f} ({admitted_errors_80} errs)")

    df_weights = pd.DataFrame(weight_records)
    df_weights.to_csv(ROOT / "FEATURE_DISTANCE_ABLATIONS.csv", index=False)
    print(f"\n[+] Saved FEATURE_DISTANCE_ABLATIONS.csv")

    best_weight_row = df_weights.sort_values(by=["auc_error_detection", "err_rate_at_80_cov"], ascending=[False, True]).iloc[0]
    best_wc = float(best_weight_row["w_conf"])
    best_wf = float(best_weight_row["w_feat"])
    print(f"[+] Optimal Validation Weights: w_conf={best_wc:.2f}, w_feat={best_wf:.2f} (Error AUROC={best_weight_row['auc_error_detection']})")

    # =========================================================================
    # PHASE 4: IMPROVE ADAPTATION QUALITY (Hyperparameter Sweep on Validation)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 4: HYPERPARAMETER & REGULARIZATION OPTIMIZATION ON VALIDATION COHORT")
    print("=" * 80)

    # Investigate:
    # 1. Update frequency: every batch vs every 2 batches
    # 2. Source-parameter regularization (lambda_anchor)
    # 3. Gradient clipping (clip_norm)
    # 4. Confidence-weighted entropy minimization
    # 5. Class-balanced sample admission

    adaptation_configs = [
        {"name": "standard_tent", "lr": 1e-4, "lambda_anchor": 0.0, "clip_norm": None, "weighted_ent": False, "balanced": False},
        {"name": "clip_norm_0.5", "lr": 1e-4, "lambda_anchor": 0.0, "clip_norm": 0.5, "weighted_ent": False, "balanced": False},
        {"name": "clip_norm_1.0", "lr": 1e-4, "lambda_anchor": 0.0, "clip_norm": 1.0, "weighted_ent": False, "balanced": False},
        {"name": "anchor_0.01", "lr": 1e-4, "lambda_anchor": 0.01, "clip_norm": 1.0, "weighted_ent": False, "balanced": False},
        {"name": "anchor_0.05", "lr": 1e-4, "lambda_anchor": 0.05, "clip_norm": 1.0, "weighted_ent": False, "balanced": False},
        {"name": "weighted_entropy", "lr": 1e-4, "lambda_anchor": 0.01, "clip_norm": 1.0, "weighted_ent": True, "balanced": False},
        {"name": "class_balanced", "lr": 1e-4, "lambda_anchor": 0.01, "clip_norm": 1.0, "weighted_ent": True, "balanced": True},
        {"name": "frequency_step2", "lr": 1e-4, "lambda_anchor": 0.01, "clip_norm": 1.0, "weighted_ent": True, "balanced": False, "step_freq": 2},
    ]

    opt_results = []
    tau_chosen = 0.80

    for cfg in adaptation_configs:
        # Re-initialize model from fresh checkpoint
        m_adapt = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
        m_adapt.load_state_dict(state_dict)
        
        # Configure BatchNorm affine params
        for m in m_adapt.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(True)
                m.bias.requires_grad_(True)
            else:
                m.requires_grad_(False)

        adapt_params = [p for p in m_adapt.parameters() if p.requires_grad]
        source_params = [p.clone().detach() for p in adapt_params]
        opt = torch.optim.Adam(adapt_params, lr=cfg["lr"])

        # Adaptation policy: Confidence + Feature distance with optimal validation weights
        policy = ConfidenceFeatureGatingPolicy(w_conf=best_wc, w_feat=best_wf, tau_joint=tau_chosen, source_centroids=centroids)
        
        # Simulate batch-streaming adaptation in batches of 32
        bs = 32
        n_batches = len(x_val) // bs
        all_post_p = []

        step_counter = 0
        step_freq = cfg.get("step_freq", 1)

        for b_idx in range(n_batches):
            xb = x_val[b_idx*bs:(b_idx+1)*bs]
            dec = policy.select(m_adapt, xb)

            mask = dec.keep_mask
            if cfg.get("balanced", False) and mask.sum() > 1:
                # Balance classes by matching counts
                p_tmp = torch.sigmoid(m_adapt(xb)).view(-1).detach()
                pos_idx = torch.nonzero(mask & (p_tmp >= 0.5), as_tuple=True)[0]
                neg_idx = torch.nonzero(mask & (p_tmp < 0.5), as_tuple=True)[0]
                min_c = min(len(pos_idx), len(neg_idx))
                if min_c > 0:
                    keep_idx = torch.cat([pos_idx[:min_c], neg_idx[:min_c]])
                    mask = torch.zeros_like(mask)
                    mask[keep_idx] = True

            if mask.sum() > 0:
                m_adapt.train()
                for m in m_adapt.modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.Dropout)):
                        m.eval()

                out_b = m_adapt(xb[mask])
                p_b = torch.sigmoid(out_b).clamp(1e-6, 1.0 - 1e-6)
                ent_loss = -(p_b * torch.log(p_b) + (1.0 - p_b) * torch.log(1.0 - p_b))

                if cfg.get("weighted_ent", False):
                    weights_b = dec.joint_scores[mask]
                    ent_loss = (ent_loss * weights_b).sum() / (weights_b.sum() + 1e-6)
                else:
                    ent_loss = ent_loss.mean()

                loss = ent_loss

                # Source-parameter regularization anchor
                if cfg.get("lambda_anchor", 0.0) > 0.0:
                    anchor_loss = sum(torch.sum((p - p0) ** 2) for p, p0 in zip(adapt_params, source_params))
                    loss = loss + cfg["lambda_anchor"] * anchor_loss

                loss.backward()

                step_counter += 1
                if step_counter % step_freq == 0:
                    if cfg.get("clip_norm") is not None:
                        torch.nn.utils.clip_grad_norm_(adapt_params, cfg["clip_norm"])
                    opt.step()
                    opt.zero_grad()

            with torch.no_grad():
                m_adapt.eval()
                post_p = torch.sigmoid(m_adapt(xb)).view(-1).numpy()
                all_post_p.append(post_p)

        all_post_p = np.concatenate(all_post_p)
        eval_labels = y_val[:len(all_post_p)]
        post_preds = (all_post_p >= 0.5).astype(int)

        post_f1 = f1_score(eval_labels, post_preds, zero_division=0)
        post_auc = roc_auc_score(eval_labels, all_post_p)
        delta_f1 = post_f1 - baseline_f1
        harm_rate = np.mean((eval_labels == pred_val[:len(all_post_p)]) & (eval_labels != post_preds))

        opt_rec = {
            "config_name": cfg["name"],
            "post_auc": round(float(post_auc), 4),
            "post_f1": round(float(post_f1), 4),
            "delta_f1": round(float(delta_f1), 4),
            "harm_rate": round(float(harm_rate), 4),
        }
        opt_results.append(opt_rec)
        print(f"  [{cfg['name']:<20}] Post F1: {post_f1:.4f} (dF1={delta_f1:+.4f}) | AUROC: {post_auc:.4f} | Harm: {harm_rate:.4f}", flush=True)

    df_opt = pd.DataFrame(opt_results)
    best_opt = df_opt.sort_values(by="post_f1", ascending=False).iloc[0]
    print(f"\n[+] Winning B7 Configuration on Validation: {best_opt['config_name']} (Post F1={best_opt['post_f1']}, Harm={best_opt['harm_rate']})")

    # Save summary of Phase 3 & 4 tuning
    tuning_summary = {
        "best_weights": {"w_conf": best_wc, "w_feat": best_wf},
        "optimal_config": best_opt.to_dict(),
        "weight_ablation_table": weight_records,
        "hyperparameter_ablation_table": opt_results,
    }
    with open(ROOT / "results" / "phase3_phase4_tuning_summary.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)
    print(f"[+] Saved results/phase3_phase4_tuning_summary.json")

if __name__ == "__main__":
    main()
