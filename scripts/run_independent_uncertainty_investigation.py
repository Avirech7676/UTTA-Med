"""
Investigate Independent Uncertainty Information & Gating on Center 1 Validation Data
Evaluates MC-Dropout predictive variance, predictive entropy, mutual information,
conditional error discrimination within confidence bins, and class-specific dynamics.
"""

import os
import sys
import io
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, accuracy_score
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.resnet18 import build_resnet18, ResNet18Classifier

def load_validation_center1(n_patches: int = 600, seed: int = 42):
    """Load authentic Camelyon17 Center 1 validation patches from parquet."""
    parquet_path = Path("c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data/validation-00002-of-00003.parquet")
    pf = pq.ParquetFile(parquet_path)
    
    n_per_class = n_patches // 2
    patches_0 = []
    patches_1 = []
    
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["image", "label", "slide"])
        df = tbl.to_pandas()
        
        for _, row in df.iterrows():
            lbl = int(row["label"])
            if lbl == 0 and len(patches_0) < n_per_class:
                b = row["image"]["bytes"]
                im = Image.open(io.BytesIO(b)).convert("RGB")
                arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                patches_0.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl, int(row["slide"])))
            elif lbl == 1 and len(patches_1) < n_per_class:
                b = row["image"]["bytes"]
                im = Image.open(io.BytesIO(b)).convert("RGB")
                arr = (np.array(im, dtype=np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                patches_1.append((torch.from_numpy(arr).permute(2, 0, 1).float(), lbl, int(row["slide"])))
                
        if len(patches_0) >= n_per_class and len(patches_1) >= n_per_class:
            break
            
    combined = patches_0 + patches_1
    rng = np.random.default_rng(seed)
    rng.shuffle(combined)
    
    imgs = torch.stack([item[0] for item in combined])
    labels = np.array([item[1] for item in combined])
    slides = np.array([item[2] for item in combined])
    return imgs, labels, slides

def main():
    print("=" * 80)
    print("TASK 3 & 4: INDEPENDENT UNCERTAINTY INVESTIGATION (CENTER 1 VALIDATION)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running evaluation on device: {device}")
    
    # Load 600 validation patches (300 normal, 300 tumor)
    n_val = 600
    print(f"[*] Loading {n_val} Center 1 validation patches...")
    imgs, labels, slides = load_validation_center1(n_patches=n_val, seed=42)
    print(f"    Loaded {len(imgs)} patches (Tumor: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)})")
    
    # Load ResNet-18 checkpoint for seed 42
    ckpt_path = ROOT / "checkpoints" / "camelyon17_resnet18_source_s42_BEST.pt"
    print(f"[*] Loading checkpoint: {ckpt_path.name}")
    model = build_resnet18(num_classes=1, dropout_p=0.5, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in state:
        model.load_state_dict(state["model"])
    elif "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    # Extract features using single backbone pass
    print("[*] Running feature extraction (single backbone pass)...")
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), 64):
            b_imgs = imgs[i:i+64].to(device)
            f = model.forward_features(b_imgs)
            feats.append(f.cpu())
    feats = torch.cat(feats, dim=0).to(device)
    print(f"    Extracted features shape: {feats.shape}")
    
    # Evaluate MC-Dropout across head (T=20)
    print("[*] Running 20 stochastic MC-Dropout passes over classifier head...")
    torch.manual_seed(42)
    probs_mc = []
    T = 20
    with torch.no_grad():
        for t in range(T):
            # activate dropout in classifier head only
            model.classifier.train()
            logits = model.classifier(feats).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs_mc.append(p)
    probs_mc = np.array(probs_mc) # (20, N)
    
    # Single standard forward pass (dropout disabled)
    model.classifier.eval()
    with torch.no_grad():
        logits_std = model.classifier(feats).squeeze(-1)
        probs_std = torch.sigmoid(logits_std).cpu().numpy()
        
    p_mean = np.mean(probs_mc, axis=0)
    preds = (p_mean >= 0.5).astype(int)
    errors = (preds != labels).astype(int)
    error_rate = np.mean(errors)
    print(f"    Validation Error Rate: {error_rate:.4f} ({np.sum(errors)} errors / {len(errors)} patches)")
    
    # Uncertainty metrics
    # 1. Confidence-based uncertainty
    conf = np.maximum(p_mean, 1.0 - p_mean)
    conf_unc = 1.0 - conf
    
    # 2. Predictive variance
    var_mc = np.var(probs_mc, axis=0)
    
    # 3. Predictive entropy
    eps = 1e-12
    p_c = np.clip(p_mean, eps, 1.0 - eps)
    pred_entropy = - (p_c * np.log2(p_c) + (1.0 - p_c) * np.log2(1.0 - p_c))
    
    # 4. Expected entropy & Mutual Information
    p_mc_c = np.clip(probs_mc, eps, 1.0 - eps)
    sample_entropy = - (p_mc_c * np.log2(p_mc_c) + (1.0 - p_mc_c) * np.log2(1.0 - p_mc_c))
    exp_entropy = np.mean(sample_entropy, axis=0)
    mutual_info = np.maximum(pred_entropy - exp_entropy, 0.0)
    
    # 5. Centroid distance (cosine distance to class prototype)
    # Using source centroids for seed 42
    src_cents = torch.load(ROOT / "checkpoints" / "source_centroids_resnet18_s42.pt", map_location="cpu", weights_only=True)
    c0 = src_cents["prototypes"][0].squeeze()
    c1 = src_cents["prototypes"][1].squeeze()
    feats_np = feats.cpu().numpy()
    c0_np = c0.numpy() / np.linalg.norm(c0.numpy())
    c1_np = c1.numpy() / np.linalg.norm(c1.numpy())
    feats_norm = feats_np / np.linalg.norm(feats_np, axis=1, keepdims=True)
    dist0 = 1.0 - np.dot(feats_norm, c0_np)
    dist1 = 1.0 - np.dot(feats_norm, c1_np)
    # assign distance based on predicted class
    feat_dist = np.where(preds == 0, dist0, dist1)
    
    metrics_dict = {
        "Confidence_Uncertainty": conf_unc,
        "Predictive_Entropy": pred_entropy,
        "MC_Dropout_Variance": var_mc,
        "Mutual_Information": mutual_info,
        "Feature_Centroid_Distance": feat_dist
    }
    
    print("\n" + "=" * 80)
    print("PART A: OVERALL & CLASS-SPECIFIC ERROR DISCRIMINATION")
    print("=" * 80)
    
    tumor_mask = (labels == 1)
    normal_mask = (labels == 0)
    
    rows_uncertainty = []
    
    for name, metric_vals in metrics_dict.items():
        # Overall AUROC & AUPRC
        auroc_all = roc_auc_score(errors, metric_vals)
        prec_all, rec_all, _ = precision_recall_curve(errors, metric_vals)
        auprc_all = auc(rec_all, prec_all)
        
        # Tumor subset AUROC
        if np.sum(errors[tumor_mask]) > 0:
            auroc_tumor = roc_auc_score(errors[tumor_mask], metric_vals[tumor_mask])
        else:
            auroc_tumor = np.nan
            
        # Normal subset AUROC
        if np.sum(errors[normal_mask]) > 0:
            auroc_normal = roc_auc_score(errors[normal_mask], metric_vals[normal_mask])
        else:
            auroc_normal = np.nan
            
        rho, _ = stats.spearmanr(metric_vals, conf_unc)
        
        mean_corr = np.mean(metric_vals[errors == 0])
        mean_err = np.mean(metric_vals[errors == 1])
        sep_ratio = mean_err / (mean_corr + 1e-9)
        
        print(f"[*] {name:26s} | All AUROC: {auroc_all:.4f} | Tumor AUROC: {auroc_tumor:.4f} | Normal AUROC: {auroc_normal:.4f} | rho(conf): {rho:.4f}")
        
        rows_uncertainty.append({
            "uncertainty_metric": name,
            "error_auroc_overall": round(auroc_all, 4),
            "error_auprc_overall": round(auprc_all, 4),
            "error_auroc_tumor_subset": round(auroc_tumor, 4),
            "error_auroc_normal_subset": round(auroc_normal, 4),
            "spearman_rho_vs_confidence": round(rho, 4),
            "mean_on_correct_patches": round(float(mean_corr), 6),
            "mean_on_incorrect_patches": round(float(mean_err), 6),
            "error_separation_ratio": round(float(sep_ratio), 4)
        })
        
    print("\n" + "=" * 80)
    print("PART B: CONDITIONAL ERROR DISCRIMINATION WITHIN CONFIDENCE STRATA")
    print("=" * 80)
    
    # Bins: Low [0.50, 0.70), Moderate [0.70, 0.85), High [0.85, 0.95), Very High [0.95, 1.00]
    conf_bins = [
        ("Low_Confidence_[0.50,0.70)", 0.50, 0.70),
        ("Moderate_Confidence_[0.70,0.85)", 0.70, 0.85),
        ("High_Confidence_[0.85,0.95)", 0.85, 0.95),
        ("Very_High_Confidence_[0.95,1.00]", 0.95, 1.001)
    ]
    
    cond_rows = []
    
    for bin_name, c_min, c_max in conf_bins:
        bin_mask = (conf >= c_min) & (conf < c_max)
        n_in_bin = np.sum(bin_mask)
        n_err_in_bin = np.sum(errors[bin_mask])
        bin_err_rate = n_err_in_bin / (n_in_bin + 1e-9)
        
        print(f"\n--- Bin: {bin_name} (N = {n_in_bin}, Errors = {n_err_in_bin}, Rate = {bin_err_rate:.3%}) ---")
        
        if n_err_in_bin > 0 and n_err_in_bin < n_in_bin:
            # Check if MC variance discriminates errors within this confidence bin
            var_bin = var_mc[bin_mask]
            err_bin = errors[bin_mask]
            auroc_var_bin = roc_auc_score(err_bin, var_bin)
            
            # Mann-Whitney U test
            u_stat, p_val_var = stats.mannwhitneyu(var_bin[err_bin == 1], var_bin[err_bin == 0], alternative='greater')
            
            # Mutual info
            mi_bin = mutual_info[bin_mask]
            auroc_mi_bin = roc_auc_score(err_bin, mi_bin)
            _, p_val_mi = stats.mannwhitneyu(mi_bin[err_bin == 1], mi_bin[err_bin == 0], alternative='greater')
            
            print(f"    MC Variance: Within-Bin AUROC = {auroc_var_bin:.4f}, Mann-Whitney p = {p_val_var:.5f}")
            print(f"    Mutual Info: Within-Bin AUROC = {auroc_mi_bin:.4f}, Mann-Whitney p = {p_val_mi:.5f}")
            print(f"    Mean Var (Errors) = {np.mean(var_bin[err_bin==1]):.6f} vs Mean Var (Correct) = {np.mean(var_bin[err_bin==0]):.6f}")
        else:
            auroc_var_bin = np.nan
            p_val_var = np.nan
            auroc_mi_bin = np.nan
            p_val_mi = np.nan
            print(f"    Insufficient error variation to compute within-bin AUROC (n_err = {n_err_in_bin})")
            
        cond_rows.append({
            "confidence_stratum": bin_name,
            "sample_count": int(n_in_bin),
            "error_count": int(n_err_in_bin),
            "stratum_error_rate": round(float(bin_err_rate), 4),
            "mc_var_within_stratum_auroc": round(float(auroc_var_bin), 4) if not np.isnan(auroc_var_bin) else None,
            "mc_var_mann_whitney_p": round(float(p_val_var), 6) if not np.isnan(p_val_var) else None,
            "mutual_info_within_stratum_auroc": round(float(auroc_mi_bin), 4) if not np.isnan(auroc_mi_bin) else None,
            "mutual_info_mann_whitney_p": round(float(p_val_mi), 6) if not np.isnan(p_val_mi) else None,
            "residual_signal_significance": "Significant (p<0.05)" if p_val_var < 0.05 else "Insignificant (p>=0.05)" if not np.isnan(p_val_var) else "N/A"
        })
        
    # Save INDEPENDENT_UNCERTAINTY_ANALYSIS.csv
    df_ind = pd.DataFrame(rows_uncertainty)
    df_cond = pd.DataFrame(cond_rows)
    
    # Combine or format for INDEPENDENT_UNCERTAINTY_ANALYSIS.csv
    # Let's save a structured CSV containing both metric comparisons and stratum breakdown
    csv_rows = []
    for r in rows_uncertainty:
        csv_rows.append({
            "analysis_dimension": "Uncertainty_Metric_Error_Discrimination",
            "identifier": r["uncertainty_metric"],
            "metric_1": f"AUROC_all={r['error_auroc_overall']}",
            "metric_2": f"AUROC_tumor={r['error_auroc_tumor_subset']}",
            "metric_3": f"AUROC_normal={r['error_auroc_normal_subset']}",
            "metric_4": f"Spearman_rho={r['spearman_rho_vs_confidence']}",
            "metric_5": f"Separation_ratio={r['error_separation_ratio']}"
        })
    for r in cond_rows:
        csv_rows.append({
            "analysis_dimension": "Conditional_Error_Discrimination_By_Confidence_Stratum",
            "identifier": r["confidence_stratum"],
            "metric_1": f"N={r['sample_count']}_Errors={r['error_count']}",
            "metric_2": f"Error_rate={r['stratum_error_rate']}",
            "metric_3": f"MC_Var_AUROC={r['mc_var_within_stratum_auroc']}",
            "metric_4": f"MC_Var_p={r['mc_var_mann_whitney_p']}",
            "metric_5": f"Verdict={r['residual_signal_significance']}"
        })
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(ROOT / "INDEPENDENT_UNCERTAINTY_ANALYSIS.csv", index=False)
    print("\n[+] Successfully written INDEPENDENT_UNCERTAINTY_ANALYSIS.csv")
    
    print("\n" + "=" * 80)
    print("PART C: EVALUATE MINIMAL UNCERTAINTY-AWARE GATING POLICIES ON CENTER 1")
    print("=" * 80)
    
    # We evaluate candidate admission policies on Center 1:
    # 1. Confidence Gating at varied thresholds: tau_conf in {0.70, 0.75, 0.80, 0.85, 0.90}
    # 2. MC-Dropout Variance Gating at matched percentiles
    # 3. Class-Aware Asymmetric Gating (separate thresholds for tumor vs normal)
    # 4. Dual Filter (Confidence + Mutual Information)
    # 5. Dual Filter (Confidence + MC Variance)
    
    val_policies = []
    
    # Confidence gating baselines
    for tau in [0.70, 0.75, 0.80, 0.85, 0.90]:
        mask = (conf >= tau)
        cov = np.mean(mask)
        sel_err = np.mean(errors[mask]) if np.sum(mask) > 0 else 0.0
        sel_acc = 1.0 - sel_err
        tumor_cov = np.mean(mask[tumor_mask])
        normal_cov = np.mean(mask[normal_mask])
        val_policies.append({
            "policy_name": f"Confidence_Gating_tau_{tau:.2f}",
            "admission_rule": f"conf >= {tau:.2f}",
            "coverage": round(cov, 4),
            "selected_accuracy": round(sel_acc, 4),
            "selected_error_rate": round(sel_err, 4),
            "tumor_coverage": round(tumor_cov, 4),
            "normal_coverage": round(normal_cov, 4),
            "tumor_normal_coverage_ratio": round(tumor_cov / (normal_cov + 1e-9), 4),
            "policy_class": "Confidence Baseline"
        })
        
    # MC Variance Percentile thresholds
    for pctl in [50, 60, 70, 80, 90]:
        thresh = np.percentile(var_mc, pctl)
        mask = (var_mc <= thresh)
        cov = np.mean(mask)
        sel_err = np.mean(errors[mask]) if np.sum(mask) > 0 else 0.0
        sel_acc = 1.0 - sel_err
        tumor_cov = np.mean(mask[tumor_mask])
        normal_cov = np.mean(mask[normal_mask])
        val_policies.append({
            "policy_name": f"MC_Variance_Pctl_{pctl}",
            "admission_rule": f"var <= {thresh:.6f}",
            "coverage": round(cov, 4),
            "selected_accuracy": round(sel_acc, 4),
            "selected_error_rate": round(sel_err, 4),
            "tumor_coverage": round(tumor_cov, 4),
            "normal_coverage": round(normal_cov, 4),
            "tumor_normal_coverage_ratio": round(tumor_cov / (normal_cov + 1e-9), 4),
            "policy_class": "MC Variance Baseline"
        })
        
    # Class-Aware Asymmetric Gating (Candidate 1)
    thresh_t80 = np.percentile(var_mc[preds == 1], 80)
    thresh_n60 = np.percentile(var_mc[preds == 0], 60)
    mask_ca1 = np.where(preds == 1, var_mc <= thresh_t80, var_mc <= thresh_n60)
    cov_ca1 = np.mean(mask_ca1)
    sel_acc_ca1 = 1.0 - np.mean(errors[mask_ca1])
    val_policies.append({
        "policy_name": "Class_Aware_Asymmetric_Var_T80_N60",
        "admission_rule": f"preds==1 ? var<={thresh_t80:.6f} : var<={thresh_n60:.6f}",
        "coverage": round(cov_ca1, 4),
        "selected_accuracy": round(sel_acc_ca1, 4),
        "selected_error_rate": round(1.0 - sel_acc_ca1, 4),
        "tumor_coverage": round(np.mean(mask_ca1[tumor_mask]), 4),
        "normal_coverage": round(np.mean(mask_ca1[normal_mask]), 4),
        "tumor_normal_coverage_ratio": round(np.mean(mask_ca1[tumor_mask]) / (np.mean(mask_ca1[normal_mask]) + 1e-9), 4),
        "policy_class": "Class-Aware Uncertainty"
    })
    
    # Class-Aware Asymmetric Gating (Candidate 2 - Balanced)
    thresh_t85 = np.percentile(var_mc[preds == 1], 85)
    thresh_n75 = np.percentile(var_mc[preds == 0], 75)
    mask_ca2 = np.where(preds == 1, var_mc <= thresh_t85, var_mc <= thresh_n75)
    cov_ca2 = np.mean(mask_ca2)
    sel_acc_ca2 = 1.0 - np.mean(errors[mask_ca2])
    val_policies.append({
        "policy_name": "Class_Aware_Asymmetric_Var_T85_N75",
        "admission_rule": f"preds==1 ? var<={thresh_t85:.6f} : var<={thresh_n75:.6f}",
        "coverage": round(cov_ca2, 4),
        "selected_accuracy": round(sel_acc_ca2, 4),
        "selected_error_rate": round(1.0 - sel_acc_ca2, 4),
        "tumor_coverage": round(np.mean(mask_ca2[tumor_mask]), 4),
        "normal_coverage": round(np.mean(mask_ca2[normal_mask]), 4),
        "tumor_normal_coverage_ratio": round(np.mean(mask_ca2[tumor_mask]) / (np.mean(mask_ca2[normal_mask]) + 1e-9), 4),
        "policy_class": "Class-Aware Uncertainty"
    })
    
    # Dual Filter: Conf >= 0.80 AND Mutual Info <= 70th percentile (Candidate 3)
    mi_thresh70 = np.percentile(mutual_info, 70)
    mask_dual1 = (conf >= 0.80) & (mutual_info <= mi_thresh70)
    cov_dual1 = np.mean(mask_dual1)
    sel_acc_dual1 = 1.0 - np.mean(errors[mask_dual1])
    val_policies.append({
        "policy_name": "Dual_Gating_Conf80_MI70",
        "admission_rule": f"conf>=0.80 and MI<={mi_thresh70:.6f}",
        "coverage": round(cov_dual1, 4),
        "selected_accuracy": round(sel_acc_dual1, 4),
        "selected_error_rate": round(1.0 - sel_acc_dual1, 4),
        "tumor_coverage": round(np.mean(mask_dual1[tumor_mask]), 4),
        "normal_coverage": round(np.mean(mask_dual1[normal_mask]), 4),
        "tumor_normal_coverage_ratio": round(np.mean(mask_dual1[tumor_mask]) / (np.mean(mask_dual1[normal_mask]) + 1e-9), 4),
        "policy_class": "Dual Confidence + Epistemic"
    })
    
    # Dual Filter: Conf >= 0.85 AND MC Var <= 80th percentile (Candidate 4)
    var_thresh80 = np.percentile(var_mc, 80)
    mask_dual2 = (conf >= 0.85) & (var_mc <= var_thresh80)
    cov_dual2 = np.mean(mask_dual2)
    sel_acc_dual2 = 1.0 - np.mean(errors[mask_dual2])
    val_policies.append({
        "policy_name": "Dual_Gating_Conf85_Var80",
        "admission_rule": f"conf>=0.85 and var<={var_thresh80:.6f}",
        "coverage": round(cov_dual2, 4),
        "selected_accuracy": round(sel_acc_dual2, 4),
        "selected_error_rate": round(1.0 - sel_acc_dual2, 4),
        "tumor_coverage": round(np.mean(mask_dual2[tumor_mask]), 4),
        "normal_coverage": round(np.mean(mask_dual2[normal_mask]), 4),
        "tumor_normal_coverage_ratio": round(np.mean(mask_dual2[tumor_mask]) / (np.mean(mask_dual2[normal_mask]) + 1e-9), 4),
        "policy_class": "Dual Confidence + Variance"
    })
    
    df_val_pol = pd.DataFrame(val_policies)
    df_val_pol.to_csv(ROOT / "VALIDATION_GATING_COMPARISON.csv", index=False)
    print("[+] Successfully written VALIDATION_GATING_COMPARISON.csv")
    print("\nTop 5 Validation Policies by Selected Accuracy:")
    print(df_val_pol.sort_values(by="selected_accuracy", ascending=False)[["policy_name", "coverage", "selected_accuracy", "tumor_normal_coverage_ratio"]].head(5))

if __name__ == "__main__":
    main()
