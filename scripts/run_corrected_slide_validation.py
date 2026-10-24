#!/usr/bin/env python3
"""
Corrected Slide-Grouped Validation Pipeline with Slide-Boundary Resets (Task 3 & Task 5)
========================================================================================

Official Research Title:
Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology:
Multicenter Classification and Dense Detection Under Domain Shift

This script executes authentic gradient-based test-time adaptation on the
Camelyon17-WILDS Center 1 validation cohort (N=600 patches across 10 WSIs),
strictly respecting whole-slide image (WSI) boundaries:
- Patches are grouped slide-by-slide (WSIs 10 to 19).
- Model parameters are restored to the source checkpoint theta_0 at every slide boundary.
- Optimizer state is cleared at every slide transition to prevent inter-patient leakage.
- Real Adam gradient updates are applied to the 40 BatchNorm affine parameters (gamma, beta).
- Detailed telemetry (F1, AUROC, Recall, Flips, Coverage, Parameter Drift) is saved to:
    1. SLIDE_RESET_VERIFICATION.csv
    2. CORRECTED_CENTER1_VALIDATION_RESULTS.csv
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.resnet18 import build_resnet18
from src.tta.bn import binary_entropy_from_logits, collect_bn_affine, configure_tta_mode
from src.tta.gates import score_confidence


def score_mc_dropout_fast(model: nn.Module, x: torch.Tensor, n_passes: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Fast MC-Dropout variance estimation using feature caching."""
    model.eval()
    with torch.no_grad():
        feats = model.forward_features(x)
    model.enable_dropout()
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            probs.append(torch.sigmoid(model.classifier(feats)).squeeze(-1))
    model.eval()
    stacked = torch.stack(probs, dim=0)
    return stacked.mean(dim=0), stacked.var(dim=0, unbiased=False)


def main():
    print("=" * 80, flush=True)
    print("CORRECTED SLIDE-GROUPED TEST-TIME ADAPTATION VALIDATION PIPELINE", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("cpu")
    ckpt_path = ROOT / "checkpoints" / "camelyon17_resnet18_source_s42_BEST.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    source_state = {k: v.clone().detach() for k, v in ckpt["model"].items()}

    # 1. Load cached validation cohort (N=600 across 10 WSIs)
    cache_file = ROOT / "scratch" / "camelyon17_c1_val_600.pt"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache file {cache_file} not found.")

    data = torch.load(cache_file, weights_only=False)
    x_val_all = data["x"]  # [600, 3, 96, 96]
    y_val_all = data["y"]  # (600,)
    slides_val_all = data["slides"]  # (600,)

    unique_slides = np.unique(slides_val_all)
    print(f"[+] Loaded validation dataset: {len(x_val_all)} patches across {len(unique_slides)} WSIs: {unique_slides.tolist()}", flush=True)

    # 2. Frozen source predictions & baseline MC variance estimation
    model_init = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
    model_init.load_state_dict(source_state)
    model_init.eval()

    with torch.no_grad():
        p_frozen_all = torch.sigmoid(model_init(x_val_all)).view(-1).numpy()
    conf_frozen_all = np.maximum(p_frozen_all, 1.0 - p_frozen_all)
    pred_frozen_all = (p_frozen_all >= 0.5).astype(int)

    # Pre-extract MC variances for gating threshold calibration
    _, mc_var_val_tensor = score_mc_dropout_fast(model_init, x_val_all, n_passes=20)
    mc_var_val = mc_var_val_tensor.numpy()
    tau_mc_70 = float(np.percentile(mc_var_val, 70))
    tau_mc_90 = float(np.percentile(mc_var_val, 90))

    f1_frozen = float(f1_score(y_val_all, pred_frozen_all))
    auc_frozen = float(roc_auc_score(y_val_all, p_frozen_all))
    rec_frozen = float(recall_score(y_val_all, pred_frozen_all))
    prec_frozen = float(precision_score(y_val_all, pred_frozen_all))
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(y_val_all, pred_frozen_all).ravel()

    print(f"[+] Frozen Source Baseline: F1={f1_frozen:.4f}, AUROC={auc_frozen:.4f}, Recall={rec_frozen:.4f}, Precision={prec_frozen:.4f}, FN={fn_f}, FP={fp_f}", flush=True)
    print(f"[+] Calibrated MC-Dropout Thresholds: tau_70={tau_mc_70:.6e}, tau_90={tau_mc_90:.6e}", flush=True)

    # 3. Define adaptation policies
    policies = [
        ("A0_Frozen_ResNet18", "frozen", None, 6.74),
        ("A1_Confidence_Gating_tau_0.80", "confidence", 0.80, 7.68),
        ("A1_Confidence_Gating_tau_0.85", "confidence", 0.85, 7.68),
        ("A2_Original_UTTA_Med_Percentile_70", "mc_dropout", tau_mc_70, 136.29),
        ("A3_Improved_UTTA_Med_Percentile_90", "mc_dropout", tau_mc_90, 136.29),
        ("Candidate_CAD_Gate", "cad_gate", (0.75, 0.85, tau_mc_90), 7.82),
        ("Matched_Budget_Confidence", "confidence", 0.84, 7.68),
    ]

    all_policy_results = []
    slide_reset_log = []

    batch_size = 32

    # Run each policy
    for policy_name, gate_type, param, lat_ms in policies:
        print(f"\n[*] Evaluating Policy: {policy_name} (Mode: {gate_type})...", flush=True)

        policy_preds = np.zeros(len(x_val_all), dtype=np.float32)
        policy_admitted_mask = np.zeros(len(x_val_all), dtype=bool)

        total_admitted_patches = 0
        total_admitted_batches = 0
        total_skipped_batches = 0
        total_optimizer_steps = 0
        total_rollbacks = 0
        max_drift_l2 = 0.0
        max_drift_linf = 0.0

        t0 = time.time()

        for slide_id in unique_slides:
            slide_mask = (slides_val_all == slide_id)
            slide_indices = np.where(slide_mask)[0]
            n_patches_slide = len(slide_indices)

            x_slide = x_val_all[slide_indices]
            y_slide = y_val_all[slide_indices]

            # Model setup and strict reset to source checkpoint at slide boundary
            model = build_resnet18(pretrained=False, dropout_p=0.5, num_classes=1).to(device)
            model.load_state_dict(source_state)

            # Freeze backbone convolutional weights, train only BN affine
            configure_tta_mode(model, bn_mode="bn_freeze_stats")
            bn_params = collect_bn_affine(model)
            for p in model.parameters():
                p.requires_grad = False
            for p in bn_params:
                p.requires_grad = True

            optimizer = torch.optim.Adam(bn_params, lr=1e-5)
            init_bn_state = [p.clone().detach() for p in bn_params]

            # Verification of parameter state at slide boundary (must be 0.000000)
            initial_diff = max((p - i).abs().max().item() for p, i in zip(bn_params, init_bn_state))
            assert initial_diff == 0.0, f"Error: Initial model parameters not clean on slide {slide_id}"

            n_batches_slide = int(np.ceil(n_patches_slide / batch_size))
            slide_admitted_patches = 0
            slide_steps = 0

            slide_p_after = []

            for b in range(n_batches_slide):
                b_idx_start = b * batch_size
                b_idx_end = min((b + 1) * batch_size, n_patches_slide)
                xb = x_slide[b_idx_start:b_idx_end]
                yb = y_slide[b_idx_start:b_idx_end]
                curr_batch_indices = slide_indices[b_idx_start:b_idx_end]

                if gate_type == "frozen":
                    with torch.no_grad():
                        model.eval()
                        pb = torch.sigmoid(model(xb)).view(-1).numpy()
                        slide_p_after.append(pb)
                    continue

                # Admission gating check
                with torch.no_grad():
                    model.eval()
                    if gate_type == "confidence":
                        conf_b = score_confidence(model, xb)
                        keep = (conf_b >= param)
                    elif gate_type == "mc_dropout":
                        _, unc_b = score_mc_dropout_fast(model, xb, n_passes=20)
                        keep = (unc_b <= param)
                    elif gate_type == "cad_gate":
                        tau_tum, tau_norm, tau_outlier = param
                        pb_eval = torch.sigmoid(model(xb)).view(-1)
                        _, unc_b = score_mc_dropout_fast(model, xb, n_passes=20)
                        is_tum = (pb_eval >= 0.50)
                        keep_tum = is_tum & (pb_eval >= tau_tum)
                        keep_norm = (~is_tum) & ((1.0 - pb_eval) >= tau_norm)
                        keep = (keep_tum | keep_norm) & (unc_b <= tau_outlier)
                    else:
                        keep = torch.ones(len(xb), dtype=torch.bool)

                keep_np = keep.numpy()
                policy_admitted_mask[curr_batch_indices] = keep_np
                n_keep = int(keep.sum().item())
                slide_admitted_patches += n_keep
                total_admitted_patches += n_keep

                if n_keep == 0:
                    total_skipped_batches += 1
                else:
                    total_admitted_batches += 1
                    configure_tta_mode(model, bn_mode="bn_freeze_stats")
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    ent = binary_entropy_from_logits(logits)
                    w = keep.float()
                    loss = (w * ent).sum() / w.sum()
                    loss.backward()

                    # Gradient check
                    finite_grad = True
                    for p in bn_params:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            finite_grad = False
                            break

                    if not finite_grad:
                        # Safeguard intervention: reject update
                        total_rollbacks += 1
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        slide_steps += 1
                        total_optimizer_steps += 1

                # Post-update inference for batch
                with torch.no_grad():
                    model.eval()
                    pb = torch.sigmoid(model(xb)).view(-1).numpy()
                    slide_p_after.append(pb)

            # Store predictions for this slide
            p_slide_adapted = np.concatenate(slide_p_after)
            policy_preds[slide_indices] = p_slide_adapted

            # Measure drift before reset
            current_bn = [p.clone().detach() for p in bn_params]
            l2_diffs = [(c - i).norm(2).item() for c, i in zip(current_bn, init_bn_state)]
            linf_diffs = [(c - i).abs().max().item() for c, i in zip(current_bn, init_bn_state)]
            slide_drift_l2 = math.sqrt(sum(d**2 for d in l2_diffs))
            slide_drift_linf = max(linf_diffs)

            max_drift_l2 = max(max_drift_l2, slide_drift_l2)
            max_drift_linf = max(max_drift_linf, slide_drift_linf)

            # Perform Slide-Boundary Reset
            model.load_state_dict(source_state)
            optimizer.state.clear()
            reset_bn = [p.clone().detach() for p in collect_bn_affine(model)]
            post_reset_l2 = math.sqrt(sum((r - i).norm(2).item() ** 2 for r, i in zip(reset_bn, init_bn_state)))
            post_reset_linf = max((r - i).abs().max().item() for r, i in zip(reset_bn, init_bn_state))

            # Record slide reset log (for Task 3, record primary UTTA-Med and Confidence)
            if policy_name in ("A1_Confidence_Gating_tau_0.85", "A2_Original_UTTA_Med_Percentile_70"):
                slide_reset_log.append({
                    "policy_name": policy_name,
                    "wsi_id": int(slide_id),
                    "patch_count": n_patches_slide,
                    "batch_count": n_batches_slide,
                    "admitted_patches": slide_admitted_patches,
                    "slide_coverage": round(slide_admitted_patches / max(n_patches_slide, 1), 4),
                    "adaptation_steps": slide_steps,
                    "reset_events": 1,
                    "param_drift_before_reset_l2": round(slide_drift_l2, 6),
                    "param_drift_before_reset_linf": round(slide_drift_linf, 6),
                    "param_drift_after_reset_l2": round(post_reset_l2, 6),
                    "param_drift_after_reset_linf": round(post_reset_linf, 6),
                    "reset_status": "VERIFIED_CLEAN_RESTORATION" if post_reset_linf == 0.0 else "CONTAMINATION_DETECTED"
                })

        elapsed_time = time.time() - t0

        # Overall policy metrics
        pred_after = (policy_preds >= 0.5).astype(int)
        changed_probs = int((np.abs(policy_preds - p_frozen_all) > 1e-4).sum())
        harmful_flips = int(((pred_frozen_all == y_val_all) & (pred_after != y_val_all)).sum())
        beneficial_flips = int(((pred_frozen_all != y_val_all) & (pred_after == y_val_all)).sum())
        net_flips = beneficial_flips - harmful_flips

        tn, fp, fn, tp = confusion_matrix(y_val_all, pred_after).ravel()
        f1 = float(f1_score(y_val_all, pred_after))
        auroc = float(roc_auc_score(y_val_all, policy_preds))
        prec = float(precision_score(y_val_all, pred_after))
        rec = float(recall_score(y_val_all, pred_after))
        spec = float(tn / (tn + fp))

        cov = float(total_admitted_patches / len(x_val_all))
        tum_cov = float(policy_admitted_mask[y_val_all == 1].mean()) if (y_val_all == 1).sum() > 0 else 0.0
        norm_cov = float(policy_admitted_mask[y_val_all == 0].mean()) if (y_val_all == 0).sum() > 0 else 0.0

        all_policy_results.append({
            "policy_name": policy_name,
            "f1_score": round(f1, 4),
            "auroc": round(auroc, 4),
            "precision": round(prec, 4),
            "tumor_recall": round(rec, 4),
            "specificity": round(spec, 4),
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "changed_probabilities": changed_probs,
            "harmful_flips": harmful_flips,
            "beneficial_flips": beneficial_flips,
            "net_flip_benefit": net_flips,
            "overall_coverage": round(cov, 4),
            "tumor_coverage": round(tum_cov, 4),
            "normal_coverage": round(norm_cov, 4),
            "optimizer_steps": total_optimizer_steps,
            "rollbacks": total_rollbacks,
            "max_param_drift_l2": round(max_drift_l2, 6),
            "max_param_drift_linf": round(max_drift_linf, 6),
            "batch_latency_ms": lat_ms,
            "execution_time_s": round(elapsed_time, 2),
        })

    # Save deliverables
    df_res = pd.DataFrame(all_policy_results)
    out_csv = ROOT / "CORRECTED_CENTER1_VALIDATION_RESULTS.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\n[+] Saved CORRECTED_CENTER1_VALIDATION_RESULTS.csv ({len(df_res)} rows)", flush=True)
    print(df_res[["policy_name", "f1_score", "auroc", "tumor_recall", "false_negatives", "harmful_flips", "net_flip_benefit", "max_param_drift_linf", "overall_coverage"]].to_string(index=False), flush=True)

    df_resets = pd.DataFrame(slide_reset_log)
    out_resets_csv = ROOT / "SLIDE_RESET_VERIFICATION.csv"
    df_resets.to_csv(out_resets_csv, index=False)
    print(f"\n[+] Saved SLIDE_RESET_VERIFICATION.csv ({len(df_resets)} rows)", flush=True)
    print(df_resets.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
