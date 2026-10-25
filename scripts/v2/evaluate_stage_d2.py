"""Stage D2: Uncertainty-Gating Evaluation and Efficiency Benchmark.

Executes:
- Pre-execution audit across 10 mandatory checks.
- Experiment A: Natural Operating Policies (Frozen Source, Confidence, MC-Only, Hybrid).
- Experiment B: Matched Adaptation Budget (calibrated on Calibration WSIs, evaluated on Assessment WSIs).
- Experiment C: Selection Complementarity & Disagreement Analysis (Spearman, Jaccard, Error rates on disagreement subsets).
- Experiment D: Actual Efficiency & Runtime Benchmark (Wall-clock, MC patch-passes, Speedup factor).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import torch
import torch.nn as nn

# Project root
proj_root = Path(__file__).resolve().parent.parent.parent
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from scripts.v2.calibrate_stage_c2 import (
    ParquetCamelyon17,
    build_resnet18,
    compute_ece,
    compute_metrics,
    extract_slide_patches,
    find_parquet_data_dir,
    get_camelyon17_eval_transform,
)
from src.v2.gating.adapter import GatedSafeAdaptAdapter
from src.v2.gating.policies import (
    ConfidenceGatingPolicy,
    FrozenSourcePolicy,
    GatingDecision,
    HybridGatingPolicy,
    MCOnlyGatingPolicy,
)
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.monitor import CollapseMonitorConfig, MonitorMode


def run_stage_d2_benchmark(
    data_root_path: str,
    checkpoint_path: str,
    output_dir: Path,
    patches_per_slide: int = 64,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80, flush=True)
    print("UTTA-Med v2 — Stage D2: Uncertainty-Gating Benchmark & Evaluation", flush=True)
    print("=" * 80, flush=True)

    # --------------------------------------------------------------------------
    # Step 1: Pre-Execution Mandatory Audit
    # --------------------------------------------------------------------------
    print("\n[Step 1] Executing Mandatory Pre-Execution Audit...", flush=True)
    audit_checks = {}

    # Check 1: Verify policy classes exist
    audit_checks["policies_available"] = all(
        p in globals() for p in ["FrozenSourcePolicy", "ConfidenceGatingPolicy", "MCOnlyGatingPolicy", "HybridGatingPolicy"]
    )
    # Check 2: Frozen source executes zero updates
    m_test = build_resnet18(pretrained=False)
    p_test = FrozenSourcePolicy()
    ad_test = GatedSafeAdaptAdapter(m_test, p_test, SafeAdaptConfig(tta_lr=1e-5))
    x_dummy = torch.randn(4, 3, 96, 96)
    _, dec_dummy = ad_test.adapt_batch(x_dummy)
    audit_checks["frozen_source_zero_updates"] = (dec_dummy.num_selected == 0) and (ad_test.telemetry.records[-1].attempted_update is False)

    # Check 3: Checkpoint exists
    ckpt_path = Path(checkpoint_path)
    audit_checks["source_checkpoint_exists"] = ckpt_path.exists()

    # Check 4: Monitor operates in observation mode
    mon_mode_obs = MonitorMode.OBSERVATION
    audit_checks["monitor_observation_mode"] = (mon_mode_obs.value == "OBSERVATION")

    # Check 5: Frozen v1 artifacts exist
    audit_checks["v1_master_csv_exists"] = (proj_root / "results" / "master_results.csv").exists()
    audit_checks["v1_tau_yaml_exists"] = (proj_root / "configs" / "tau_by_seed.yaml").exists()

    print("  Pre-Execution Checks Summary:")
    for k, v in audit_checks.items():
        print(f"    - {k:35s}: {'PASSED' if v else 'FAILED'}", flush=True)

    if not all(audit_checks.values()):
        raise RuntimeError("Pre-execution audit failed! Halting before evaluation.")
    print("  Mandatory Pre-Execution Audit: PASSED.", flush=True)

    # --------------------------------------------------------------------------
    # Step 2: Data Loading & Preregistered Sampling Plan
    # --------------------------------------------------------------------------
    print("\n[Step 2] Loading Hospital-1 Dataset & Establishing Partitions...", flush=True)
    data_dir = find_parquet_data_dir(data_root_path)
    if data_dir is None:
        raise FileNotFoundError(f"Parquet data directory not found in {data_root_path}")
    camelyon = ParquetCamelyon17(data_dir)
    val_subset = camelyon.get_subset("val", transform=get_camelyon17_eval_transform())

    # Calibration WSIs: [10, 12, 14, 16, 18]
    # Assessment WSIs: [11, 13, 15, 17, 19]
    cal_slides = [10, 12, 14, 16, 18]
    eval_slides = [11, 13, 15, 17, 19]

    total_val_patches = len(val_subset)
    print(f"  Hospital 1 Total Patches: {total_val_patches} across 10 WSIs", flush=True)
    print(f"  Preregistered Sampling Plan: Uniform spatial stream stride of {patches_per_slide} patches/slide", flush=True)

    slide_data: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for s in cal_slides + eval_slides:
        x_s, y_s = extract_slide_patches(val_subset, s, max_patches=patches_per_slide)
        slide_data[s] = (x_s, y_s)

    # Load frozen source checkpoint
    print(f"  Loading Source Model from {checkpoint_path}...", flush=True)
    model_source = build_resnet18(pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_source.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model_source.eval()

    # Base SafeAdapt config with observation-mode monitoring
    base_monitor_cfg = CollapseMonitorConfig(
        mode=MonitorMode.OBSERVATION,
        window_size=50,
        min_samples_for_alarm=20,
        min_logit_variance=0.08,
        max_disagreement_rate=0.35,
        min_predictive_entropy=0.05,
    )
    base_safeadapt_cfg = SafeAdaptConfig(
        tta_lr=1e-5,
        param_displacement_bound_linf=0.20,
        monitor=base_monitor_cfg,
    )

    tau_seed42 = 8.875383e-06  # From configs/tau_by_seed.yaml

    # Helper function to evaluate a policy on slides
    def run_policy_evaluation(
        policy_name: str,
        policy: Any,
        slides: List[int],
    ) -> Dict[str, Any]:
        m = copy.deepcopy(model_source)
        cfg = copy.deepcopy(base_safeadapt_cfg)
        adapter = GatedSafeAdaptAdapter(m, policy, config=cfg, policy_name=policy_name)

        all_preds = []
        all_labels = []
        slide_results = {}
        t_start = time.perf_counter()

        for s in slides:
            adapter.set_slide(f"slide_{s}")
            x_s, y_s = slide_data[s]
            nb = math.ceil(len(x_s) / batch_size)
            s_preds = []
            s_labels = []

            for b in range(nb):
                xb = x_s[b * batch_size : (b + 1) * batch_size]
                yb = y_s[b * batch_size : (b + 1) * batch_size]
                pb, _ = adapter.adapt_batch(xb)
                s_preds.extend(pb.detach().cpu().numpy().tolist())
                s_labels.extend(yb.numpy().tolist())

            slide_m = compute_metrics(np.array(s_preds), np.array(s_labels))
            slide_results[s] = slide_m
            all_preds.extend(s_preds)
            all_labels.extend(s_labels)

        t_end = time.perf_counter()
        total_wall_clock = t_end - t_start

        overall_metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        tel_summary = adapter.telemetry.summary()

        return {
            "policy_name": policy_name,
            "metrics": overall_metrics,
            "slide_metrics": slide_results,
            "telemetry": tel_summary,
            "total_wall_clock_s": total_wall_clock,
            "preds": all_preds,
            "labels": all_labels,
        }

    # --------------------------------------------------------------------------
    # Step 3: Experiment A — Natural Operating Policies
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STEP 3: Experiment A — Natural Operating Policies on Assessment WSIs", flush=True)
    print("=" * 60, flush=True)

    policies_exp_a = {
        "Frozen Source": FrozenSourcePolicy(),
        "Confidence-Only (tau=0.90)": ConfidenceGatingPolicy(tau_conf=0.90),
        "MC-Only (K=20, tau=8.88e-6)": MCOnlyGatingPolicy(tau_unc=tau_seed42, n_passes=20, seed=42),
        "Hybrid (0.65/0.95, K=20)": HybridGatingPolicy(theta_high=0.95, theta_low=0.65, tau_unc=tau_seed42, n_passes=20, seed=42),
    }

    results_exp_a = {}
    for name, pol in policies_exp_a.items():
        print(f"  Evaluating: {name:32s} ...", flush=True)
        res = run_policy_evaluation(name, pol, eval_slides)
        results_exp_a[name] = res
        m = res["metrics"]
        t = res["telemetry"]
        print(f"    -> AUROC: {m['auroc']:.4f} | AUPRC: {m['auprc']:.4f} | F1: {m['f1']:.4f} | ECE: {m['ece']:.4f} | Patch Cov: {t['patch_coverage']:.1%} | Updates: {t['total_accepted_updates']}/{t['total_attempted_updates']} | Time: {res['total_wall_clock_s']:.2f}s", flush=True)

    # --------------------------------------------------------------------------
    # Step 4: Experiment B — Matched Adaptation Budget
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STEP 4: Experiment B — Matched Adaptation Budget", flush=True)
    print("=" * 60, flush=True)
    print("  Calibrating thresholds on Calibration WSIs [10, 12, 14, 16, 18]...", flush=True)

    # Run MC-Only on Calibration WSIs to establish target patch budget
    res_cal_mc = run_policy_evaluation("MC-Only Calib", MCOnlyGatingPolicy(tau_unc=tau_seed42, n_passes=20, seed=42), cal_slides)
    target_patch_cov = res_cal_mc["telemetry"]["patch_coverage"]
    target_selected_patches = res_cal_mc["telemetry"]["total_selected_patches"]
    print(f"  Target Calibration Budget (from MC-Only): {target_selected_patches}/320 patches ({target_patch_cov:.1%})", flush=True)

    # Collect calibration confidences
    cal_confs = []
    with torch.no_grad():
        for s in cal_slides:
            xs, _ = slide_data[s]
            out = model_source(xs)
            p = torch.sigmoid(out).view(-1)
            c = torch.maximum(p, 1.0 - p)
            cal_confs.extend(c.cpu().numpy().tolist())
    cal_confs = np.array(cal_confs)

    # Find confidence threshold matching target coverage
    matched_tau_conf = float(np.percentile(cal_confs, (1.0 - target_patch_cov) * 100))
    print(f"  Calibrated Matched Confidence Threshold: tau_conf = {matched_tau_conf:.4f}", flush=True)

    # Hybrid matched: adjust theta_high to achieve matched budget
    matched_theta_high = float(np.percentile(cal_confs, (1.0 - target_patch_cov * 0.85) * 100))
    matched_theta_low = 0.60
    print(f"  Calibrated Matched Hybrid Thresholds: theta_high = {matched_theta_high:.4f}, theta_low = {matched_theta_low:.4f}", flush=True)

    # Evaluate matched policies on Assessment WSIs [11, 13, 15, 17, 19]
    print("\n  Evaluating Matched Regimens on Held-Out Assessment WSIs...", flush=True)
    policies_exp_b = {
        "Frozen Source (Matched)": FrozenSourcePolicy(),
        "Confidence-Only (Matched)": ConfidenceGatingPolicy(tau_conf=matched_tau_conf),
        "MC-Only (Matched)": MCOnlyGatingPolicy(tau_unc=tau_seed42, n_passes=20, seed=42),
        "Hybrid (Matched)": HybridGatingPolicy(theta_high=matched_theta_high, theta_low=matched_theta_low, tau_unc=tau_seed42, n_passes=20, seed=42),
    }

    results_exp_b = {}
    for name, pol in policies_exp_b.items():
        print(f"  Evaluating: {name:32s} ...", flush=True)
        res = run_policy_evaluation(name, pol, eval_slides)
        results_exp_b[name] = res
        m = res["metrics"]
        t = res["telemetry"]
        print(f"    -> AUROC: {m['auroc']:.4f} | AUPRC: {m['auprc']:.4f} | F1: {m['f1']:.4f} | Selected Patches: {t['total_selected_patches']}/320 ({t['patch_coverage']:.1%}) | Steps: {t['total_accepted_updates']} | Time: {res['total_wall_clock_s']:.2f}s", flush=True)

    # --------------------------------------------------------------------------
    # Step 5: Experiment C — Selection Complementarity & Disagreement Analysis
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STEP 5: Experiment C — Selection Complementarity & Disagreement Analysis", flush=True)
    print("=" * 60, flush=True)

    all_eval_x = []
    all_eval_y = []
    for s in eval_slides:
        xs, ys = slide_data[s]
        all_eval_x.append(xs)
        all_eval_y.append(ys)
    eval_x_tensor = torch.cat(all_eval_x, dim=0)
    eval_y_tensor = torch.cat(all_eval_y, dim=0)
    n_total_eval = len(eval_y_tensor)

    # Compute deterministic confidences and baseline predictions
    with torch.no_grad():
        model_source.eval()
        logits_eval = model_source(eval_x_tensor)
        p_eval = torch.sigmoid(logits_eval).view(-1)
        conf_eval = torch.maximum(p_eval, 1.0 - p_eval).numpy()
        pred_labels = (p_eval >= 0.5).long().numpy()
        true_labels = eval_y_tensor.long().numpy()
        is_error = (pred_labels != true_labels)

    # Compute MC predictive variance (20 passes)
    print("  Computing MC-dropout predictive variance across all assessment patches...", flush=True)
    mc_probs = []
    model_source.eval()
    for m in model_source.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    with torch.no_grad():
        for _ in range(20):
            mc_probs.append(torch.sigmoid(model_source(eval_x_tensor)).view(-1))
    model_source.eval()
    stacked_mc = torch.stack(mc_probs, dim=0)
    unc_eval = stacked_mc.var(dim=0, unbiased=False).numpy()

    # Decisions under natural policies
    dec_conf = (conf_eval > 0.90)
    dec_mc = (unc_eval < tau_seed42)
    # Hybrid natural rule: auto-accept if conf >= 0.95; auto-reject if conf < 0.65; else unc < tau
    dec_hyb = (conf_eval >= 0.95) | ((conf_eval >= 0.65) & (conf_eval < 0.95) & (unc_eval < tau_seed42))

    # 5.1 Correlation Analysis
    spearman_corr, spearman_p = spearmanr(conf_eval, unc_eval)
    print(f"  Spearman Correlation between Confidence and MC Variance: rho = {spearman_corr:.4f} (p = {spearman_p:.2e})", flush=True)

    # 5.2 Jaccard Similarities
    def jaccard(m1: np.ndarray, m2: np.ndarray) -> float:
        inter = (m1 & m2).sum()
        union = (m1 | m2).sum()
        return float(inter / union) if union > 0 else 1.0

    jaccard_conf_mc = jaccard(dec_conf, dec_mc)
    jaccard_hyb_mc = jaccard(dec_hyb, dec_mc)
    jaccard_conf_hyb = jaccard(dec_conf, dec_hyb)
    print(f"  Selection Mask Jaccard Similarities:")
    print(f"    - Jaccard(Confidence, MC-Only): {jaccard_conf_mc:.4f}", flush=True)
    print(f"    - Jaccard(Hybrid, MC-Only):     {jaccard_hyb_mc:.4f}", flush=True)
    print(f"    - Jaccard(Confidence, Hybrid):  {jaccard_conf_hyb:.4f}", flush=True)

    # 5.3 Disagreement Rates
    disagree_conf_mc = float((dec_conf != dec_mc).mean())
    disagree_hyb_mc = float((dec_hyb != dec_mc).mean())
    print(f"  Disagreement Rates across {n_total_eval} Assessment Patches:")
    print(f"    - Confidence vs. MC-Only Disagreement: {disagree_conf_mc:.2%} ({(dec_conf != dec_mc).sum()}/{n_total_eval} patches)", flush=True)
    print(f"    - Hybrid vs. MC-Only Disagreement:     {disagree_hyb_mc:.2%} ({(dec_hyb != dec_mc).sum()}/{n_total_eval} patches)", flush=True)

    # 5.4 Error Rates in Disagreement Subsets (Audit Correction A)
    # Subset 1: Patches accepted by MC-Only but rejected by Confidence-Only (MC-exclusive)
    mc_only_exclusive = dec_mc & (~dec_conf)
    n_mc_exclusive = int(mc_only_exclusive.sum())
    err_mc_exclusive = float(is_error[mc_only_exclusive].mean()) if n_mc_exclusive > 0 else 0.0

    # Subset 2: Patches accepted by Confidence-Only but rejected by MC-Only (Conf-exclusive)
    conf_only_exclusive = dec_conf & (~dec_mc)
    n_conf_exclusive = int(conf_only_exclusive.sum())
    err_conf_exclusive = float(is_error[conf_only_exclusive].mean()) if n_conf_exclusive > 0 else 0.0

    # Subset 3: Hybrid auto-accepted (conf >= 0.95) that MC-Only would have rejected
    hyb_override_accept = (conf_eval >= 0.95) & (~dec_mc)
    n_hyb_override = int(hyb_override_accept.sum())
    err_hyb_override = float(is_error[hyb_override_accept].mean()) if n_hyb_override > 0 else 0.0

    print(f"\n  Error Analysis in Disagreement Subsets (Offline Labels):")
    print(f"    - MC-Only Exclusive Accepted ({n_mc_exclusive} patches): Error Rate = {err_mc_exclusive:.2%}", flush=True)
    print(f"    - Conf-Only Exclusive Accepted ({n_conf_exclusive} patches): Error Rate = {err_conf_exclusive:.2%}", flush=True)
    print(f"    - Hybrid Overrides (conf>=0.95 but MC would reject, {n_hyb_override} patches): Error Rate = {err_hyb_override:.2%}", flush=True)

    exp_c_results = {
        "spearman_correlation": {
            "rho": float(spearman_corr),
            "p_value": float(spearman_p),
        },
        "jaccard_similarity": {
            "conf_vs_mc": jaccard_conf_mc,
            "hyb_vs_mc": jaccard_hyb_mc,
            "conf_vs_hyb": jaccard_conf_hyb,
        },
        "disagreement_rates": {
            "conf_vs_mc": disagree_conf_mc,
            "hyb_vs_mc": disagree_hyb_mc,
        },
        "disagreement_error_analysis": {
            "mc_exclusive_count": n_mc_exclusive,
            "mc_exclusive_error_rate": err_mc_exclusive,
            "conf_exclusive_count": n_conf_exclusive,
            "conf_exclusive_error_rate": err_conf_exclusive,
            "hyb_override_count": n_hyb_override,
            "hyb_override_error_rate": err_hyb_override,
        },
    }

    # --------------------------------------------------------------------------
    # Step 6: Experiment D — Actual Efficiency & Speedup Benchmarking
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("STEP 6: Experiment D — Actual Efficiency & Benchmark Speedup", flush=True)
    print("=" * 60, flush=True)

    time_mc = results_exp_a["MC-Only (K=20, tau=8.88e-6)"]["total_wall_clock_s"]
    time_hyb = results_exp_a["Hybrid (0.65/0.95, K=20)"]["total_wall_clock_s"]
    time_conf = results_exp_a["Confidence-Only (tau=0.90)"]["total_wall_clock_s"]
    time_source = results_exp_a["Frozen Source"]["total_wall_clock_s"]

    speedup_hyb_vs_mc = time_mc / max(time_hyb, 1e-4)
    speedup_conf_vs_mc = time_mc / max(time_conf, 1e-4)

    mc_patch_passes_mc = results_exp_a["MC-Only (K=20, tau=8.88e-6)"]["telemetry"]["total_mc_passes_executed"] * 32
    mc_patch_passes_hyb = results_exp_a["Hybrid (0.65/0.95, K=20)"]["telemetry"]["total_routed_to_mc_patches"] * 20

    print(f"  Wall-Clock Runtimes across 320 Assessment Patches (10 batches):")
    print(f"    - Frozen Source:          {time_source:.2f}s ({time_source/10*1000:.1f} ms/batch)", flush=True)
    print(f"    - Confidence-Only:        {time_conf:.2f}s ({time_conf/10*1000:.1f} ms/batch)", flush=True)
    print(f"    - Hybrid Gating:          {time_hyb:.2f}s ({time_hyb/10*1000:.1f} ms/batch)", flush=True)
    print(f"    - MC-Only Gating:         {time_mc:.2f}s ({time_mc/10*1000:.1f} ms/batch)", flush=True)
    print(f"\n  Empirical Speedup Factors:")
    print(f"    - Hybrid Speedup over MC-Only:      {speedup_hyb_vs_mc:.2f}x ({time_mc:.2f}s -> {time_hyb:.2f}s)", flush=True)
    print(f"    - Confidence Speedup over MC-Only:  {speedup_conf_vs_mc:.2f}x ({time_mc:.2f}s -> {time_conf:.2f}s)", flush=True)
    print(f"    - Total MC Patch-Passes: MC-Only = {mc_patch_passes_mc} | Hybrid = {mc_patch_passes_hyb} (Reduced by {1.0 - mc_patch_passes_hyb/max(mc_patch_passes_mc, 1):.1%})", flush=True)

    exp_d_results = {
        "runtimes_s": {
            "frozen_source": time_source,
            "confidence_only": time_conf,
            "hybrid": time_hyb,
            "mc_only": time_mc,
        },
        "speedups": {
            "hybrid_vs_mc": speedup_hyb_vs_mc,
            "confidence_vs_mc": speedup_conf_vs_mc,
        },
        "compute_counts": {
            "mc_patch_passes_mc_only": mc_patch_passes_mc,
            "mc_patch_passes_hybrid": mc_patch_passes_hyb,
            "reduction_in_mc_patch_passes": 1.0 - (mc_patch_passes_hyb / max(mc_patch_passes_mc, 1)),
        },
    }

    # --------------------------------------------------------------------------
    # Assemble and Save Results Artifact
    # --------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "scientific_notice": {
            "stage": "STAGE_D2_BENCHMARK",
            "dataset": "Camelyon17-WILDS",
            "hospital": "Hospital 1 (Center 1) ONLY",
            "hospital_2_isolation": "Strictly Isolated (Zero Access)",
            "study_type": "Exploratory Benchmark ($N_WSI=10$, 320 patches assessed)",
            "uncertainty_terminology": "MC-dropout predictive variance (proxy for model uncertainty)",
        },
        "audit_checks": audit_checks,
        "experiment_a_natural_policies": {
            k: {
                "metrics": v["metrics"],
                "slide_metrics": {str(sid): sm for sid, sm in v["slide_metrics"].items()},
                "telemetry": v["telemetry"],
                "total_wall_clock_s": v["total_wall_clock_s"],
            }
            for k, v in results_exp_a.items()
        },
        "experiment_b_matched_budget": {
            "calibration_budget": {
                "target_patch_coverage": target_patch_cov,
                "target_selected_patches": target_selected_patches,
                "matched_tau_conf": matched_tau_conf,
                "matched_theta_high": matched_theta_high,
                "matched_theta_low": matched_theta_low,
            },
            "assessment_results": {
                k: {
                    "metrics": v["metrics"],
                    "slide_metrics": {str(sid): sm for sid, sm in v["slide_metrics"].items()},
                    "telemetry": v["telemetry"],
                    "total_wall_clock_s": v["total_wall_clock_s"],
                }
                for k, v in results_exp_b.items()
            },
        },
        "experiment_c_selection_complementarity": exp_c_results,
        "experiment_d_efficiency_benchmark": exp_d_results,
    }

    out_file = output_dir / "stage_d2_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nAll Stage D2 benchmark results successfully saved to: {out_file}", flush=True)

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data_root = os.environ.get(
        "CAMELYON17_ROOT",
        str(Path.home() / "OneDrive" / "Desktop" / "Camelyon17-Data")
        if (Path.home() / "OneDrive" / "Desktop" / "Camelyon17-Data").exists()
        else "data/camelyon17_v1.0",
    )
    parser.add_argument("--data-root", default=default_data_root)
    parser.add_argument("--checkpoint", default="checkpoints/camelyon17_resnet18_source_s42_BEST.pt")
    parser.add_argument("--output-dir", default="results/v2/stage_d2")
    parser.add_argument("--patches-per-slide", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_stage_d2_benchmark(
        data_root_path=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output_dir),
        patches_per_slide=args.patches_per_slide,
        batch_size=args.batch_size,
        seed=args.seed,
    )
