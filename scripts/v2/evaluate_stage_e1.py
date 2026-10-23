#!/usr/bin/env python3
"""Stage E1: Confidence-Gated SafeAdapt Evaluation & Stress Benchmark.

Executes the four-arm evaluation on Hospital-1 Assessment WSIs:
- Arm A: Frozen Source Baseline
- Arm B: Confidence-Only TTA (Unsafeguarded)
- Arm C: Stage B Confidence-Gated SafeAdapt (Parameter Bounds & Numerical Safeguards)
- Arm D: Stage C2 Active SafeAdapt (Stage B + Active Collapse Monitoring & Fallback)

Protocols:
1. Primary Protocol: Per-WSI Independent Reset (Natural, Reverse, and Tumor-Ascending Orders)
2. Continuous-Stream Stress Test: No Reset Across WSI Boundaries (Natural and Tumor-Ascending)

Prediction Timing: Strictly PRE-ADAPTATION (batch predictions recorded before optimizer step).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

ROOT = Path("c:/Users/avina/OneDrive/Desktop/UTTA-Med")
sys.path.insert(0, str(ROOT))

from scripts.v2.calibrate_stage_c2 import (
    ParquetCamelyon17,
    build_resnet18,
    compute_ece,
    compute_metrics,
    extract_slide_patches,
    find_parquet_data_dir,
    get_camelyon17_eval_transform,
)
from src.v2.gating.policies import ConfidenceGatingPolicy
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController
from src.v2.safeadapt.monitor import (
    AlarmLevel,
    CollapseMonitor,
    CollapseMonitorConfig,
    MonitorMode,
)


def evaluate_batch_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    preds = (probs >= 0.5).astype(int)
    res: Dict[str, Any] = {
        "n": len(labels),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "brier": float(np.mean((probs - labels) ** 2)),
        "ece": float(compute_ece(probs, labels)),
    }
    if len(np.unique(labels)) > 1:
        res["auroc"] = float(roc_auc_score(labels, probs))
        res["auprc"] = float(average_precision_score(labels, probs))
    else:
        res["auroc"] = None  # Explicitly mathematically undefined for single-class slide
        res["auprc"] = 0.0
    return res


class EvaluatorArm:
    """Standardized execution harness for an experimental arm."""

    def __init__(
        self,
        arm_id: str,
        arm_name: str,
        checkpoint_path: Path,
        tau_conf: float = 0.90,
        lr: float = 1e-5,
    ):
        self.arm_id = arm_id
        self.arm_name = arm_name
        self.checkpoint_path = checkpoint_path
        self.tau_conf = tau_conf
        self.lr = lr

        # Load reference source model
        self.source_model = build_resnet18(pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.source_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        self.source_model.eval()

        self._immutable_source_state = {
            k: v.detach().clone() for k, v in self.source_model.state_dict().items()
        }

        # Active working model
        self.model = build_resnet18(pretrained=False)
        self.reset_to_source()

    def reset_to_source(self) -> None:
        self.model.load_state_dict(deepcopy(self._immutable_source_state))
        self.model.eval()

        # Isolate BN affine parameters
        for p in self.model.parameters():
            p.requires_grad = False
        self.adaptable_params = []
        for mod in self.model.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and mod.affine:
                if mod.weight is not None:
                    mod.weight.requires_grad = True
                    self.adaptable_params.append(mod.weight)
                if mod.bias is not None:
                    mod.bias.requires_grad = True
                    self.adaptable_params.append(mod.bias)

        self.optimizer = torch.optim.Adam(self.adaptable_params, lr=self.lr)
        self._source_param_anchors = {
            id(p): p.detach().clone() for p in self.adaptable_params
        }

        # Arm-specific controllers
        if self.arm_id == "arm_c":
            # Stage B Safeguards: bounds, finite checks, rollback
            cfg = SafeAdaptConfig(
                tta_lr=self.lr,
                param_displacement_bound_linf=0.05,
                max_grad_norm=1.0,
                check_finite_loss=True,
                check_finite_gradients=True,
                check_finite_parameters=True,
                monitor=CollapseMonitorConfig(mode=MonitorMode.DISABLED),
            )
            self.controller = SafeAdaptController(
                model=self.model,
                optimizer=self.optimizer,
                config=cfg,
                adaptable_params=self.adaptable_params,
            )
        elif self.arm_id == "arm_d":
            # Stage C2 Active Safeguards: Stage B + Active Collapse Monitor
            mon_cfg = CollapseMonitorConfig(
                mode=MonitorMode.ACTIVE,
                window_size=10,
                min_logit_variance=0.08,
                max_disagreement_rate=0.35,
                min_predictive_entropy=0.05,
                consecutive_suspicious_for_rollback=2,
                consecutive_suspicious_for_fallback=4,
            )
            cfg = SafeAdaptConfig(
                tta_lr=self.lr,
                param_displacement_bound_linf=0.05,
                max_grad_norm=1.0,
                check_finite_loss=True,
                check_finite_gradients=True,
                check_finite_parameters=True,
                monitor=mon_cfg,
            )
            self.controller = SafeAdaptController(
                model=self.model,
                optimizer=self.optimizer,
                config=cfg,
                adaptable_params=self.adaptable_params,
            )
        else:
            self.controller = None

    def execute_batch(self, x: torch.Tensor) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Execute pre-adaptation prediction followed by adaptation step."""
        b = x.size(0)

        # ----------------------------------------------------------------------
        # 1. PRE-ADAPTATION PREDICTION (Locked Stage E1 timing)
        # ----------------------------------------------------------------------
        with torch.no_grad():
            self.model.eval()
            logits_pre = self.model(x)
            p_pre = torch.sigmoid(logits_pre).view(-1)
            eval_probs = p_pre.cpu().numpy()

        telemetry: Dict[str, Any] = {
            "batch_size": b,
            "candidate_updates": 1 if self.arm_id != "arm_a" else 0,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "selected_patches": 0,
            "non_finite_failures": 0,
            "param_bound_clamps": 0,
            "monitor_warnings": 0,
            "rollbacks": 0,
            "fallbacks": 0,
            "max_displacement_linf": 0.0,
            "in_fallback": False,
        }

        # Arm A: Zero adaptation
        if self.arm_id == "arm_a":
            return eval_probs, telemetry

        # Confidence Selection (Threshold 0.90)
        c_pre = np.maximum(eval_probs, 1.0 - eval_probs)
        keep_mask = (c_pre >= self.tau_conf)
        n_selected = int(keep_mask.sum())
        telemetry["selected_patches"] = n_selected

        if n_selected == 0:
            # Zero-selected patch bypass
            telemetry["max_displacement_linf"] = self._get_max_disp()
            return eval_probs, telemetry

        w = torch.from_numpy(keep_mask).float()

        # Loss function
        def compute_loss() -> torch.Tensor:
            self.model.train()
            # Enforce frozen BN stats & dropout eval
            for m in self.model.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Dropout)):
                    m.eval()
            out = self.model(x)
            p = torch.sigmoid(out).clamp(1e-6, 1.0 - 1e-6).view(-1)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            return (w * ent).sum() / w.sum()

        # ----------------------------------------------------------------------
        # 2. ADAPTATION UPDATE
        # ----------------------------------------------------------------------
        if self.arm_id == "arm_b":
            # Arm B: Unsafeguarded TTA
            try:
                loss = compute_loss()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                telemetry["accepted_updates"] = 1
            except Exception as e:
                telemetry["rejected_updates"] = 1
                telemetry["non_finite_failures"] = 1
            telemetry["max_displacement_linf"] = self._get_max_disp()

        elif self.arm_id in ("arm_c", "arm_d"):
            # Arm C & Arm D: SafeAdapt Controller
            _, accepted = self.controller.step(
                x=x,
                compute_loss_fn=compute_loss,
            )
            if accepted:
                telemetry["accepted_updates"] = 1
            else:
                telemetry["rejected_updates"] = 1

            # Extract controller telemetry
            latest_events = self.controller.telemetry.events[-3:] if self.controller.telemetry.events else []
            for ev in latest_events:
                if ev.event_type == "ROLLBACK":
                    telemetry["rollbacks"] += 1
                    if "non_finite" in ev.reason:
                        telemetry["non_finite_failures"] += 1
                elif ev.event_type == "FALLBACK":
                    telemetry["fallbacks"] += 1
                elif ev.event_type == "ALARM_WARNING":
                    telemetry["monitor_warnings"] += 1

            telemetry["in_fallback"] = self.controller.in_fallback_mode
            telemetry["max_displacement_linf"] = self._get_max_disp()

        return eval_probs, telemetry

    def _get_max_disp(self) -> float:
        max_d = 0.0
        for p in self.adaptable_params:
            src_p = self._source_param_anchors[id(p)]
            disp = torch.max(torch.abs(p.data - src_p.data)).item()
            if disp > max_d:
                max_d = disp
        return float(max_d)


def run_evaluation_suite() -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print("STAGE E1: CONFIDENCE-GATED SAFEADAPT FOUR-ARM BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    ckpt_path = ROOT / "checkpoints" / "camelyon17_resnet18_source_s42_BEST.pt"
    data_dir = find_parquet_data_dir("C:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data")
    camelyon = ParquetCamelyon17(data_dir)
    val_subset = camelyon.get_subset("val", transform=get_camelyon17_eval_transform())

    all_eval_wsis = [11, 13, 15, 17, 19]
    slide_data: Dict[int, Tuple[torch.Tensor, np.ndarray]] = {}
    for s in all_eval_wsis:
        xs, ys = extract_slide_patches(val_subset, s, max_patches=64)
        slide_data[s] = (xs, ys.numpy())

    arm_defs = [
        ("arm_a", "Frozen Source"),
        ("arm_b", "Confidence TTA (Unsafeguarded)"),
        ("arm_c", "Stage B SafeAdapt (Bounds & Numerical)"),
        ("arm_d", "Stage C2 Active SafeAdapt (Dual Safeguards)"),
    ]

    # Schedules
    schedules = {
        "order_1_natural": [11, 13, 15, 17, 19],
        "order_2_reverse": [19, 17, 15, 13, 11],
        "order_3_tumor_ascending": [15, 19, 17, 11, 13],  # Label-informed stress test
    }

    # ==========================================================================
    # EXPERIMENT 1: PRIMARY PROTOCOL (RESET PER WSI)
    # ==========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 1: PRIMARY EVALUATION (PER-WSI BOUNDARY RESET)", flush=True)
    print("=" * 70, flush=True)

    primary_results: Dict[str, Dict[str, Any]] = {}

    for arm_id, arm_name in arm_defs:
        print(f"\n--- Evaluating {arm_name} (Order 1: Natural Clinical) ---", flush=True)
        evaluator = EvaluatorArm(arm_id, arm_name, ckpt_path)

        arm_preds_all: List[float] = []
        arm_labels_all: List[float] = []
        slide_metrics: Dict[int, Any] = {}
        slide_telemetry: Dict[int, Any] = {}

        t0 = time.perf_counter()

        for s in schedules["order_1_natural"]:
            evaluator.reset_to_source()
            if evaluator.controller is not None:
                evaluator.controller.set_slide(str(s))

            xs, ys = slide_data[s]
            s_preds: List[float] = []
            s_tel: Dict[str, Any] = {
                "candidate_updates": 0,
                "accepted_updates": 0,
                "rejected_updates": 0,
                "selected_patches": 0,
                "non_finite_failures": 0,
                "param_bound_clamps": 0,
                "monitor_warnings": 0,
                "rollbacks": 0,
                "fallbacks": 0,
                "max_displacement_linf": 0.0,
            }

            # Batch size 32 -> 2 batches per slide
            for b_idx in range(0, len(xs), 32):
                xb = xs[b_idx : b_idx + 32]
                pb, tb = evaluator.execute_batch(xb)
                s_preds.extend(pb.tolist())

                s_tel["candidate_updates"] += tb["candidate_updates"]
                s_tel["accepted_updates"] += tb["accepted_updates"]
                s_tel["rejected_updates"] += tb["rejected_updates"]
                s_tel["selected_patches"] += tb["selected_patches"]
                s_tel["non_finite_failures"] += tb["non_finite_failures"]
                s_tel["param_bound_clamps"] += tb["param_bound_clamps"]
                s_tel["monitor_warnings"] += tb["monitor_warnings"]
                s_tel["rollbacks"] += tb["rollbacks"]
                s_tel["fallbacks"] += tb["fallbacks"]
                s_tel["max_displacement_linf"] = max(s_tel["max_displacement_linf"], tb["max_displacement_linf"])

            s_m = evaluate_batch_metrics(np.array(s_preds), ys)
            slide_metrics[s] = s_m
            slide_telemetry[s] = s_tel

            arm_preds_all.extend(s_preds)
            arm_labels_all.extend(ys.tolist())

            auroc_str = f"{s_m['auroc']:.4f}" if s_m['auroc'] is not None else "UNDEFINED"
            print(f"  WSI {s:<4} | AUROC: {auroc_str:<9} | F1: {s_m['f1']:.4f} | ECE: {s_m['ece']:.4f} | Brier: {s_m['brier']:.6f} | Steps: {s_tel['accepted_updates']}/{s_tel['candidate_updates']} | Patches: {s_tel['selected_patches']}/64 | Disp: {s_tel['max_displacement_linf']:.6f}", flush=True)

        t1 = time.perf_counter()
        elapsed_s = t1 - t0

        overall_m = evaluate_batch_metrics(np.array(arm_preds_all), np.array(arm_labels_all))
        # Macro AUROC excluding undefined single-class slides
        valid_aurocs = [m["auroc"] for m in slide_metrics.values() if m["auroc"] is not None]
        overall_m["macro_auroc_valid_slides"] = float(np.mean(valid_aurocs))

        print(f"  >> Pooled AUROC: {overall_m['auroc']:.4f} | Macro Valid AUROC: {overall_m['macro_auroc_valid_slides']:.4f} | F1: {overall_m['f1']:.4f} | ECE: {overall_m['ece']:.4f} | Brier: {overall_m['brier']:.6f} | Elapsed: {elapsed_s:.2f}s", flush=True)

        primary_results[arm_id] = {
            "arm_name": arm_name,
            "overall_metrics": overall_m,
            "slide_metrics": {str(k): v for k, v in slide_metrics.items()},
            "slide_telemetry": {str(k): v for k, v in slide_telemetry.items()},
            "elapsed_s": elapsed_s,
            "preds": arm_preds_all,
            "labels": arm_labels_all,
        }

    # ==========================================================================
    # EXPERIMENT 2: ORDER REPRODUCIBILITY (RESET PER WSI UNDER ORDER 2 & 3)
    # ==========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 2: SLIDE ORDER INVARIANCE CHECK (WITH PER-WSI RESET)", flush=True)
    print("=" * 70, flush=True)

    order_invariance_results: Dict[str, Any] = {}

    for sched_name, sched_slides in [("order_2_reverse", schedules["order_2_reverse"]), ("order_3_tumor_ascending", schedules["order_3_tumor_ascending"])]:
        print(f"\n--- Checking Schedule: {sched_name} ({sched_slides}) ---", flush=True)
        sched_res: Dict[str, Any] = {}
        for arm_id, arm_name in arm_defs:
            evaluator = EvaluatorArm(arm_id, arm_name, ckpt_path)
            slide_preds_map: Dict[int, List[float]] = {}
            for s in sched_slides:
                evaluator.reset_to_source()
                if evaluator.controller is not None:
                    evaluator.controller.set_slide(str(s))
                xs, ys = slide_data[s]
                s_preds: List[float] = []
                for b_idx in range(0, len(xs), 32):
                    xb = xs[b_idx : b_idx + 32]
                    pb, _ = evaluator.execute_batch(xb)
                    s_preds.extend(pb.tolist())
                slide_preds_map[s] = s_preds

            # Reconstruct natural order [11, 13, 15, 17, 19] to verify bit-exact invariance
            reconstructed_preds: List[float] = []
            for s in [11, 13, 15, 17, 19]:
                reconstructed_preds.extend(slide_preds_map[s])

            diff_with_order1 = float(np.max(np.abs(np.array(reconstructed_preds) - np.array(primary_results[arm_id]["preds"]))))
            is_identical = (diff_with_order1 < 1e-6)
            print(f"  {arm_name:<42} | Diff with Order 1: {diff_with_order1:.2e} | Invariant: {'PASS' if is_identical else 'FAIL'}", flush=True)
            sched_res[arm_id] = {
                "max_abs_diff_with_order1": diff_with_order1,
                "is_invariant": is_identical,
            }
        order_invariance_results[sched_name] = sched_res

    # ==========================================================================
    # EXPERIMENT 3: CONTINUOUS-STREAM STRESS TEST (NO RESET ACROSS WSIS)
    # ==========================================================================
    print("\n" + "=" * 70, flush=True)
    print("EXPERIMENT 3: CONTINUOUS-STREAM STRESS TEST (NO RESET ACROSS WSIS)", flush=True)
    print("=" * 70, flush=True)

    continuous_results: Dict[str, Any] = {}

    for sched_name, sched_slides in [("continuous_natural", schedules["order_1_natural"]), ("continuous_tumor_ascending", schedules["order_3_tumor_ascending"])]:
        print(f"\n--- Running Continuous Stream: {sched_name} ({sched_slides}) ---", flush=True)
        stream_res: Dict[str, Any] = {}

        for arm_id, arm_name in arm_defs:
            evaluator = EvaluatorArm(arm_id, arm_name, ckpt_path)
            # INITIALIZE ONCE AT START OF STREAM, ZERO RESETS IN BETWEEN
            evaluator.reset_to_source()

            stream_preds: List[float] = []
            stream_labels: List[float] = []
            stream_slide_m: Dict[int, Any] = {}
            stream_slide_t: Dict[int, Any] = {}

            t0 = time.perf_counter()

            for s in sched_slides:
                if evaluator.controller is not None:
                    # Notify slide ID without weight reset
                    evaluator.controller.set_slide(str(s))

                xs, ys = slide_data[s]
                s_preds: List[float] = []
                s_tel: Dict[str, Any] = {
                    "accepted_updates": 0,
                    "rejected_updates": 0,
                    "rollbacks": 0,
                    "fallbacks": 0,
                    "max_displacement_linf": 0.0,
                    "in_fallback": False,
                }

                for b_idx in range(0, len(xs), 32):
                    xb = xs[b_idx : b_idx + 32]
                    pb, tb = evaluator.execute_batch(xb)
                    s_preds.extend(pb.tolist())

                    s_tel["accepted_updates"] += tb["accepted_updates"]
                    s_tel["rejected_updates"] += tb["rejected_updates"]
                    s_tel["rollbacks"] += tb["rollbacks"]
                    s_tel["fallbacks"] += tb["fallbacks"]
                    s_tel["max_displacement_linf"] = max(s_tel["max_displacement_linf"], tb["max_displacement_linf"])
                    s_tel["in_fallback"] = tb["in_fallback"]

                s_m = evaluate_batch_metrics(np.array(s_preds), ys)
                stream_slide_m[s] = s_m
                stream_slide_t[s] = s_tel

                stream_preds.extend(s_preds)
                stream_labels.extend(ys.tolist())

            t1 = time.perf_counter()
            stream_ov = evaluate_batch_metrics(np.array(stream_preds), np.array(stream_labels))
            valid_aurocs = [m["auroc"] for m in stream_slide_m.values() if m["auroc"] is not None]
            stream_ov["macro_auroc_valid_slides"] = float(np.mean(valid_aurocs))

            end_disp = evaluator._get_max_disp()
            print(f"  {arm_name:<42} | AUROC: {stream_ov['auroc']:.4f} | F1: {stream_ov['f1']:.4f} | ECE: {stream_ov['ece']:.4f} | End Disp: {end_disp:.6f} | Total Fallbacks: {sum(t['fallbacks'] for t in stream_slide_t.values())}", flush=True)

            stream_res[arm_id] = {
                "arm_name": arm_name,
                "overall_metrics": stream_ov,
                "slide_metrics": {str(k): v for k, v in stream_slide_m.items()},
                "slide_telemetry": {str(k): v for k, v in stream_slide_t.items()},
                "final_param_displacement_linf": end_disp,
                "elapsed_s": t1 - t0,
            }

        continuous_results[sched_name] = stream_res

    # ==========================================================================
    # SAVE MACHINE-READABLE BENCHMARK ARTIFACT
    # ==========================================================================
    out_dir = ROOT / "results" / "v2" / "stage_e1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage_e1_evaluation_results.json"

    # Strip raw preds/labels from final payload to keep compact
    clean_primary = {}
    for k, v in primary_results.items():
        clean_primary[k] = {
            "arm_name": v["arm_name"],
            "overall_metrics": v["overall_metrics"],
            "slide_metrics": v["slide_metrics"],
            "slide_telemetry": v["slide_telemetry"],
            "elapsed_s": v["elapsed_s"],
        }

    results_payload = {
        "scientific_notice": {
            "stage": "STAGE_E1_EVALUATION",
            "dataset": "Camelyon17-WILDS",
            "hospital": "Hospital 1 (Center 1) ONLY",
            "hospital_2_isolation": "Strictly Isolated (Zero Access)",
            "prediction_timing": "PRE-ADAPTATION (batch predicted before parameter update)",
            "checkpoint_sha256": "f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000",
        },
        "primary_experiment_wsi_reset": clean_primary,
        "order_invariance_checks": order_invariance_results,
        "continuous_stream_stress_test": continuous_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nStage E1 Evaluation JSON successfully saved to: {out_file}", flush=True)
    return results_payload


if __name__ == "__main__":
    run_evaluation_suite()
