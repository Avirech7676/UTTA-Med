"""Stage E2: Frozen Hospital-2 Target-Domain Evaluation Runner.

Executes prespecified target evaluation of the four experimental arms on
Camelyon17-WILDS Center 2 (Test slides 20-29, 128 patches/WSI).

Phase 1: Mandatory runtime configuration audit & hard invariant assertions.
Phase 2: Four-arm target evaluation (Pre-adaptation timing, per-WSI reset).
Phase 3: Monitoring validity tracking (Eligible decisions vs alarms).
Phase 4: Output telemetry persistence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent.parent
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


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_batch_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """Compute classification and calibration metrics on a batch or slide."""
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
        res["auroc"] = None  # Mathematically undefined on single-class slide
        res["auprc"] = 0.0
    return res


class TargetEvaluatorArm:
    """Standardized execution harness for a target evaluation arm."""

    def __init__(
        self,
        arm_id: str,
        arm_name: str,
        checkpoint_path: Path,
        tau_conf: float = 0.90,
        lr: float = 1e-5,
        mon_window_size: int = 50,
        mon_min_samples: int = 20,
    ):
        self.arm_id = arm_id
        self.arm_name = arm_name
        self.checkpoint_path = checkpoint_path
        self.tau_conf = tau_conf
        self.lr = lr
        self.mon_window_size = mon_window_size
        self.mon_min_samples = mon_min_samples

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
        """Reset model, optimizer, and controller to clean source state."""
        self.model.load_state_dict(deepcopy(self._immutable_source_state))
        self.model.eval()

        # Isolate BN affine parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.adaptable_params = []
        for mod in self.model.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and mod.affine:
                mod.eval()  # Freeze running stats
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

        # Arm-specific controller construction
        if self.arm_id == "arm_c":
            # Stage B: Numerical bounds, finite checks, atomic rollback
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
            # Stage C2 Active SafeAdapt: Stage B + Active Collapse Monitor
            mon_cfg = CollapseMonitorConfig(
                mode=MonitorMode.ACTIVE,
                window_size=self.mon_window_size,
                min_samples_for_alarm=self.mon_min_samples,
                min_window_samples=self.mon_min_samples,
                decision_threshold=0.5,
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
        # 1. PRE-ADAPTATION INFERENCE (Clinical Evaluation Time)
        # ----------------------------------------------------------------------
        with torch.no_grad():
            self.model.eval()
            out_pre = self.model(x)
            p_pre = torch.sigmoid(out_pre).view(-1)
            eval_probs = p_pre.cpu().numpy()

        telemetry: Dict[str, Any] = {
            "batch_size": b,
            "candidate_updates": 1 if self.arm_id != "arm_a" else 0,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "selected_patches": 0,
            "non_finite_failures": 0,
            "param_bound_clamps": 0,
            "eligible_monitor_decisions": 0,
            "monitor_warnings": 0,
            "rollbacks": 0,
            "fallbacks": 0,
            "max_displacement_linf": 0.0,
            "in_fallback": False,
        }

        # Arm A: Frozen source baseline (zero updates)
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

            # Monitor statistics tracking for Arm D
            if self.arm_id == "arm_d" and self.controller.monitor is not None:
                stats = self.controller.monitor.get_current_window_stats()
                if stats is not None and stats.is_sufficient_samples:
                    telemetry["eligible_monitor_decisions"] = 1
                    if stats.alarm_level == "WARNING":
                        telemetry["monitor_warnings"] = 1
                    elif stats.alarm_level in ("REJECT_UPDATE", "ROLLBACK"):
                        telemetry["rollbacks"] = 1
                    elif stats.alarm_level == "FALLBACK":
                        telemetry["fallbacks"] = 1

            telemetry["in_fallback"] = self.controller._in_fallback_mode
            telemetry["max_displacement_linf"] = self._get_max_disp()

        return eval_probs, telemetry

    def _get_max_disp(self) -> float:
        """Measure current maximum L_inf parameter displacement from source anchor."""
        max_d = 0.0
        for p in self.adaptable_params:
            src_p = self._source_param_anchors[id(p)]
            disp = torch.max(torch.abs(p.data - src_p.data)).item()
            if disp > max_d:
                max_d = disp
        return float(max_d)


def run_stage_e2_evaluation() -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print("UTTA-MED V2: STAGE E2 FROZEN HOSPITAL-2 TARGET EVALUATION", flush=True)
    print("=" * 80, flush=True)

    config_path = ROOT / "configs" / "v2" / "safeadapt_e2_frozen.yaml"
    ckpt_path = ROOT / "checkpoints" / "camelyon17_resnet18_source_s42_BEST.pt"

    # ==========================================================================
    # PHASE 1: MANDATORY RUNTIME CONFIGURATION AUDIT & INVARIANT ASSERTIONS
    # ==========================================================================
    print("\n--- PHASE 1: RUNTIME CONFIGURATION AUDIT & INVARIANT ASSERTIONS ---", flush=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Missing required configuration file: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    actual_ckpt_sha256 = compute_sha256(ckpt_path)
    expected_ckpt_sha256 = cfg["model"]["checkpoint_sha256"]

    print(f"Loaded Configuration: {config_path.name}")
    print(f"Target Hospital: {cfg['metadata']['target_hospital']}")
    print(f"Target WSIs: {cfg['execution']['eval_slides']}")
    print(f"Patches per WSI: {cfg['execution']['patches_per_slide']}")
    print(f"Batch Size: {cfg['adaptation']['batch_size']}")
    print(f"Confidence Threshold: {cfg['gating']['tau_conf']}")
    print(f"Prediction Timing: {cfg['execution']['prediction_timing']}")
    print(f"Monitor Window Size (W): {cfg['monitor']['window_size']}")
    print(f"Monitor Min Samples (min): {cfg['monitor']['min_samples_for_alarm']}")
    print(f"Source Checkpoint SHA-256: {actual_ckpt_sha256}")

    # Mandatory Hard Invariant Assertions (Fail Closed)
    assert actual_ckpt_sha256 == expected_ckpt_sha256, (
        f"Checkpoint SHA-256 mismatch! Got {actual_ckpt_sha256}, expected {expected_ckpt_sha256}"
    )
    assert cfg["monitor"]["window_size"] == 50, (
        f"Invariant Violation: monitor.window_size must be 50, got {cfg['monitor']['window_size']}"
    )
    assert cfg["monitor"]["min_samples_for_alarm"] == 20, (
        f"Invariant Violation: monitor.min_samples_for_alarm must be 20, got {cfg['monitor']['min_samples_for_alarm']}"
    )
    assert cfg["monitor"]["min_samples_for_alarm"] <= cfg["monitor"]["window_size"], (
        "Invariant Violation: min_samples_for_alarm exceeds window_size"
    )
    assert cfg["execution"]["patches_per_slide"] == 128, (
        f"Invariant Violation: patches_per_slide must be 128, got {cfg['execution']['patches_per_slide']}"
    )
    assert cfg["adaptation"]["batch_size"] == 32, (
        f"Invariant Violation: batch_size must be 32, got {cfg['adaptation']['batch_size']}"
    )
    assert cfg["gating"]["tau_conf"] == 0.90, (
        f"Invariant Violation: tau_conf must be 0.90, got {cfg['gating']['tau_conf']}"
    )
    assert cfg["execution"]["prediction_timing"] == "pre_adaptation", (
        f"Invariant Violation: prediction_timing must be pre_adaptation, got {cfg['execution']['prediction_timing']}"
    )
    assert cfg["execution"]["reset_policy"] == "per_wsi_boundary", (
        f"Invariant Violation: reset_policy must be per_wsi_boundary, got {cfg['execution']['reset_policy']}"
    )
    assert cfg["controller"]["max_param_displacement_linf"] == 0.05, (
        f"Invariant Violation: max_param_displacement_linf must be 0.05, got {cfg['controller']['max_param_displacement_linf']}"
    )

    print(">> ALL 10 PRE-EXECUTION INVARIANTS VERIFIED. RUNTIME CONFIGURATION LOCKED.\n", flush=True)

    # ==========================================================================
    # DATASET PREPARATION: CAMELYON17-WILDS CENTER 2 (TEST WSIS 20-29)
    # ==========================================================================
    print("Loading Center-2 Test Parquet Metadata ...", flush=True)
    default_data_root = os.environ.get(
        "CAMELYON17_ROOT",
        str(Path.home() / "OneDrive" / "Desktop" / "Camelyon17-Data" / "data")
        if (Path.home() / "OneDrive" / "Desktop" / "Camelyon17-Data" / "data").exists()
        else "data/camelyon17_v1.0",
    )
    data_dir = find_parquet_data_dir(default_data_root)
    camelyon = ParquetCamelyon17(data_dir)
    test_subset = camelyon.get_subset("test", transform=get_camelyon17_eval_transform())

    target_wsis = cfg["execution"]["eval_slides"]  # [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    patches_per_slide = cfg["execution"]["patches_per_slide"]  # 128
    batch_size = cfg["adaptation"]["batch_size"]  # 32

    print(f"Extracting {patches_per_slide} patches per WSI across {len(target_wsis)} target slides ...", flush=True)
    slide_data: Dict[int, Tuple[torch.Tensor, np.ndarray]] = {}
    for s in target_wsis:
        xs, ys = extract_slide_patches(test_subset, s, max_patches=patches_per_slide)
        slide_data[s] = (xs, ys.numpy())
        n_pos = int((ys == 1.0).sum())
        n_neg = int((ys == 0.0).sum())
        pos_pct = 100.0 * n_pos / len(ys)
        print(f"  Slide {s:2d} | Patches: {len(ys):3d} | Tumor: {n_pos:3d} ({pos_pct:5.1f}%) | Normal: {n_neg:3d}", flush=True)

    # ==========================================================================
    # PHASE 2: FOUR-ARM TARGET EVALUATION
    # ==========================================================================
    print("\n" + "=" * 70, flush=True)
    print("PHASE 2: FOUR-ARM TARGET BENCHMARK (WSIS 20-29, 1,280 PATCHES)", flush=True)
    print("=" * 70, flush=True)

    arm_defs = [
        ("arm_a", "Frozen Source"),
        ("arm_b", "Confidence TTA (Unsafeguarded)"),
        ("arm_c", "Stage B SafeAdapt (Bounds & Numerical)"),
        ("arm_d", "Stage C2 Active SafeAdapt (Dual Safeguards)"),
    ]

    results: Dict[str, Any] = {
        "metadata": {
            "evaluation_stage": "STAGE_E2_TARGET_EVALUATION",
            "target_cohort": "Camelyon17-WILDS Center 2 (Hospital 2)",
            "slides_evaluated": target_wsis,
            "patches_per_wsi": patches_per_slide,
            "total_patches": len(target_wsis) * patches_per_slide,
            "prediction_timing": cfg["execution"]["prediction_timing"],
            "reset_policy": cfg["execution"]["reset_policy"],
            "checkpoint_sha256": actual_ckpt_sha256,
            "config_sha256": compute_sha256(config_path),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "effective_runtime_config": cfg,
        "arms": {},
    }

    for arm_id, arm_name in arm_defs:
        print(f"\n--- Evaluating {arm_name} ---", flush=True)
        evaluator = TargetEvaluatorArm(
            arm_id=arm_id,
            arm_name=arm_name,
            checkpoint_path=ckpt_path,
            tau_conf=cfg["gating"]["tau_conf"],
            lr=cfg["adaptation"]["optimizer"]["lr"],
            mon_window_size=cfg["monitor"]["window_size"],
            mon_min_samples=cfg["monitor"]["min_samples_for_alarm"],
        )

        arm_preds_all: List[float] = []
        arm_labels_all: List[float] = []
        slide_metrics: Dict[int, Any] = {}
        slide_telemetry: Dict[int, Any] = {}

        t0 = time.perf_counter()

        for s in target_wsis:
            # WSI Boundary Reset
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
                "eligible_monitor_decisions": 0,
                "monitor_warnings": 0,
                "rollbacks": 0,
                "fallbacks": 0,
                "max_displacement_linf": 0.0,
                "in_fallback": False,
            }

            # 4 batches of 32
            n_batches = int(np.ceil(len(ys) / batch_size))
            for b_idx in range(n_batches):
                start = b_idx * batch_size
                end = min(start + batch_size, len(ys))
                bx = xs[start:end]

                b_probs, b_tel = evaluator.execute_batch(bx)
                s_preds.extend(b_probs.tolist())

                s_tel["candidate_updates"] += b_tel["candidate_updates"]
                s_tel["accepted_updates"] += b_tel["accepted_updates"]
                s_tel["rejected_updates"] += b_tel["rejected_updates"]
                s_tel["selected_patches"] += b_tel["selected_patches"]
                s_tel["non_finite_failures"] += b_tel["non_finite_failures"]
                s_tel["param_bound_clamps"] += b_tel["param_bound_clamps"]
                s_tel["eligible_monitor_decisions"] += b_tel["eligible_monitor_decisions"]
                s_tel["monitor_warnings"] += b_tel["monitor_warnings"]
                s_tel["rollbacks"] += b_tel["rollbacks"]
                s_tel["fallbacks"] += b_tel["fallbacks"]
                s_tel["max_displacement_linf"] = max(
                    s_tel["max_displacement_linf"], b_tel["max_displacement_linf"]
                )
                s_tel["in_fallback"] = b_tel["in_fallback"]

            # Compute slide metrics
            s_preds_arr = np.array(s_preds)
            s_m = evaluate_batch_metrics(s_preds_arr, ys)
            slide_metrics[s] = s_m
            slide_telemetry[s] = s_tel

            arm_preds_all.extend(s_preds)
            arm_labels_all.extend(ys.tolist())

            auroc_str = f"{s_m['auroc']:.4f}" if s_m["auroc"] is not None else "UNDEFINED"
            print(
                f"  WSI {s:2d} | AUROC: {auroc_str:9s} | F1: {s_m['f1']:.4f} | ECE: {s_m['ece']:.4f} | "
                f"Brier: {s_m['brier']:.6f} | Steps: {s_tel['accepted_updates']}/{s_tel['candidate_updates']} | "
                f"Patches: {s_tel['selected_patches']}/{len(ys)} | Disp: {s_tel['max_displacement_linf']:.6f} | "
                f"Decisions: {s_tel['eligible_monitor_decisions']}",
                flush=True,
            )

        elapsed = time.perf_counter() - t0
        all_preds = np.array(arm_preds_all)
        all_labels = np.array(arm_labels_all)
        overall_m = evaluate_batch_metrics(all_preds, all_labels)

        # Macro AUROC across valid dual-class slides
        valid_aurocs = [
            m["auroc"] for m in slide_metrics.values() if m["auroc"] is not None
        ]
        overall_m["macro_auroc_valid_slides"] = (
            float(np.mean(valid_aurocs)) if valid_aurocs else None
        )
        overall_m["valid_slides_count"] = len(valid_aurocs)

        results["arms"][arm_id] = {
            "arm_name": arm_name,
            "overall_metrics": overall_m,
            "slide_metrics": {str(k): v for k, v in slide_metrics.items()},
            "slide_telemetry": {str(k): v for k, v in slide_telemetry.items()},
            "elapsed_s": elapsed,
        }

        print(
            f"  >> Pooled AUROC: {overall_m['auroc']:.4f} | "
            f"Macro Valid AUROC: {overall_m['macro_auroc_valid_slides']:.4f} ({len(valid_aurocs)} slides) | "
            f"F1: {overall_m['f1']:.4f} | ECE: {overall_m['ece']:.4f} | "
            f"Brier: {overall_m['brier']:.6f} | Elapsed: {elapsed:.2f}s",
            flush=True,
        )

    # Save output telemetry
    out_dir = ROOT / "results" / "v2" / "stage_e2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage_e2_target_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nStage E2 Target Results successfully saved to: {out_file}", flush=True)
    return results


if __name__ == "__main__":
    run_stage_e2_evaluation()
