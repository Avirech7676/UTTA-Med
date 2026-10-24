# Stage D2: Mandatory Pre-Execution Audit

**Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Reproducibility Auditor  
**Date:** October 2026  
**Status:** Audit Verified & Passed  
**Execution Environment:** Intel CPU, Windows 11, PyTorch 2.14.0+cpu  

---

## 1. Executive Summary

Prior to launching the Stage D2 evaluation and efficiency benchmark, the mandatory pre-execution audit was conducted across all ten designated criteria. All ten criteria have been verified, confirmed, and passed.

---

## 2. Ten Mandatory Pre-Execution Verification Points

| Item | Audit Check | Verification Mechanism | Status |
| :---: | :--- | :--- | :---: |
| **01** | Policy Implementations Match D1 Semantics | Verified in `src/v2/gating/policies.py`: Policy A (Frozen Source), Policy B (Confidence-only), Policy C (MC-only, 20 passes, $\text{Var}(p_k)$ with `unbiased=False`), and Policy D (Two-stage Hybrid). | **CONFIRMED** |
| **02** | Frozen Source Executes Zero Optimizer Steps | Tested in `test_8_zero_selected_patch_skips_optimizer_step` and `evaluate_stage_d2.py`: keep mask is all `False`, 0 backward passes, 0 optimizer updates. | **CONFIRMED** |
| **03** | Patch-Level Loss Masking Integrity | Verified in `GatedSafeAdaptAdapter.adapt_batch`: $\mathcal{L} = \frac{\sum w_i H(p_i)}{\sum w_i}$ with zero-selected-patch bypass. | **CONFIRMED** |
| **04** | 20 Stochastic Passes for MC Inference | Verified in `MCOnlyGatingPolicy` and `HybridGatingPolicy`: `n_passes = 20` exactly matching v1 specification. | **CONFIRMED** |
| **05** | Frozen BatchNorm Running Statistics | Verified in `test_4_frozen_bn_statistics_during_mc_passes`: `running_mean` and `running_var` are bit-exact invariant during MC inference and adaptation. | **CONFIRMED** |
| **06** | Identical Source Checkpoint Initialization | Verified: all regimens instantiate `model_source` loaded from `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` (`seed=42`). | **CONFIRMED** |
| **07** | Independent Optimizer & Identical WSI Ordering | Verified: every regimen creates a distinct `Adam` optimizer instance with $\text{lr}=10^{-5}$ and evaluates slides in identical sequential order `[11, 13, 15, 17, 19]`. | **CONFIRMED** |
| **08** | Collapse Monitor in Observation Mode | Verified: `SafeAdaptConfig.monitor.mode = MonitorMode.OBSERVATION` for all primary comparisons, avoiding conflation of collapse interventions with gating effects. | **CONFIRMED** |
| **09** | Zero Access to Hospital 2 | Verified: Center 2 (slides 20–29) was strictly unmounted and unaccessed. Zero Hospital-2 images or annotations queried. | **CONFIRMED** |
| **10** | Complete Regression Suite Passed | Executed `python -m pytest tests/ -q`: **99 passed, 2 skipped, 0 failed** in 25.08s. | **CONFIRMED** |

---

## 3. Preregistration of the WSI-Stratified Sampling Plan

In adherence to scientific guidelines from Stage C2.5:
- **Full Assessment Population:** 23,987 patches across WSIs `[11, 13, 15, 17, 19]`.
- **Compute Feasibility:** On host CPU, 20 MC passes on 23,987 patches requires $\approx 6,825\text{ seconds}$ ($\sim 1.9\text{ hours}$) per MC regimen.
- **Preregistered Stratified Sampling Plan:**
  - Patches per WSI: 64 patches sampled using deterministic uniform spatial stream striding (`step = n / max_patches`).
  - Total Evaluated Assessment Patches: **320 patches** ($1.33\%$ sampling fraction).
  - Total Evaluated Calibration Patches: **320 patches** ($2.93\%$ sampling fraction across slides `[10, 12, 14, 16, 18]`).
  - Pre-registered reporting constraint: No report shall claim 320 patches as a full-dataset evaluation.

---

## 4. Audit Sign-off

All pre-execution criteria are satisfied. Stage D2 benchmark execution is authorized.
