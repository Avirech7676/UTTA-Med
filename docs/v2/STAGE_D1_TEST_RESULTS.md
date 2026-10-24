# Stage D1: Deterministic CPU Test Results

**Auditor:** Independent Scientific Auditor & PyTorch Reliability Engineer  
**Date:** October 2026  
**Status:** All 14 Gating Tests Passed (99/101 Full Regression Suite Passed, 2 Skipped)  
**Execution Environment:** Intel CPU, Windows 11, PyTorch 2.14.0+cpu, pytest-9.1.1  

---

## 1. Executive Summary

This report documents the verification results of the Stage D1 CPU test suite (`tests/test_gating_v2.py`) and the complete repository regression suite.

All 14 unit and integration tests designed for Stage D1 gating pass deterministically without warnings or non-finite operations. The complete repository regression suite achieves **99 passed, 2 skipped, 0 failed**.

---

## 2. Stage D1 Gating Test Suite Breakdown (`tests/test_gating_v2.py`)

| Test ID | Test Function | Verified Property | Status |
| :---: | :--- | :--- | :---: |
| **01** | `test_1_patch_level_confidence_selection` | Verifies deterministic patch-by-patch confidence screening based on $\max(p, 1-p) > \tau_{\mathrm{conf}}$. | **PASSED** |
| **02** | `test_2_mc_predictive_variance_calculation` | Verifies predictive variance across stochastic passes matches v1 formula (`unbiased=False`). | **PASSED** |
| **03** | `test_3_reproducibility_under_fixed_random_seeds` | Verifies that identical random seeds produce bit-exact MC uncertainties and keep masks. | **PASSED** |
| **04** | `test_4_frozen_bn_statistics_during_mc_passes` | Verifies that BatchNorm `running_mean` and `running_var` are completely invariant during MC passes. | **PASSED** |
| **05** | `test_5_zero_mutation_of_source_weights` | Verifies that evaluating gating policies does not mutate any model weights or buffers. | **PASSED** |
| **06** | `test_6_correct_hybrid_arbitration` | Verifies that Hybrid gating auto-accepts high confidence, auto-rejects noise, and routes only ambiguous patches to MC. | **PASSED** |
| **07** | `test_7_correct_loss_masking_selected_patches` | Verifies that masked entropy loss evaluates strictly over selected patches ($\sum w_i > 0$). | **PASSED** |
| **08** | `test_8_zero_selected_patch_skips_optimizer_step` | Verifies that a batch with 0 selected patches bypasses backward pass, preserves exact model weights, and records 0 updates. | **PASSED** |
| **09** | `test_9_all_selected_patches_executes_full_adaptation` | Verifies that a batch with 100% selected patches executes a successful update and adapts BN affine weights. | **PASSED** |
| **10** | `test_10_matched_budget_accounting` | Verifies that telemetry correctly tracks patch coverage, update coverage, and prediction coverage ($100\%$). | **PASSED** |
| **11** | `test_11_monitor_disabled_and_observation_equivalence` | Verifies that setting the collapse monitor to `DISABLED` vs `OBSERVATION` produces bit-exact parameter updates. | **PASSED** |
| **12** | `test_12_preservation_of_stage_b_rollback` | Verifies that non-finite gradients trigger atomic rollback to pre-step state, preserving safe source predictions. | **PASSED** |
| **13** | `test_13_no_target_label_ingress` | Verifies through signature inspection that neither `policy.select` nor `adapter.adapt_batch` accepts target labels $y$. | **PASSED** |
| **14** | `test_14_no_modifications_to_frozen_v1_artifacts` | Verifies that frozen v1 checkpoints, tables, and authoritative configurations exist and match authoritative records. | **PASSED** |

---

## 3. Full Repository Regression Suite Execution

Executed via:
```bash
python -m pytest tests/ -q
```

### Official Output:
```text
......................................ss................................ [ 71%]
.............................                                            [100%]
============================== warnings summary ===============================
tests/test_coverage_risk.py::test_compute_selective_adaptation_slice
  ... UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined in that case.
99 passed, 2 skipped, 5 warnings in 25.08s
```

### Breakdown of Test Results:
- **Total Test Items Collected:** 101
- **Passed:** 99
- **Skipped:** 2 (`tests/test_no_leakage.py::test_dataset_paths_contain_no_test_hospital_references` and `tests/test_no_leakage.py::test_wilds_metadata_split_isolation`, which require raw uncompressed WILDS tarball paths when parquet metadata is present)
- **Failed:** 0
- **Deselected:** 0
- **Execution Time:** 25.08 seconds on CPU
