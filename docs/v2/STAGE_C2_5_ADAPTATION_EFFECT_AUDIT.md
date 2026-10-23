# Stage C2.5: Adaptation-Effect & Metric Identity Audit

**Auditor:** Independent Scientific Auditor & PyTorch Reliability Engineer  
**Status:** Complete  
**Date:** October 2026  
**Dataset:** Camelyon17-WILDS (Hospital 1 Assessment Partition)  
**Scope:** Forensic Verification of Adaptation Dynamics, Coverage, and Metric Invariance  

---

## 1. Executive Summary

This forensic audit investigates why the Stage C2 comparative evaluation reported identical metrics (AUROC 0.9740, AUPRC 0.9548, F1 0.8632, ECE 0.0795) across five distinct test-time adaptation regimens: Frozen Source, Stage B (Unmonitored), Stage C1 (Observation), Stage C2 (Active Calibrated), and Confidence-Gated TTA.

### Forensic Conclusions:
1. **Implementation Flaw in Stage C2 Script for Frozen Source:** In `scripts/v2/calibrate_stage_c2.py`, the regimen labelled "Frozen Source" was inadvertently evaluated by calling the active adaptation controller (`evaluate_controller_mode`), executing real Stage B gradient updates on the model, and then manually overwriting `"coverage": 0.0`. It did **not** evaluate a frozen source model.
2. **True Source vs. Stage B Performance Disclosed:** When a truly frozen source model is evaluated without adaptation, the metrics differ subtly:
   - **True Frozen Source:** AUROC 0.973593, AUPRC 0.954238, F1 0.863248, ECE 0.078624, Brier 0.082867.
   - **True Stage B (Adapted):** AUROC 0.974026, AUPRC 0.954830, F1 0.863248, ECE 0.079454, Brier 0.082422.
   - Adaptation produced a tiny ranking gain ($\Delta \text{AUROC} = +0.000433$, $\Delta \text{AUPRC} = +0.000592$), a slight Brier improvement ($\Delta \text{Brier} = -0.000445$), and a slight calibration worsening ($\Delta \text{ECE} = +0.000830$).
3. **Mathematical Cause of Identical F1 (0.863248):** Across all 320 patches, **zero predictions crossed the $\tau=0.5$ decision boundary**. The confusion matrices for True Frozen Source and Adapted Stage B were 100% identical (101 TP, 23 FP, 187 TN, 9 FN).
4. **Why Monitoring and Gating Regimens Matched Stage B:**
   - **Stage C1 (Observation):** Telemetry-only by definition; zero interventions.
   - **Stage C2 (Active):** Zero alarms tripped on healthy assessment tissue; zero rollbacks; accepted 100% of Stage B steps.
   - **Confidence-Gated:** The gating threshold required mean batch confidence $< 0.65$ to skip updates; empirical mean confidence was $0.9851$ (minimum batch confidence $> 0.90$). The condition never triggered, accepting 100% of Stage B steps.

---

## 2. Task B — Identical Performance Metrics Forensic

### 2.1 Code Inspection of `evaluate_controller_mode`
In `scripts/v2/calibrate_stage_c2.py`:
```python
results_source = evaluate_controller_mode("Frozen Source", MonitorMode.DISABLED, use_confidence_gating=False)
results_source["coverage"] = 0.0

results_stage_b = evaluate_controller_mode("Stage B (Unmonitored)", MonitorMode.DISABLED)
results_stage_c1 = evaluate_controller_mode("Stage C1 (Observation)", MonitorMode.OBSERVATION)
results_stage_c2 = evaluate_controller_mode("Stage C2 (Active Calibrated)", MonitorMode.ACTIVE)
results_conf_gate = evaluate_controller_mode("Confidence-Gated", MonitorMode.DISABLED, use_confidence_gating=True)
```

Inside `evaluate_controller_mode`:
```python
m = copy.deepcopy(model_source)
ctrl, opt = make_bn_controller(m, cfg)
...
for b in range(n_batches):
    ...
    if should_skip_update:
        with torch.no_grad():
            preds_b = torch.sigmoid(m(x_b)).view(-1)
        accepted = False
    else:
        preds_b, accepted = ctrl.step(x_b, compute_loss_fn=compute_loss)
```

For `results_source`, `use_confidence_gating=False`, meaning `should_skip_update=False`. The controller executed `ctrl.step(...)` on every batch, adapting the model weights. The post-hoc assignment `results_source["coverage"] = 0.0` changed a metadata label but did not alter the predictions, which were the output of Stage B adaptation.

### 2.2 Forensic Re-Evaluation: True Frozen Source vs. Adapted Regimens
Evaluating the true frozen source model without any adaptation steps yields the authoritative high-precision comparison on the 320 assessment patches:

| Regimen | AUROC | AUPRC | F1 ($\tau=0.5$) | ECE | Brier Score | Max Prob Diff vs. Source | Mean Prob Diff vs. Source | Threshold Flips ($\Delta \hat{y} \ne 0$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True Frozen Source** | **0.973593** | **0.954238** | **0.863248** | **0.078624** | **0.082867** | — | — | — |
| **Stage B (Unmonitored)** | **0.974026** | **0.954830** | **0.863248** | **0.079454** | **0.082422** | 0.036898 | 0.000989 | 0 / 320 |
| **Stage C1 (Observation)** | **0.974026** | **0.954830** | **0.863248** | **0.079454** | **0.082422** | 0.036898 | 0.000989 | 0 / 320 |
| **Stage C2 (Active Calibrated)**| **0.974026** | **0.954830** | **0.863248** | **0.079454** | **0.082422** | 0.036898 | 0.000989 | 0 / 320 |
| **Confidence-Gated** | **0.974026** | **0.954830** | **0.863248** | **0.079454** | **0.082422** | 0.036898 | 0.000989 | 0 / 320 |

### 2.3 Why F1 is Invariant at $\tau=0.5$
The mean absolute change in probability from adaptation is $0.000989$ ($0.099\%$). The maximum single-patch probability change across all 320 patches is $0.036898$ ($3.69\%$). Because the source model was already well-separated (normal patches near 0.0, tumor patches near 1.0), this magnitude of probability adjustment did not move a single sample across $0.5$. 

Consequently:
$$\text{TP} = 101, \quad \text{FP} = 23, \quad \text{TN} = 187, \quad \text{FN} = 9$$
$$F_1 = \frac{2 \times 101}{2 \times 101 + 23 + 9} = \frac{202}{234} = 0.8632478632478632...$$
The F1 score is mathematically identical across all regimens because the confusion matrix did not change by a single entry.

---

## 3. Task C — Comprehensive Coverage Accounting

To prevent misleading claims regarding adaptation coverage, the three distinct dimensions of coverage must be formally separated:

1. **Patch Coverage:** Fraction of eligible assessment patches evaluated.
2. **Update Coverage:** Fraction of candidate gradient steps accepted by the controller.
3. **Prediction Coverage:** Fraction of evaluated patches receiving a valid prediction.

| Regimen | Eligible Patches | Evaluated Patches | Patch Coverage | Attempted Optimizer Steps | Accepted Optimizer Steps | Rejected Steps | Update Coverage | Prediction Coverage | Changed Predictions ($\Delta \hat{y} \ne 0$) | Max Param Displacement ($\|\theta - \theta_0\|_\infty$) | Mean Absolute Prob Diff ($\mathbb{E}[\|\hat{p} - p_0\|]$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True Frozen Source** | 23,987 | 320 | 1.33% | 0 | 0 | 0 | 0.0% | 100.0% | 0 | 0.000000 | 0.000000 |
| **Stage B (Unmonitored)** | 23,987 | 320 | 1.33% | 10 | 10 | 0 | 100.0% | 100.0% | 0 | 0.000099 | 0.000989 |
| **Stage C1 (Observation)** | 23,987 | 320 | 1.33% | 10 | 10 | 0 | 100.0% | 100.0% | 0 | 0.000099 | 0.000989 |
| **Stage C2 (Active)** | 23,987 | 320 | 1.33% | 10 | 10 | 0 | 100.0% | 100.0% | 0 | 0.000099 | 0.000989 |
| **Confidence-Gated** | 23,987 | 320 | 1.33% | 10 | 10 | 0 | 100.0% | 100.0% | 0 | 0.000099 | 0.000989 |

### Parameter Displacement Analysis:
After 10 cumulative Adam steps with learning rate $10^{-5}$ across 5 slides, the maximum parameter displacement across all 9,600 BatchNorm affine parameters was:
$$\max \|\theta - \theta_0\|_\infty = 0.00009946 \approx 1.0 \times 10^{-4}.$$
This confirms that the optimizer stepped as mathematically expected ($10 \times 10^{-5}$ scaled by Adam momentum), producing minor affine adjustments that caused subtle probability calibration changes without causing model drift.

---

## 4. Task E — Alarm Integration & Real Controller Verification

Audit confirmed that synthetic collapse testing and controller verification are not mock-based:

1. **Absence of Mock Frameworks:** Neither `unittest.mock`, `MagicMock`, nor monkeypatching is used in the SafeAdapt controller test suite.
2. **Real Layer Execution:** All tests in `tests/test_safeadapt_v2.py` and `tests/test_safeadapt_c1_monitor.py` execute real PyTorch modules (`BatchNorm1d`, `Linear`, `ResNet`), compute autograd gradients, execute real `torch.optim.Adam` updates, and verify state dictionary equality.
3. **Distinction Between Unit and Integration Tests:**
   - **Unit Tests (`test_2` through `test_8` in `test_safeadapt_c1_monitor.py`):** Directly invoke `monitor.update()` with crafted tensor distributions to isolate rolling-window mathematics, variance calculation, and boundary reset logic.
   - **Integration Tests (`test_1`, `test_9`, `test_10`, `test_11` in `test_safeadapt_c1_monitor.py`):** Exercise the complete controller execution pipeline:
     $$\text{Forward} \to \text{Loss} \to \text{Backward} \to \text{Optimizer Step} \to \text{Monitor Update} \to \text{Rollback} \to \text{Fallback}.$$
4. **Current-Batch Recomputation Verified:** `test_9_alarm_triggered_atomic_rollback` verifies that when an update is rejected, the predictions returned for that batch are computed from the frozen source state, matching the source predictions.

---

## 5. Task F — CPU Regression Suite Execution

The complete regression test suite was executed via:
```bash
python -m pytest tests/ -q
```

### Official Test Results:
- **Total Test Items:** 87
- **Passed:** 85
- **Skipped:** 2 (`tests/test_no_leakage.py::test_dataset_paths_contain_no_test_hospital_references` and `tests/test_no_leakage.py::test_wilds_metadata_split_isolation`, which require raw uncompressed WILDS tarball paths when parquet metadata is present)
- **Failed:** 0
- **Deselected:** 0
- **Execution Time:** 30.25 seconds on CPU
- **Warnings:** 5 (Standard `UndefinedMetricWarning` from scikit-learn in synthetic single-class tests where ROC AUC is computed on synthetic all-zero slices).
