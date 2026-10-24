# Stage E1: Prespecified Evaluation Plan and Protocol

**Author:** Independent Scientific Auditor, Senior PyTorch Engineer, and Medical Imaging Reproducibility Researcher  
**Date:** October 2026  
**Status:** Prespecified & Preregistered  
**Scope:** Hospital-1 Comparative Safety and Adaptation Benchmark  

---

## 1. Objective and Experimental Rationale

Stage E1 is a prespecified, preregistered comparative evaluation on Hospital-1 development data. The goal is to determine whether SafeAdapt's architectural safeguards (Stage B atomic parameter bounds and Stage C2 active collapse monitoring) prevent adaptation failure and decision boundary drift under confidence-gated TTA without compromising downstream diagnostic accuracy.

### Strict Scientific Boundaries
- **Hospital 1 ONLY:** Evaluated exclusively on Hospital 1 (Center 1) WSIs `[11, 13, 15, 17, 19]`.
- **Zero Hospital-2 Access:** Center 2 (slides 20–29) remains completely sealed and unaccessed.
- **Frozen Source Model:** ResNet-18 (`camelyon17_resnet18_source_s42_BEST.pt`, SHA-256: `f0b85ec1a897...`).
- **No v1 Modifications:** Frozen v1 artifacts, tables, and scripts remain unaltered.

---

## 2. Four Experimental Arms

All active adaptation arms utilize identical loss formulations (Shannon entropy on selected patches), batch size ($32$), learning rate ($1.0 \times 10^{-5}$), Adam optimizer settings, and primary confidence threshold ($\tau_{\text{conf}} = 0.90$).

```
================================================================================
STAGE E1 EXPERIMENTAL ARMS
================================================================================
Arm A: True Frozen Source Baseline
       - Updates: Zero updates executed (immutable reference).
       - Controller: None.
       - Role: Establishes the authoritative diagnostic floor.

Arm B: Unsafeguarded Confidence-Gated TTA
       - Updates: Entropy minimization on patches with confidence >= 0.90.
       - Parameter Updates: BN affine parameters only.
       - Controller: Disabled (no parameter bounds, no non-finite rollback, no collapse monitoring).
       - Role: Isolates standard TTA behavior and exposes vulnerability to silent drift.

Arm C: Stage B Confidence-Gated SafeAdapt (Atomic Parameter Safeguards)
       - Updates: Entropy minimization on patches with confidence >= 0.90.
       - Parameter Updates: BN affine parameters only.
       - Controller: Stage B Atomic Controller active.
       - Safeguards: Non-finite loss/grad rollback; L_inf displacement clipping (|theta_t - theta_0|_inf <= 0.05).
       - Collapse Monitor: Observation mode (passive telemetry only).
       - Role: Evaluates whether parameter-space bounding alone prevents catastrophic divergence.

Arm D: Stage C2 Active SafeAdapt (Dual Parameter & Collapse Safeguards)
       - Updates: Entropy minimization on patches with confidence >= 0.90.
       - Parameter Updates: BN affine parameters only.
       - Controller: Stage B Atomic Controller active.
       - Safeguards: Stage B parameter bounding PLUS active collapse monitoring.
       - Interventions: 2 consecutive suspicious batches -> Atomic Rollback.
                        4 consecutive suspicious batches -> Immutable Source Fallback.
       - Role: Full production candidate evaluating automated behavioral intervention.
================================================================================
```

---

## 3. Slide Ordering and Stress-Testing Protocol

Test-time adaptation is inherently sequential and vulnerable to order-dependent drift. Stage E1 preregisters three distinct WSI processing schedules:

| Schedule | Name | WSI Sequence | Stress-Test Objective |
| :---: | :--- | :--- | :--- |
| **Order 1** | Natural Clinical Stream | `[11, 13, 15, 17, 19]` | Simulates standard chronological acquisition across clinical cases. |
| **Order 2** | Reverse Acquisition | `[19, 17, 15, 13, 11]` | Tests sensitivity to reverse slide presentation order. |
| **Order 3** | Tumor-Burden Ascending | `[15, 19, 17, 11, 13]` | **Adversarial Stress Test:** Starts with pure normal tissue (WSI 15, 0% tumor) followed by low tumor burden (WSI 19, 6% tumor), testing whether early uninformative batches cause boundary collapse before high-tumor slides are encountered. |

*Slide Reset Protocol:* Under all schedules, model parameters, optimizer momentum, and monitor sliding windows are reset to the immutable source state at every WSI boundary.

---

## 4. Prespecified Endpoints and Evaluation Metrics

### Primary Statistical Units
The 5 assessment WSIs `[11, 13, 15, 17, 19]` are treated as the primary independent units. Aggregations shall report mean, standard deviation, and worst-WSI performance across slides.

### Diagnostic & Calibration Metrics
For each WSI and pooled:
1. **AUROC:** Area under the ROC curve.
2. **AUPRC:** Area under the precision-recall curve.
3. **F1 Score:** Macro F1 at default decision threshold ($0.5$).
4. **ECE:** Expected calibration error ($10$ bins).
5. **Brier Score:** Mean squared probability error.
6. **Worst-WSI Degradation ($\Delta_{\text{worst}}$):** Maximum performance drop relative to Frozen Source on any single WSI.

### Safety & Telemetry Metrics
1. **Alarm Frequency:** Total suspicious batches flagged by the collapse monitor.
2. **Rollback Count:** Number of atomic parameter rollbacks executed.
3. **Fallback Count:** Number of slide-level immutable source fallbacks triggered.
4. **Parameter Displacement:** Maximum observed $\|\theta_t - \theta_0\|_\infty$.
5. **Computational Cost:** Total wall-clock time and mean batch latency (ms).

---

## 5. Statistical Reporting and Limitation Disclosures

In accordance with strict auditor standards:
- **Small Sample Caveat:** $N_{\text{WSI}} = 5$ slides represents a small sample size. Confidence intervals and hypothesis tests shall be reported with explicit acknowledgement of limited statistical power.
- **Patch Independence Ban:** Patches within a single WSI share biological and staining correlations; they must never be treated as independent identically distributed (i.i.d.) replicates in significance testing.
- **Preservation of Negative Findings:** Any failure of SafeAdapt to prevent degradation or any unnecessary false-alarm fallbacks shall be reported with full transparency.

---

## 6. Prespecified Decision Criteria for Stage E1 Exit

- **GO for Stage E2 (Hospital-2 Preregistration):**
  If Arm D (Active SafeAdapt) demonstrates non-inferior AUROC/F1 relative to Frozen Source across all slide orders, prevents worst-WSI collapse, and executes with $< 5\%$ false-alarm fallback rate.
- **CONDITIONAL GO:**
  If Arm C (Stage B parameter bounding alone) performs identically to Arm D with lower algorithmic overhead, suggesting active collapse monitoring is redundant in the presence of tight parameter bounds.
- **NO-GO:**
  If any SafeAdapt arm causes catastrophic performance degradation ($\Delta_{\text{AUROC}} < -0.05$ or $\Delta_{\text{F1}} < -0.10$) or exhibits instability under slide-order stress tests.
