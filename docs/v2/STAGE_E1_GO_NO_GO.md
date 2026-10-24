# Stage E1: Go / No-Go Decision and Stage E2 Target Preregistration

**Auditor & Lead Investigator:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Decision Status:** **FORMAL GO** for Stage E2 Target Evaluation  
**Target Domain:** Camelyon17-WILDS Hospital 2 (Center 2, Test WSIs `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]`)  
**Source Checkpoint:** `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`  
**Checkpoint Hash:** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` (Verified SHA-256)

---

## 1. Executive Determination

Following the successful execution and auditing of Stage E1 on Hospital 1 development whole-slide images, the independent audit confirms that the UTTA-Med v2 SafeAdapt framework satisfies all preregistered reproducibility, stability, and protocol integrity criteria.

The formal decision is **GO** to unseal and evaluate on the out-of-distribution Hospital 2 target cohort under strictly frozen experimental protocols.

---

## 2. Comprehensive Criteria Verification Checklist

| Criterion | Requirement | Observed Stage E1 Evidence | Status |
| :--- | :--- | :--- | :---: |
| **1. Protocol Alignment** | Prediction timing must reflect authentic online streaming. | Pre-adaptation timing ($p_{\text{eval}} = \sigma(f_{\theta_t}(x_t))$ before step $\theta_t \to \theta_{t+1}$) strictly implemented, verified, and audited across all arms. | **MET** |
| **2. Slide Order Invariance** | WSI-boundary resets must guarantee deterministic invariance to slide order. | Experiment 2 evaluated reverse and tumor-ascending arrival sequences. Max absolute probability difference with natural order was **0.00e+00** (bit-exact identical across all arms). | **MET** |
| **3. Stress-Test Pathology** | Continuous stream behavior without resets must be documented. | Experiment 3 documented monotonic parameter drift ($L_\infty = 9.94 \times 10^{-5}$) and a $+8.1\%$ calibration degradation ($\text{ECE} = 0.0850$ vs $0.0786$) when resets were omitted. Proves need for boundary resets. | **MET** |
| **4. Metric Reconciliation** | Source metrics must match Stage C2.5 canonical values. | True Frozen Source Brier score reproduced at canonical **0.082867**; macro valid AUROC reproduced at **0.9706**. Handled single-class WSI 15 cleanly. | **MET** |
| **5. Safeguard Integrity** | Safeguards must not trigger false alarms on benign data. | Arm C and Arm D achieved zero false alarms, zero bound violations, and zero false fallbacks on Hospital 1. | **MET** |
| **6. Computational Feasibility** | Evaluation latency must remain manageable on CPU. | SafeAdapt dual-safeguard overhead quantified at $475\text{ms/patch}$ ($152\text{s}$ total on 320 patches). Completely feasible for target evaluation. | **MET** |
| **7. Software Reliability** | Zero regressions across test suite. | Complete pytest suite passed: **99 passed, 2 skipped, 0 failed** in 40.99s. | **MET** |
| **8. Isolation Integrity** | Zero contamination of Hospital 2 data or labels. | Confirmed zero file opens, zero dataloaders, zero inferences on Center 2 data throughout Stages B through E1. | **MET** |

---

## 3. Preregistration for Stage E2 Target Evaluation

To preserve the scientific validity of the final evaluation on Hospital 2 (the true unseen out-of-distribution target domain), the protocol for Stage E2 is formally frozen prior to any data access.

### A. Evaluated Cohort and Data Access
- **Dataset:** Camelyon17-WILDS Hospital 2 (Center 2).
- **Slides:** WSIs `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` (10 total whole-slide images).
- **Patch Resolution:** $224 \times 224$, standard eval transform (Resize 224, ImageNet normalization).
- **Label Handling:** Target labels are loaded **strictly for offline metric computation** after all model predictions for that slide have been completed and saved. Labels are never passed to the model, optimizer, or collapse monitor.

### B. Preregistered Experimental Arms
1. **Arm A: True Frozen Source Baseline** (Zero updates, frozen source weights).
2. **Arm B: Unsafeguarded Confidence TTA** (Confidence threshold $\tau=0.90$, $\text{lr}=10^{-5}$, entropy minimization).
3. **Arm C: Stage B SafeAdapt** (Confidence TTA + finite checks + $L_\infty \le 0.05$ bound + rollback).
4. **Arm D: Stage C2 Active SafeAdapt** (Stage B + rolling collapse monitor + automatic source fallback).

### C. Execution Invariants
- **Prediction Timing:** Strict **pre-adaptation** timing ($p_{\text{eval}}$ recorded before $\theta$ is updated).
- **Boundary Reset:** Strict atomic reset of model ($\theta_0$), optimizer, and monitor state at every WSI boundary.
- **Primary Metrics:**
  - Macro AUROC (computed over valid dual-class slides).
  - Macro AUPRC.
  - Overall / Pooled F1 Score (threshold 0.5).
  - Expected Calibration Error (ECE, 10 bins).
  - Brier Score.
  - Telemetry: Count of accepted updates, rejected updates, bound clamps, collapse alarms, and fallbacks.

### D. Hypothesis and Scientific Criteria
- **Primary Hypothesis:** Under out-of-distribution domain shift (Hospital 2), unconstrained TTA (Arm B) risks parameter divergence or class collapse, whereas Stage C2 Active SafeAdapt (Arm D) will prevent catastrophic performance degradation through displacement containment and fallback mechanisms.
- **Reporting Commitment:** All metrics, whether favorable or unfavorable, will be reported transparently without cherry-picking or post-hoc threshold tuning.

---

## 4. Final Sign-off

Stage E1 is officially completed and signed off. Permission is granted to proceed to Stage E2.
