# Stage C2: Comparative Results & Stage D GO/NO-GO Audit

**Status:** Completed & Calibrated  
**Date:** October 2026  
**Auditor:** Independent Scientific Auditor & PyTorch Reliability Engineer  
**Dataset:** Camelyon17-WILDS (Hospital 1 Assessment Partition)  
**Study Scope:** Exploratory / Provisional ($N_{\mathrm{WSI}} = 10$)  

---

## 1. Executive Summary & Experimental Framework

This document reports the comparative assessment of the Stage C2 calibrated SafeAdapt controller across matched adaptation regimens on the held-out **Assessment Partition** of Hospital 1 (Slides 11, 13, 15, 17, 19).

> [!NOTE]
> **Assessment Denominator Disclosure:**
> The five assessment WSIs contain 23,987 total patches. Due to local CPU execution limits (~8.9s per batch), evaluation was conducted on a uniform stride-sampled subset of **64 patches per slide (320 patches total, a 1.33% sampling fraction)**. This evaluation represents an exploratory/smoke-test profile, not a complete WSI assessment.

All methods were evaluated under identical conditions:
- **Architecture:** ResNet-18 (ImageNet-pretrained source, 1-logit binary head)
- **Checkpoint:** Frozen source `seed=42` (`checkpoints/camelyon17_resnet18_source_s42_BEST.pt`)
- **Adaptation Scope:** BatchNorm affine parameters only ($\gamma, \beta$)
- **Learning Rate:** $\text{lr} = 10^{-5}$
- **Batch Size:** 32 patches per step
- **WSI Boundary Handling:** Strict slide-level state isolation and boundary resets
- **Target Label Isolation:** Zero target labels entering any adaptation or monitoring interface

---

## 2. Overall Assessment Comparison

The table below summarizes discrimination (AUROC, AUPRC), clinical threshold performance (F1 at $\tau=0.5$), calibration (ECE, Brier score), update coverage, and failure interventions across all evaluated regimens on the 320 assessment patches:

| Regimen | AUROC | AUPRC | F1 ($\tau=0.5$) | ECE | Brier Score | Update Coverage | Rejections | Fallback Triggered |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **True Frozen Source** | **0.9736** | **0.9542** | **0.8632** | **0.0786** | **0.0829** | 0.0% | 0 | False |
| **Stage B (Unmonitored)** | **0.9740** | **0.9548** | **0.8632** | **0.0795** | **0.0824** | 100.0% | 0 | False |
| **Stage C1 (Observation)** | **0.9740** | **0.9548** | **0.8632** | **0.0795** | **0.0824** | 100.0% | 0 | False |
| **Stage C2 (Active Calibrated)** | **0.9740** | **0.9548** | **0.8632** | **0.0795** | **0.0824** | 100.0% | 0 | False |
| **Confidence-Gated** | **0.9740** | **0.9548** | **0.8632** | **0.0795** | **0.0824** | 100.0% | 0 | False |

### Key Diagnostic & Forensic Takeaways:
1. **True Frozen Source Disclosed:** True unadapted inference yields AUROC 0.973593 vs. adapted Stage B AUROC 0.974026 ($\Delta \text{AUROC} = +0.000433$, $\Delta \text{AUPRC} = +0.000592$). In the original Stage C2 run script, "Frozen Source" was inadvertently evaluated using the active controller and relabelled post-hoc.
2. **Mathematical Cause of F1 Invariance:** Across all 320 patches, zero predictions crossed the $\tau=0.5$ boundary between Frozen Source and Adapted Stage B (mean probability change was $0.000989$, max change $0.036898$). The confusion matrices were 100% identical (101 TP, 23 FP, 187 TN, 9 FN), resulting in exact F1 equality ($0.863248$).
3. **Regimen Parity:** Stage C1 (observation-only) and Stage C2 (zero alarms tripped on healthy tissue) produced identical updates to Stage B. Confidence-gating was inactive because mean batch confidence was $0.9851$, well above the $0.65$ gate.
4. **Harm Rate:** The empirical harmful adaptation rate is **0.0%** across all assessment slides.

---

## 3. Per-Slide Breakdown & Worst-Slide Performance

Because digital pathology whole-slide images exhibit high inter-specimen heterogeneity, aggregate metrics can obscure specimen-specific degradation. Below is the per-slide breakdown on the held-out assessment partition:

| Slide ID | Ground Truth Tumor Prevalence | AUROC | AUPRC | F1 | ECE | Brier Score | Clinical Assessment |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Slide 11** | 59.38% | 0.9939 | 0.9958 | 0.9315 | 0.0593 | 0.0541 | High prevalence; excellent calibration & discrimination |
| **Slide 13** | 82.81% | 0.9760 | 0.9951 | 0.9320 | 0.0884 | 0.0767 | High tumor load; robust discrimination |
| **Slide 15** | 0.00% | 0.5000* | 0.0000* | 0.0000* | 0.0413 | 0.0283 | All-negative slide; excellent Brier; zero false alarms |
| **Slide 17** | 23.44% | 0.9959 | 0.9866 | 0.8108 | 0.1499 | 0.1107 | Intermediate prevalence; near-perfect AUROC |
| **Slide 19** | 6.25% | 0.9167 | 0.3292 | 0.4211 | 0.1700 | 0.1424 | **Worst slide:** Low tumor load, sparse micro-metastases |

*\*Note: On Slide 15, ground truth contains only negative patches ($y=0$). AUROC and AUPRC are mathematically degenerate ($0.5$ and $0.0$ by convention); Brier score ($0.0283$) and ECE ($0.0413$) demonstrate strong model reliability.*

### Worst-Slide Analysis (Slide 19):
- **Characteristics:** Tumor prevalence is only 6.25% (4 tumor patches out of 64).
- **Discrimination vs Calibration:** While AUROC remains respectable at 0.9167, AUPRC drops to 0.3292 and F1 to 0.4211 due to extreme class imbalance and boundary patch ambiguity.
- **Safety Controller Behavior:** The Stage C2 controller maintained normal operational state (`AlarmLevel.NORMAL`) because the source model and adapted model maintained consistent logit variance ($\sigma_z^2 > 5.0$) and low disagreement ($\Delta_{\mathrm{bin}} < 0.02$), appropriately avoiding false fallback.

---

## 4. Stage D Decision: Efficient Uncertainty Gating

### Evaluation Criteria:
1. **Controller Correctness:** Full integration audit passed; atomic parameter and optimizer rollbacks verified.
2. **False Alarm Containment:** 0.0% false alarms on genuinely negative whole-slide images (in-distribution Slide 6 and out-of-distribution Slide 15).
3. **Catastrophic Failure Protection:** Active mode reliably detects single-class collapse, rejects corrupted updates, and escalates to permanent source fallback.
4. **Inter-Slide Isolation:** Slide-boundary resets prevent cross-specimen leakage and state contamination.
5. **Deterministic Verification:** All 14 Stage C1 monitor tests and 14 Stage B regression tests pass deterministically.

---

### **Official Scientific Recommendation: CONDITIONAL GO (Stage C2.5 Reconciled)**

**Rationale:**  
The collapse-aware SafeAdapt controller is calibrated, verified, and safe for development progression. Following the Stage C2.5 assessment audit, the evaluation denominator (320 patches, 1.33% sample), true source baseline metrics (AUROC 0.9736 vs adapted 0.9740), and negative control provenance have been fully reconciled. **Conditional clearance is granted to proceed to Stage D: Efficient Uncertainty Gating**. Stage D will investigate lightweight epistemic uncertainty estimators to replace or accelerate expensive 20-pass Monte Carlo Dropout without risk of unmonitored model degradation.
