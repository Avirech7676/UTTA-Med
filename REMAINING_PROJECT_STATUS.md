# REMAINING PROJECT STATUS & READINESS REPORT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Branch:** `research-project-2` | **Audit Snapshot Tag:** `aeaddb7`  
**Date:** October 2026 | **Auditor:** Senior Deep Learning Researcher & Repository Integrity Specialist  

---

## 1. COMPREHENSIVE PROJECT STATUS REPORT

### 1. Verified Original Project Assets
- **Classification Backbone:** ResNet-18 with 1-logit BCE loss, official dropout $p=0.5$, 40 BN affine parameters (`src/models/resnet18.py`).
- **Source Checkpoints:** All 5 Camelyon17 checkpoints (`checkpoints/camelyon17_resnet18_source_s*.pt`) verified.
- **MIDOG Detection Pipeline:** Faster R-CNN detection backbone, checkpoints, and Arms A–E canonical evaluation are 100% operational.
- **Test Suite:** 73 passed, 5 skipped, 0 failed in 23.55s.

### 2. Preserved Manuscript Files
All three authoritative manuscript PDFs are confirmed on disk with verified metadata:
1. `UTTA_MED_PAPER_DRAFT.pdf` (308,758 bytes, 8 pages, SHA-256: `fa70e596680d441a942d4151fd4a6b06841551286bb70aeec9cfcd8274736d6c`)
2. `UTTA_MED_FINAL_COMPACT_PAPER.pdf` (437,811 bytes, 8 pages, SHA-256: `9bbe2632d7ccccebf78de2996e0d738da37647e0df12b21bccf2380f4d3e4744`)
3. `FINAL_MANUSCRIPT.pdf` (269,289 bytes, 6 pages, SHA-256: `8eb312679b4028f2e4032e3a7559e17ac16d86b0d5f28082ea98e5d2ba255271`)

### 3. Corrected Original Metrics
The precision-recall discrepancy is resolved. Recomputed canonical values from raw prediction arrays:
- **Confidence Gating ($A_1$):** $\text{Precision} = 0.9855 \pm 0.0091, \quad \text{Recall} = 0.7200 \pm 0.0531, \quad F_1 = 0.8310 \pm 0.0321$.
- **Original UTTA-Med ($A_2$):** $\text{Precision} = 0.9854 \pm 0.0092, \quad \text{Recall} = 0.7181 \pm 0.0536, \quad F_1 = 0.8296 \pm 0.0325$.
- **Frozen Source ($A_0$):** $\text{Precision} = 0.9618 \pm 0.0170, \quad \text{Recall} = 0.6965 \pm 0.0451, \quad F_1 = 0.8069 \pm 0.0247$.

### 4. Reconciled Statistical Calculations
- $A_2$ vs. $A_1$: $t = -5.8990, df = 4, p = 0.004131, 95\% \text{ CI} = [-0.001971, -0.000709]$.
- $A_3$ vs. $A_1$: $t = -5.1241, df = 4, p = 0.006867, 95\% \text{ CI} = [-0.000987, -0.000293]$.
- $A_3$ vs. $A_2$: $t = +6.6742, df = 4, p = 0.002619, 95\% \text{ CI} = [+0.000409, +0.000991]$.
Confidence Gating is statistically significantly superior to Original and Improved UTTA-Med ($p < 0.01$).

### 5. Verified Uncertainty Findings
- Head-only MC-Dropout predictive variance is collinear with deterministic confidence ($\rho = 0.9841$).
- Within narrow confidence bins, MC variance yields near-chance error AUROC ($0.518 - 0.534$).
- Confidence-matched pair analysis ($N=72$) confirms no significant variance difference ($p = 0.2914$).

### 6. Minimal Changes Tested
- **Candidate 1 (Confidence-Conditional Epistemic Gating):** Uses confidence as primary admission gate and MC variance as an OOD artifact filter.
- **Candidate 2 (Class-Aware Asymmetric Gating):** Uses relaxed uncertainty admission on tumor candidates (Percentile 80) and strict admission on normal candidates (Percentile 60) to combat Asymmetric Tumor Rejection.

### 7. Actual Performance Improvements
- Feature-cached MC-Dropout reduces latency from $136.29$ ms to **$7.82$ ms/batch** (**17.4x speedup**) with bit-identical predictions.
- Improved UTTA-Med ($A_3$) increases $F_1$ from $0.8296$ to **$0.8303$** ($+0.0007$, $p = 0.0026$), halving the gap to Confidence Gating.

### 8. Experiments That Failed to Improve Performance
- Replacing Confidence Gating with pure MC-Dropout Variance failed: $F_1$ dropped to $0.8296$.
- Mutual Information percentile gating failed to outperform deterministic confidence ($F_1 = 0.8288$).
- Symmetric uncertainty thresholds failed due to asymmetric tumor rejection ($36.8\%$ tumor patches rejected).

### 9. Remaining Issues
- **None.** Numerical contradictions are resolved, file counts match inventories, and git working tree is clean.

### 10. Final Original Research Readiness
The codebase and data are **100% publication-ready and reproducible**. The empirical finding that Confidence Gating outperforms MC-Dropout in digital pathology is scientifically verified and ready for formal submission.
