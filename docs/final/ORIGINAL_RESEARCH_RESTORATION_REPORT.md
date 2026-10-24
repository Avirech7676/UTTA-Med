# ORIGINAL RESEARCH RESTORATION & INTEGRITY REPORT

**Official Research Title:**  
*Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*

**Branch:** `research-project-2` | **Audit Date:** October 2026  
**Auditor / Lead Engineer:** Senior Computational Pathology Researcher, Statistical Auditor & Reproducibility Specialist  
**Deliverable Type:** Comprehensive Restoration & Integrity Audit Report

---

## 1. RESTORATION EXECUTIVE SUMMARY

This report documents the comprehensive restoration of UTTA-Med Project 2 back to its **authoritative approved research direction** as established by the original research manuscript ([`UTTA_MED_FINAL_COMPACT_PAPER.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.tex) / [`UTTA_MED_PAPER_DRAFT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_PAPER_DRAFT.pdf)).

Every architectural component, dataset partition, experimental definition, and safeguard protocol has been systematically reconciled against authentic source files. No replacement architectures have been introduced. Useful bug fixes and computational optimizations have been preserved. Unrelated research programs remain safely quarantined in external archives.

---

## 2. AUDIT OF ORIGINAL VS. CURRENT STATE

### 2.1 What Was Correct in the Existing Implementation
1. **Official Research Title:** Permanently preserved across all LaTeX sources, markdown reports, and codebase docstrings.
2. **Classification Architecture:** ResNet-18 ($11{,}449{,}301$ parameters) with a single-scalar logit output, global average pooling, dropout ($p=0.5$), and binary cross-entropy loss in [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py).
3. **Certified Checkpoints:** All 5 source ResNet-18 checkpoints (`camelyon17_resnet18_source_s*.pt`) and the MIDOG Faster R-CNN checkpoint (`midog_faster_rcnn_source.pt`) match canonical cryptographic SHA-256 hashes recorded in [`PRESERVED_CORE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/PRESERVED_CORE_FILES.csv).
4. **Trainable Scope & Normalization Policy:** Strictly restricted to 40 BatchNorm2d affine parameters ($\gamma, \beta$) with running statistics frozen (`bn_freeze_stats`) in [`src/tta/bn.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/bn.py).
5. **MC-Dropout Formulation:** $N=20$ stochastic forward passes with head dropout ($p=0.5$) active at inference time.
6. **MIDOG Detection Protocol:** Faster R-CNN with ResNet-50 FPN backbone on Scanners 1, 2 (source) and Scanner 3 (target domain shift across 50 cases and 1,100 stratified patches).
7. **Original Detection Arms A–E:** Preserved in [`09_VERIFIED_DETECTION_BASELINES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/09_VERIFIED_DETECTION_BASELINES.csv) and historical evaluation scripts.

---

### 2.2 What Had Deviated and What Was Restored

| Component | Nature of Deviation | Root Cause | Action Taken & Restoration |
|:---|:---|:---|:---|
| **Center 2 Patient Accounting** | Displayed 10 slide IDs and 10 distinct patient IDs while total row stated 9 patients | Typographical transcription errors in patient IDs (typed 43 instead of 44; 48 instead of 46 on slide 26; 49 instead of 48) | Directly verified raw Parquet metadata (`test-*.parquet`). Proved Center 2 contains **10 WSIs from exactly 9 distinct patients**: $\{40, 41, 42, 44, 45, 46, 48, 51, 52\}$. Patient 46 contributes both Slide 25 (node 3) and Slide 26 (node 4). Canonical cohort table updated. |
| **Clinical Staging Claims** | Assigned AJCC staging labels (ITC, pN0(i+), pN1mi, pN1a) to target slides | Inferred staging from patch prevalence heuristics without continuous lesion diameter ground truth | Audited WILDS benchmark schema. Proved WILDS contains only binary patch labels and coordinates, not AJCC lesion diameter measurements. Excised all unsupported clinical staging claims from canonical tables. |
| **Adaptation Harm Denominators** | Ambiguously labeled metric as "Errors / Accepted" while using total cohort $N=85{,}054$ | Conflation of cohort-level error rate ($H/N$) with accepted-update error rate ($H/k$) | Mathematically delineated both quantities from raw arrays: Cohort Harm Rate $H/N = 0.585\% \pm 0.224\%$; Accepted Harm Rate $H/k = 0.630\% \pm 0.241\%$ under unconstrained $\tau=0.85$, and $1.204\%$ under matched-budget $58.10\%$. Net flip benefit: $+1{,}741.8$ flips. |
| **Improved UTTA-Med ($A_3$) Identity** | $A_3$ was mislabeled "CAD-Gate", listed at AUROC $0.9353$, and latency was listed as $136.29$ ms | Conflated 5-seed target benchmark $A_3$ with exploratory validation-only candidate CAD-Gate and copied $A_2$ latency | Disentangled $A_3$ (Feature-Cached MC-Dropout, latency $7.82$ ms, AUROC $0.9351$, F1 $0.8303$ on 5-seed Center 2) from CAD-Gate (Center 1 validation candidate only; never evaluated on Center 2). Clean separation restored. |
| **Coverage Definitions** | Multiple coverage figures ($58.1\%$, $87.95\%$, $92.88\%$) appeared without operational distinction | Interchanged coverage values across different operating constraints | Mapped and documented exact operating points: $58.10\%$ = matched-budget constraint; $87.95\%$ = static source confidence filter; $92.88\%$ = unconstrained online stream admission under $\tau=0.85$. |
| **Statistical Test Values** | Historical typo $t=5.679$ appeared in narrative alongside rounded $t=5.8990$ and unrounded $t=5.2723$ | Typographical transposition during manuscript preparation | Formalized canonical unrounded Student's $t$-test: $t = 5.2723, p = 0.006202$ ($df=4$) in [`STATISTICAL_CORRECTION.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/STATISTICAL_CORRECTION.md). Deprecated historical typo. |
| **Research Readiness Claims** | Declared 100/100 production standard and clinical safety | Confused code regression testing and benchmark reproduction with clinical validation | Formally calibrated readiness into 6 distinct scientific dimensions separating software correctness from prospective clinical trial requirements. |

---

### 2.3 Useful Bug Fixes Retained
1. **Permanent Excision of Validation Shortcut:** In [`scripts/run_corrected_slide_validation.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_corrected_slide_validation.py), the legacy shortcut `np.where(mask, p_mean_val, p_frozen)` was completely excised and replaced with authentic PyTorch backpropagation and Adam optimizer steps across slide groupings. Verified non-zero parameter drift ($\Delta\theta_{L_2} = 0.007026$) is confirmed by automated regression tests in [`tests/test_validation_no_shortcut.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/tests/test_validation_no_shortcut.py).
2. **Episodic Slide Resets:** Verified exact parameter restoration ($\Delta\theta = 0.000000$) and optimizer state clearing across 20 slide transitions in [`SLIDE_RESET_VERIFICATION.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/SLIDE_RESET_VERIFICATION.csv).
3. **Fail-Closed Safeguard Stress Testing:** 100% rollback success rate certified across 5 fault injection suites in [`FAIL_CLOSED_STRESS_TEST_REPORT.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FAIL_CLOSED_STRESS_TEST_REPORT.md).

---

### 2.4 Useful Computational Optimizations Retained
1. **Feature-Cached MC-Dropout:** In [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py), caching backbone features (`forward_features`) and executing 20 stochastic passes exclusively through the classification head accelerates inference from $136.29$ ms/batch to $7.82$ ms/batch ($\mathbf{17.4\times}$ speedup) while preserving output equivalence to full-pass MC-Dropout to $10^{-10}$ numerical precision. Both the original full-pass implementation and the optimized feature-cached implementation are maintained.

---

### 2.5 Unrelated Research Quarantined in Archives
The following unrequested extensions remain strictly archived outside the active research workflow on Git branch `archive/unrequested-research-extensions` and external archive `C:/Users/avina/Documents/UTTA_MED_RESEARCH_ARCHIVE/`:
- DenseNet-121 classification replacement.
- ResNet-50 classification replacement.
- EMA Teacher-Student adaptation framework.
- Unrequested exploratory matrices (C0–C7 classification sweeps and D0–D9 Bio-SafeAdapt extensions).

---

### 2.6 Genuine Scientific Limitations (Documented Without Concealment)
1. **Absence of Whole-Slide Lesion Diameter Metadata:** The Camelyon17-WILDS patch benchmark provides binary patch classifications and coordinates, not continuous lesion diameters or institutional TNM stages.
2. **Target Sample Scope:** Target evaluation is conducted on a single held-out center (Center 2) consisting of 10 slides from 9 patients. 5-seed testing proves optimization consistency across random weight initializations, but does not constitute multi-center patient population generalization.
3. **Clinical Deployment Status:** The software is a research prototype. It is **not** clinically validated and **not** approved for clinical diagnosis.
