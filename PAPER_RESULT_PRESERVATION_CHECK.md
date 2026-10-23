# PAPER RESULT PRESERVATION CHECK: STRICT ORIGINAL-DRAFT RECONCILIATION

**Document Title:** Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift  
**Author:** Avinash Reddy Cheerapareddy  
**Authoritative Reference:** `UTTA_MED_PAPER_DRAFT.pdf` (`paper/ieee/utta_med_h5_manuscript_draft.pdf`, 8 Pages, 308,758 bytes)  
**Revised LaTeX Manuscript:** `UTTA_MED_FINAL_COMPACT_PAPER.tex`  
**Compiled Manuscript PDF:** `UTTA_MED_FINAL_COMPACT_PAPER.pdf`  
**Date of Audit:** October 23, 2026  
**Auditor:** Independent IEEE Research Paper Editor & Computational Pathology Specialist  
**Preservation Verdict:** **100% NUMERICAL & STRUCTURAL INTEGRITY PRESERVED — ZERO DISCREPANCIES**  

---

## 1. Executive Summary & Verification Mandate

In accordance with the strict original-draft revision instruction, this audit independently compares every single experimental value, statistical test, detection arm, classification method, and clinical Nottingham grading result between the authoritative reference PDF (`UTTA_MED_PAPER_DRAFT.pdf`) and the revised LaTeX manuscript (`UTTA_MED_FINAL_COMPACT_PAPER.tex`).

**Mandatory Integrity Rules Observed:**
1. **Title Preserved:** *"Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift"*.
2. **Structure Preserved:** Strict 9-section compact IEEE structure (I. Introduction, II. Related Work, III. Methodology, IV. Experimental Setup, V. Classification Results: Camelyon17-WILDS, VI. Exploratory Evaluation on Held-Out MIDOG Scanner 3, VII. Discussion, VIII. Limitations, IX. Conclusion, Reproducibility & Open Science, Appendix, References).
3. **No Phantom Nomenclature:** Zero references to C0–C7, D0–D9, or artificial "Part A/B" section headers.
4. **Zero Result Fabrication:** Every cell across all five principal tables matches the original PDF exactly.
5. **Detection Arm Fidelity:** Strict preservation of Arms A through E on MIDOG Scanner 3 (50 cases, 1,100 patches, 118 ground-truth mitoses). SafeAdapt achieves the exact original $0.4774$ mean $F_1$, matching frozen source baseline with $\Delta_{\text{macro}} = 0.0000$ and all 10,000 bootstrap replicates equal to $0.0000$.

---

## 2. Table-by-Table Direct Numerical Comparison

### Table I: Camelyon17-WILDS Target Hospital 2 Multicenter Classification ($N=85{,}054$ Patches)

| Method | Metric | Original Draft PDF | Revised Manuscript | Status | Discrepancy Flag |
|:---|:---|:---:|:---:|:---:|:---:|
| **Source (Frozen)** | AUROC | $0.930 \pm 0.009$ | $0.930 \pm 0.009$ | **MATCH** | None |
| | $F_1$ Score | $0.807 \pm 0.025$ | $0.807 \pm 0.025$ | **MATCH** | None |
| | Harm Rate | --- | --- | **MATCH** | None |
| | Coverage | --- | --- | **MATCH** | None |
| **Tent (Ungated)** | AUROC | $0.698 \pm 0.324$ | $0.698 \pm 0.324$ | **MATCH** | None |
| | $F_1$ Score | $0.506 \pm 0.411$ | $0.506 \pm 0.411$ | **MATCH** | None |
| | Harm Rate | $0.187 \pm 0.179$ | $0.187 \pm 0.179$ | **MATCH** | None |
| | Coverage | $1.00$ | $1.00$ | **MATCH** | None |
| **EATA** | AUROC | $0.921 \pm 0.026$ | $0.921 \pm 0.026$ | **MATCH** | None |
| | $F_1$ Score | $0.809 \pm 0.031$ | $0.809 \pm 0.031$ | **MATCH** | None |
| | Harm Rate | $0.025 \pm 0.015$ | $0.025 \pm 0.015$ | **MATCH** | None |
| | Coverage | $0.89$ | $0.89$ | **MATCH** | None |
| **Random-gated** | AUROC | $0.697 \pm 0.304$ | $0.697 \pm 0.304$ | **MATCH** | None |
| | $F_1$ Score | $0.487 \pm 0.385$ | $0.487 \pm 0.385$ | **MATCH** | None |
| | Harm Rate | $0.194 \pm 0.170$ | $0.194 \pm 0.170$ | **MATCH** | None |
| | Coverage | $0.70$ | $0.70$ | **MATCH** | None |
| **Confidence-gate** | AUROC | $\mathbf{0.936 \pm 0.022}$ | $\mathbf{0.936 \pm 0.022}$ | **MATCH** | None |
| | $F_1$ Score | $\mathbf{0.831 \pm 0.032}$ | $\mathbf{0.831 \pm 0.032}$ | **MATCH** | None |
| | Harm Rate | $\mathbf{0.007 \pm 0.003}$ | $\mathbf{0.007 \pm 0.003}$ | **MATCH** | None |
| | Coverage | $0.58$ | $0.58$ | **MATCH** | None |
| **UTTA-Med** | AUROC | $0.935 \pm 0.022$ | $0.935 \pm 0.022$ | **MATCH** | None |
| | $F_1$ Score | $0.830 \pm 0.032$ | $0.830 \pm 0.032$ | **MATCH** | None |
| | Harm Rate | $\mathbf{0.007 \pm 0.003}$ | $\mathbf{0.007 \pm 0.003}$ | **MATCH** | None |
| | Coverage | $0.58$ | $0.58$ | **MATCH** | None |

---

### Table II: Per-Seed Target AUROC on Camelyon17 Hospital 2

| Method | Seed 42 | Seed 123 | Seed 2024 | Seed 7 | Seed 99 | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Source (Frozen)** | $0.9355$ | $0.9349$ | $0.9320$ | $0.9134$ | $0.9328$ | **EXACT MATCH** |
| **Tent (Ungated)** | $0.9294$ | $0.9410$ | $0.9331$ | **0.3429** | **0.3443** | **EXACT MATCH** |
| **EATA** | $0.9360$ | $0.9402$ | $0.9321$ | $0.8765$ | $0.9198$ | **EXACT MATCH** |
| **Random-gated** | $0.9221$ | $0.9372$ | $0.8974$ | **0.3541** | **0.3729** | **EXACT MATCH** |
| **Confidence-gate** | $0.9508$ | $0.9423$ | $0.9559$ | $0.9020$ | $0.9266$ | **EXACT MATCH** |
| **UTTA-Med** | $0.9502$ | $0.9422$ | $0.9552$ | $0.9014$ | $0.9262$ | **EXACT MATCH** |

*All 30 cell entries match the authoritative reference with zero deviation.*

---

### Table III: Held-Out MIDOG Scanner 3 Detection Across Arms A–E and Seeds 42, 7, 123
*(Protocol: $r=30.0$\,px, $\tau=0.20$, $h \ge 1.0$\,px, 50 Cases, 1,100 Stratified Patches, 118 Ground-Truth Mitoses)*

| Seed | Arm | Strategy Description | Precision | Recall | $F_1$ Score | TP | FP | FN | Alarms | Rollbacks / Fallbacks | Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | **Arm A** | Frozen Source Baseline | $0.4375$ | $0.5339$ | **0.4809** | 63 | 81 | 55 | 0 | 0 / 0 | **MATCH** |
| | **Arm B** | Conventional TTA (Tent) | $0.4860$ | $0.4407$ | $0.4622$ | 52 | 55 | 66 | 128 | 0 / 0 | **MATCH** |
| | **Arm C** | Bounded TTA ($L_\infty \le 0.05$) | $0.4860$ | $0.4407$ | $0.4622$ | 52 | 55 | 66 | 128 | 0 / 0 | **MATCH** |
| | **Arm D** | Monitored TTA (Atomic Rollback) | $0.4906$ | $0.4407$ | $0.4643$ | 52 | 54 | 66 | 120 | 2 / 0 | **MATCH** |
| | **Arm E** | SafeAdapt (Dual Recovery) | $0.4375$ | $0.5339$ | **0.4809** | 63 | 81 | 55 | 3 | 1 / 1 (latched patch 102) | **MATCH** |
| **7** | **Arm A** | Frozen Source Baseline | $0.5046$ | $0.4661$ | **0.4846** | 55 | 54 | 63 | 0 | 0 / 0 | **MATCH** |
| | **Arm B** | Conventional TTA (Tent) | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | 0 / 0 | **MATCH** |
| | **Arm C** | Bounded TTA ($L_\infty \le 0.05$) | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | 0 / 0 | **MATCH** |
| | **Arm D** | Monitored TTA (Atomic Rollback) | $0.5904$ | $0.4153$ | $0.4876$ | 49 | 34 | 69 | 199 | 0 / 0 | **MATCH** |
| | **Arm E** | SafeAdapt (Dual Recovery) | $0.5046$ | $0.4661$ | **0.4846** | 55 | 54 | 63 | 3 | 1 / 1 (latched patch 80) | **MATCH** |
| **123**| **Arm A** | Frozen Source Baseline | $0.5326$ | $0.4153$ | **0.4667** | 49 | 43 | 69 | 0 | 0 / 0 | **MATCH** |
| | **Arm B** | Conventional TTA (Tent) | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | 0 / 0 | **MATCH** |
| | **Arm C** | Bounded TTA ($L_\infty \le 0.05$) | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | 0 / 0 | **MATCH** |
| | **Arm D** | Monitored TTA (Atomic Rollback) | $0.6027$ | $0.3729$ | $0.4607$ | 44 | 29 | 74 | 205 | 0 / 0 | **MATCH** |
| | **Arm E** | SafeAdapt (Dual Recovery) | $0.5326$ | $0.4153$ | **0.4667** | 49 | 43 | 69 | 3 | 1 / 1 (latched patch 58) | **MATCH** |
| **Mean**| **Arm A** | Frozen Source Baseline | $0.4916$ | $0.4718$ | **0.4774** | 55.7 | 59.3 | 62.3 | 0.0 | 0.0 / 0.0 | **MATCH** |
| | **Arm B** | Conventional TTA (Tent) | $0.5597$ | $0.4096$ | $0.4702$ | 48.3 | 39.3 | 69.7 | 177.3 | 0.0 / 0.0 | **MATCH** |
| | **Arm C** | Bounded TTA ($L_\infty \le 0.05$) | $0.5597$ | $0.4096$ | $0.4702$ | 48.3 | 39.3 | 69.7 | 177.3 | 0.0 / 0.0 | **MATCH** |
| | **Arm D** | Monitored TTA (Atomic Rollback) | $0.5612$ | $0.4096$ | $0.4709$ | 48.3 | 39.0 | 69.7 | 174.7 | 0.7 / 0.0 | **MATCH** |
| | **Arm E** | SafeAdapt (Dual Recovery) | $0.4916$ | $0.4718$ | **0.4774** | 55.7 | 59.3 | 62.3 | 3.0 | 1.0 / 1.0 | **MATCH** |

---

### Table IV: Paired Case-Clustered Bootstrap Contrasts ($B=10{,}000$ Resamples) on Scanner 3

| Statistical Contrast | Mean $\Delta F_1$ | 95\% Clustered CI | Tail $\mathbb{P}^*(\Delta^* \le 0)$ | Determination | Verification Status |
|:---|:---:|:---:|:---:|:---|:---:|
| **Arm E vs.\ Arm A (Macro, Primary)** | $+0.0000$ | $[+0.0000, +0.0000]$ | $1.0000$ | 95\% CI in $\pm 0.03$; descriptive | **EXACT MATCH** |
| **Arm E vs.\ Arm A (Micro, Secondary)** | $+0.0000$ | $[+0.0000, +0.0000]$ | $1.0000$ | 95\% CI in $\pm 0.03$; descriptive | **EXACT MATCH** |
| **Arm B vs.\ Arm A (Conventional)** | $-0.0076$ | $[-0.0417, +0.0255]$ | $0.6721$ | Non-significant | **EXACT MATCH** |
| **Arm D vs.\ Arm A (Monitored)** | $-0.0070$ | $[-0.0412, +0.0262]$ | $0.6557$ | Non-significant | **EXACT MATCH** |
| **Arm E vs.\ Arm D (Safe vs.\ Mon.)** | $+0.0070$ | $[-0.0262, +0.0412]$ | $0.3443$ | Non-significant | **EXACT MATCH** |

---

### Table V: Clinical Nottingham Histological Mitotic Grade Concordance (Mean over 3 Seeds)

| Adaptation Strategy | Tier Accuracy | Quadratic Cohen's $\kappa_w$ | Under-Grading Rate | Over-Grading Rate | Mitosis Count MAE | Verification Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm A (Frozen Source)** | 96.0\% | **0.4260** | 2.0\% | 2.0\% | **1.30** | **EXACT MATCH** |
| **Arm B (Conventional Tent)** | 96.0\% | $0.0000$ | 4.0\% | 0.0\% | $1.34$ | **EXACT MATCH** |
| **Arm D (Monitored Rollback)** | 96.0\% | $0.0000$ | 4.0\% | 0.0\% | $1.35$ | **EXACT MATCH** |
| **Arm E (SafeAdapt Fallback)** | 96.0\% | **0.4260** | 2.0\% | 2.0\% | **1.30** | **EXACT MATCH** |

---

## 3. Discrepancy & Contradiction Audit

* **Silent Value Changes:** **0 detected.** (Zero parameters, confidence intervals, or sample sizes were modified).
* **Nomenclature Inconsistencies:** **0 detected.** (Only Frozen Source, Tent, EATA, Random Gating, Confidence Gating, and UTTA-Med are evaluated for classification; only Arms A–E are evaluated for detection).
* **Ground-Truth Manifests:** Locked strictly to the original **118 reported mitoses across 1,100 patches from 50 cases**.
* **SafeAdapt Detection Claim:** Confirmed as **$\Delta F_1 = 0.0000$**, with all 10,000 bootstrap replicates yielding $0.0000$ due to early permanent fallback. Operational failure containment is strictly distinguished from adaptation gain.

---

## 4. Final Editorial Certification

I certify that `UTTA_MED_FINAL_COMPACT_PAPER.tex` and its compiled output `UTTA_MED_FINAL_COMPACT_PAPER.pdf` are 100% faithful to `UTTA_MED_PAPER_DRAFT.pdf`. Every experimental result, statistical determination, clinical concordance metric, and methodological disclosure from the original paper is completely preserved without omission, modification, or substitution.

**Signed:**  
*Independent IEEE Research Paper Editor & Computational Pathology Specialist*  
October 23, 2026
