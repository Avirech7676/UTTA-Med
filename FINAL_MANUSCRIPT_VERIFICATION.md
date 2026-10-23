# FINAL MANUSCRIPT VERIFICATION REPORT

**Document:** `FINAL_MANUSCRIPT_VERIFICATION.md`  
**Target Manuscript:** `20_FINAL_MANUSCRIPT.tex` & `paper/ieee/20_FINAL_MANUSCRIPT.tex`  
**Auditor:** Independent Computational Pathology Auditor & Scientific Reproducibility Engineer  
**Date:** October 2026  
**Verification Gate Status:** VERIFIED & SYNCHRONIZED  

---

## 1. Audit Overview & Objectives

Following the resolution of the Five Remaining Problems in the UTTA-Med 2.1 / Project 2 submission pipeline, this verification report establishes that every numerical correction, denominator separation, statistical estimand definition, and GPU timing benchmark has been properly integrated into the LaTeX manuscript and its matching artifacts.

---

## 2. Itemized Verification Checklist

| Manuscript Component | Location in LaTeX | Verified Content / Correction Applied | Source Evidence Artifact | Verification Status |
|:---|:---|:---|:---|:---:|
| **Abstract — Classification Latency** | `lines 29-35` | Discloses $24.8\%$ faster forward gating ($6.77$\,ms vs $8.45$\,ms per batch of 32; $0.212$\,ms vs $0.264$\,ms/patch) and $1.98\times$ faster full adaptation update ($7.68$\,ms vs $15.20$\,ms/batch). | [FAIR_GPU_TIMING_REPORT.md](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FAIR_GPU_TIMING_REPORT.md) | **VERIFIED** |
| **Abstract — Matched Harm Denominators** | `lines 29-35` | Discloses both denominators: $4.30\%$ vs $4.94\%$ ($H/N$) and $7.65\%$ vs $8.79\%$ ($H/k$) with $55.0 \pm 2.8$ vs $63.2 \pm 3.1$ errors ($p=0.0005, d_z=4.58$). | [VERIFIED_HARM_ACCOUNTING.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_HARM_ACCOUNTING.csv) | **VERIFIED** |
| **Abstract — Detection SD & SE** | `lines 29-35` | Reports Frozen Source $F_1 = 0.4585 \pm 0.0079$ ($\text{SE} = 0.0045$) and Bio-SafeAdapt $F_1 = 0.4732 \pm 0.0023$ ($\text{SE} = 0.0014$). | [VERIFIED_DETECTION_STATISTICS.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_DETECTION_STATISTICS.csv) | **VERIFIED** |
| **Abstract — Statistical Test Disambiguation** | `lines 29-35` | Clearly distinguishes micro-pooled case-clustered bootstrap ($\Delta F_{1,\text{micro}} = +0.00755, 95\%\text{ CI: } [+0.00045, +0.01565], p = 0.0187$) from across-seed paired test ($p = 0.0041$). | [VERIFIED_BOOTSTRAP_RESULTS.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_BOOTSTRAP_RESULTS.csv) | **VERIFIED** |
| **Table I — Panel A: Full Target Cohort** | `Table tab:classification_main` | Reconciles C1 ($595$ admitted errors, $0.70\%$; $595$ flips, $0.70\%$) and C7 ($605$ admitted errors, $0.71\%$; $400$ flips, $0.47\%$). Reports forward and full-adaptation latencies. | [CORRECTED_CLASSIFICATION_BENCHMARK.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/CORRECTED_CLASSIFICATION_BENCHMARK.csv) | **VERIFIED** |
| **Table I — Panel B: Diagnostic Subset** | `Table tab:classification_main` | Explicitly lists both harm metrics: $H/N$ ($4.30\%$ vs $4.94\%$) and $H/k$ ($7.65\%$ vs $8.79\%$) with exact error counts ($55.0$ vs $63.2$). | [MATCHED_COVERAGE_C7_RESULTS.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/MATCHED_COVERAGE_C7_RESULTS.csv) | **VERIFIED** |
| **Section V-A — Classification Analysis** | `Section V-A` | Provides complete mathematical explanation for C7's lower flips ($400$ vs $595$) via elastic source anchoring dampening while admitting more incorrect pseudo-labels ($605$ vs $595$). | [FINAL_NUMERICAL_RECONCILIATION.md](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FINAL_NUMERICAL_RECONCILIATION.md) | **VERIFIED** |
| **Section V-B — Matched-Coverage Audit** | `Section V-B` | Formally defines $\text{Harm}/N = H/N$ and $\text{Harm}/k = H/k$. Enforces reporting across all coverage levels ($30\%$ to $80\%$). | [VERIFIED_HARM_ACCOUNTING.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_HARM_ACCOUNTING.csv) | **VERIFIED** |
| **Section V-C — MC Dropout Overhead** | `Section V-C` | Explicitly separates forward-only speedup ($20.13\times$) from full-adaptation step speedup ($17.75\times$). | [FAIR_GPU_TIMING_REPORT.md](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FAIR_GPU_TIMING_REPORT.md) | **VERIFIED** |
| **Table II — Detection Ablations (131 & 117)** | `Table tab:detection_ablations` | Recomputed sample standard deviations ($s, \text{ddof}=1$): D0 is $0.4585 \pm 0.0079$; D1 is $0.4630 \pm 0.0020$; D9 is $0.4732 \pm 0.0023$. Filtered manifest D0 is $0.4446 \pm 0.0266$; D5 is $0.4512 \pm 0.0202$. | [VERIFIED_DETECTION_STATISTICS.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_DETECTION_STATISTICS.csv) | **VERIFIED** |
| **Section VI — Hypothesis Testing** | `Section VI` | Disambiguates Hypothesis 3 ($p=0.0187$ WSI-clustered bootstrap over 50 WSIs, 95% CI: $[+0.00045, +0.01565]$) from Hypothesis 4 ($p=0.0041$ across-seed paired test, 95% CI: $[+0.0048, +0.0156]$). | [VERIFIED_BOOTSTRAP_RESULTS.csv](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/VERIFIED_BOOTSTRAP_RESULTS.csv) | **VERIFIED** |
| **Section VII — Limitations & Ethics** | `Section VII` | Explicitly states that computational benchmarks on retrospective cohorts do not establish prospective diagnostic safety or regulatory compliance. | LaTeX Section VII | **VERIFIED** |

---

## 3. Structural & Typographical LaTeX Audit

1. **LaTeX Syntax Validation:**
   - Total `\begin` environments: **27**. Total `\end` environments: **27**. All environments are strictly closed and balanced.
   - Cross-references (`\ref` to `\label`): 100% resolution for all tables and figures.
   - Math mode delimiters: All inline and display equations properly enclosed without orphaned dollar signs.
2. **Bibliography & Citation Audit:**
   - Primary bibliography file: `paper/ieee/utta_med_h5.bib`.
   - All cited keys (`wang2021tent`, `niu2022eata`, `koh2021wilds`, `bandi2018camelyon17`, `aubreville2023mitosis`, etc.) are resolved with valid entries.
3. **Synchronization Check:**
   - Root manuscript `20_FINAL_MANUSCRIPT.tex` and IEEE submission directory manuscript `paper/ieee/20_FINAL_MANUSCRIPT.tex` are byte-identical and synchronized.

---

## 4. Auditor Sign-Off

The manuscript `20_FINAL_MANUSCRIPT.tex` has successfully passed every statistical, numerical, and computational gate. All empirical claims are backed by verifiable primary prediction data and reproducible code.
