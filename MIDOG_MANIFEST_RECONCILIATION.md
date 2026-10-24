# MIDOG Evaluation Manifest Reconciliation

**Project:** UTTA-Med Project 2 — Experimental Provenance Audit  
**Status:** COHORTS STRICTLY SEGREGATED & AUDITED  
**Date:** October 2026  

---

## 1. Overview of the Three MIDOG Experimental Cohorts

To prevent data contamination and conflicting benchmark reporting, the project establishes strict boundaries between three distinct MIDOG Scanner 3 evaluation cohorts:

| Cohort Identifier | Ground-Truth Mitoses | WSI Cases | Evaluated Patches | Matching Radius | Primary Role | Associated Benchmark |
|:---|:---:|:---:|:---:|:---:|:---|:---|
| **Cohort 1: Original Paper Benchmark** | **118 mitoses** | 50 cases | 1,100 patches | $30.0\text{ px}$ | Historical Preservation | Table III in `UTTA_MED_FINAL_COMPACT_PAPER.tex` |
| **Cohort 2: Advanced Manifest A (Unfiltered)** | **131 mitoses** | 50 cases | 1,100 patches | $30.0\text{ px}$ | Advanced Research (D0–D9) | `CORRECTED_DETECTION_BENCHMARK.csv` (Manifest A) |
| **Cohort 3: Advanced Manifest B (Boundary-Filtered)**| **117 mitoses** | 50 cases | 1,100 patches | $30.0\text{ px}$ | Edge Sensitivity Analysis | `CORRECTED_DETECTION_BENCHMARK.csv` (Manifest B) |

---

## 2. Cohort Details & Non-Interchangeability Rules

1. **Cohort 1 (118 Mitoses):**
   * Corresponds to the official historical experiment reported in the original paper (`UTTA_MED_PAPER_DRAFT.pdf`).
   * Results: Frozen Source (Arm A) $F_1 = 0.4774$, Tent (Arm B) $F_1 = 0.4702$, SafeAdapt (Arm E) $F_1 = 0.4774$.
   * **Rule:** Must be preserved without alteration in the compact IEEE paper. Never mix with 131 or 117 mitoses.

2. **Cohort 2 (131 Mitoses):**
   * Canonical full-annotation manifest used for Bio-SafeAdapt development and incremental ablations (D0 through D9).
   * Includes all annotated mitoses within the 1,100 spatially stratified patches without border exclusions.
   * Results: Frozen Source (D0) $F_1 = 0.4585 \pm 0.0079$; Bio-SafeAdapt (D8/D9) $F_1 = 0.4732 \pm 0.0023$ ($+0.0147$ over frozen, $+0.0102$ over SafeAdapt).

3. **Cohort 3 (117 Mitoses):**
   * Boundary-filtered sensitivity manifest excluding annotations within 16 pixels of patch margins to avoid partial mitotic figure truncations.
   * Results: Frozen Source (D0) $F_1 = 0.4446 \pm 0.0266$; Bio-SafeAdapt (D5) $F_1 = 0.4512 \pm 0.0202$ ($+0.0066$ gain).
