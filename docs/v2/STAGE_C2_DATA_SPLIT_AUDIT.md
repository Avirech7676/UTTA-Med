# Stage C2: Hospital-1 Data Split and Isolation Audit Report

**Document Version:** 1.0-DataAudit  
**Date:** October 10, 2026  
**Auditor / Roles:** Senior Medical Imaging ML Researcher, PyTorch Reliability Engineer, Independent Scientific Auditor  
**Dataset:** Camelyon17-WILDS Whole-Slide Pathology Dataset  
**Partition Scope:** Hospital 1 (Center 1) OOD Validation Split ONLY  
**Target Isolation Mandate:** Hospital 2 (Center 2, Slides 20–29) Strictly Isolated (Zero Access)

---

## 1. Executive Summary & Regulatory Boundary

This document records the data provenance, split boundary invariants, and WSI-level partitioning used for calibrating the UTTA-Med v2 SafeAdapt collapse monitor in Stage C2.

### Critical Scientific Governance Invariants
1. **Zero Access to Hospital 2:** Center 2 (Test split, 85,054 patches across Slides 20–29) is strictly locked. No labels, features, predictions, or images from Hospital 2 are accessed, read, or evaluated during calibration, threshold selection, debugging, or model choices.
2. **Hospital-1-Only Calibration:** All empirical distributions of logit variance, predictive entropy, and source-versus-adapted disagreement are derived solely from Hospital 1 (Center 1, 34,904 patches across Slides 10–19).
3. **Exploratory / Provisional Designation:** Because Hospital 1 contains only $N_{\mathrm{WSI}} = 10$ independent whole-slide images, this calibration study is formally designated **exploratory and provisional**. It is not claimed to provide universal or clinical safety bounds across arbitrary hospitals.

---

## 2. Camelyon17 Dataset Structure & Domain Identifiers

Data files are read directly from local HuggingFace parquet files at `Camelyon17-Data/data/` via `ParquetCamelyon17`. Domain IDs and slide allocations are verified against the physical metadata tables:

| Split Name | Centers / Domains | Total Patches | Total WSIs | Slide IDs | Primary Clinical Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `train` | Centers 0, 3, 4 | 302,436 | 30 | 0–9, 30–49 | Source Model Training (Frozen v1) |
| `id_val` | Centers 0, 3, 4 | 33,560 | 30 | 0–9, 30–49 | In-Distribution Validation |
| `val` (OOD) | **Center 1** | **34,904** | **10** | **10–19** | **Hospital 1 Development / Calibration** |
| `test` (Target) | **Center 2** | **85,054** | **10** | **20–29** | **Hospital 2 Held-Out Evaluation (FROZEN)** |

### Mathematical Disjointness Guarantee
$$\mathrm{WSIs}(\text{train}) \cap \mathrm{WSIs}(\text{val}) = \emptyset, \quad \mathrm{WSIs}(\text{train}) \cap \mathrm{WSIs}(\text{test}) = \emptyset, \quad \mathrm{WSIs}(\text{val}) \cap \mathrm{WSIs}(\text{test}) = \emptyset$$
$$\{0\dots 9, 30\dots 49\} \cap \{10\dots 19\} = \emptyset, \quad \{10\dots 19\} \cap \{20\dots 29\} = \emptyset$$
Mathematical disjointness across centers and whole-slide image boundaries is absolute.

---

## 3. Hospital-1 Whole-Slide Image Inventory

Hospital 1 comprises 10 Whole-Slide Images from Center 1 with substantial natural variation in tumor burden and patch count:

| Slide ID | Center | Total Patches | Tumor Patches | Normal Patches | Tumor Prevalence | Clinical Tumor Category |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **10** | 1 | 2,078 | 174 | 1,904 | 8.37% | Low burden |
| **11** | 1 | 5,371 | 3,226 | 2,145 | 60.06% | High burden |
| **12** | 1 | 1,705 | 157 | 1,548 | 9.21% | Low burden |
| **13** | 1 | 10,404 | 8,776 | 1,628 | 84.35% | Very high burden |
| **14** | 1 | 888 | 32 | 856 | 3.60% | Very low burden |
| **15** | 1 | 2,966 | 17 | 2,949 | 0.57% | Near-zero (Benign-dominant) |
| **16** | 1 | 4,971 | 4,094 | 877 | 82.36% | High burden |
| **17** | 1 | 2,659 | 629 | 2,030 | 23.66% | Moderate burden |
| **18** | 1 | 1,275 | 178 | 1,097 | 13.96% | Moderate burden |
| **19** | 1 | 2,587 | 169 | 2,418 | 6.53% | Low burden |
| **Total** | **1** | **34,904** | **17,452** | **17,452** | **50.00%** | **Aggregate Balanced** |

---

## 4. WSI-Disjoint Stage C2 Experimental Partitioning

To avoid calibration overfitting on Hospital 1, the 10 WSIs are partitioned into two strictly WSI-disjoint subsets, stratified across the range of tumor burdens:

### 4.1 Calibration Partition (5 WSIs)
Used exclusively in **Step 3** to record observation-mode empirical distributions and calibrate candidate alarm thresholds.
- **Slides Selected:** `[10, 12, 14, 16, 18]`
- **Total Patches Available:** 10,917
- **Composition:**
  - Slide 14: 3.6% tumor (very low burden)
  - Slide 10: 8.4% tumor (low burden)
  - Slide 12: 9.2% tumor (low burden)
  - Slide 18: 14.0% tumor (moderate burden)
  - Slide 16: 82.4% tumor (high burden)
- **Role:** Empirical logit variance, predictive entropy, and source-disagreement percentile estimation.

### 4.2 Assessment Partition (5 WSIs)
Used in **Step 4** for controlled evaluation of the frozen calibrated controller against alternative regimens.
- **Slides Selected:** `[11, 13, 15, 17, 19]`
- **Total Patches Available:** 23,987
- **Composition:**
  - Slide 15: 0.57% tumor (near-zero / benign-dominant tissue)
  - Slide 19: 6.5% tumor (low burden)
  - Slide 17: 23.7% tumor (moderate burden)
  - Slide 11: 60.1% tumor (high burden)
  - Slide 13: 84.4% tumor (very high burden)
- **Role:** Objective evaluation of AUROC, AUPRC, F1, ECE, Brier score, coverage, and worst-slide performance under frozen thresholds.

### 4.3 Negative Control Slide (Stress Testing)
- **Slide Selected:** `Slide 6` (from Center 0 in-distribution validation split, containing 343 normal patches and exactly 0 tumor patches).
- **Role:** Verifying that a genuinely 100% negative slide experiences a **0.0% false-alarm rate**.

---

## 5. Auditor Verification Checklist

- [x] Hospital 2 data files (`test-*.parquet`) are unopened and untouched.
- [x] No target labels entered the adaptation, gating, or monitoring interfaces.
- [x] Calibration and assessment partitions share **zero** whole-slide images ($\mathrm{WSI}_{\mathrm{cal}} \cap \mathrm{WSI}_{\mathrm{eval}} = \emptyset$).
- [x] Slide 15 (0.57% tumor) and Slide 6 (0.0% tumor) provide realistic clinical benchmarks for negative tissue handling.
- [x] Small-sample limitation ($N_{\mathrm{WSI}} = 10$) is prominently documented and flagged as exploratory.
