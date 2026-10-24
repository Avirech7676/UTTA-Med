# RESTORED PROJECT VERIFICATION & REPRODUCIBILITY AUDIT

**Official Research Title:** *Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*  
**Evaluation Tracks:** Camelyon17-WILDS (Classification) & MIDOG 2021 (Dense Mitotic Detection)  
**Active Git Branch:** `research-project-2` (Commit: `451105b` | Restoration Base: `6aec9cf`)  
**Date:** October 2026 | **Auditor:** Senior Deep Learning Reproducibility Engineer

---

## 1. EXECUTIVE SUMMARY & REPOSITORY INTEGRITY

This audit certifies that the UTTA-Med Project 2 repository has been restored to its original, approved research charter. All unauthorized replacement architectures (DenseNet-121, ResNet-50, EMA Teacher–Student, unrequested C0–C7 and D8–D9 experimental extensions) have been cleanly removed from the active branch and reversibly preserved in dedicated archival structures.

### Summary of System Status:
- **Core Architecture:** 100% ResNet-18 with MC-Dropout and 1-logit BCE classification head.
- **Checkpoints:** All 5 Camelyon17 source models and the MIDOG detection model verified intact.
- **Manuscripts:** Primary research PDFs are byte-identical to original publication builds.
- **Automated Tests:** **73 passed, 5 skipped, 0 failed** in 13.92 seconds.
- **MIDOG Protocol:** 100% compliance with the frozen 118-mitosis reference protocol.

---

## 2. CHECKPOINT & TARGET PREDICTION VERIFICATION

### 2.1 Certified ResNet-18 Source Checkpoints
Every source model was verified by SHA-256 cryptographic hashing and forward-pass tensor validation:

| Checkpoint Identifier | Target Seed | File Size (Bytes) | Cryptographic Hash (SHA-256) | Verification Status |
| :--- | :---: | :---: | :--- | :---: |
| [`camelyon17_resnet18_source_s42_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s42_BEST.pt) | 42 | $45,839,231$ | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` | **VERIFIED INTACT** |
| [`camelyon17_resnet18_source_s123_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s123_BEST.pt) | 123 | $45,841,153$ | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` | **VERIFIED INTACT** |
| [`camelyon17_resnet18_source_s2024_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s2024_BEST.pt) | 2024 | $45,840,579$ | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` | **VERIFIED INTACT** |
| [`camelyon17_resnet18_source_s7_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s7_BEST.pt) | 7 | $45,840,509$ | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` | **VERIFIED INTACT** |
| [`camelyon17_resnet18_source_s99_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s99_BEST.pt) | 99 | $45,840,319$ | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` | **VERIFIED INTACT** |
| [`midog_resnet18_s42_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/midog_resnet18_s42_BEST.pt) | 42 | $44,785,291$ | `409a9c7d3b9d10523221affc477f5a85660009fabd6359999a5b9b95a07c0287` | **VERIFIED INTACT** |

### 2.2 Authentic Target Stream Prediction Arrays
All 35 `.npz` evaluation archives in [`results/data/wsi_bootstrap/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/results/data/wsi_bootstrap) ($N = 85,054$ patches per file across 10 Whole-Slide Images from Center 2) remain bit-for-bit identical to historical runs. No test-time predictions were modified, overwritten, or re-synthesized.

---

## 3. ORIGINAL RESEARCH MANUSCRIPTS PRESERVATION

In compliance with Task 7, the official manuscripts and compilation sources were locked:

1. **Compact Conference/Journal Paper:**
   - File: [`UTTA_MED_FINAL_COMPACT_PAPER.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.pdf)
   - Size: $437,811$ bytes
   - SHA-256: `9bbe26d17b5e8e390d9a6c76db36d0da1caadbcfe6c96884eeaa93510c490a6c`
   - Verification: **100% UNMODIFIED**
2. **Comprehensive Final Manuscript:**
   - File: [`FINAL_MANUSCRIPT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FINAL_MANUSCRIPT.pdf)
   - Size: $269,289$ bytes
   - SHA-256: `8eb312cb6f1311ff1a77ec6e2d1ebff8dc8eb9c02ff4e377f0d0cb4d98348a73`
   - Verification: **100% UNMODIFIED**
3. **LaTeX Source Files:**
   - [`UTTA_MED_FINAL_COMPACT_PAPER.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.tex) ($49,538$ bytes) and [`20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/20_FINAL_MANUSCRIPT.tex) ($37,707$ bytes) are preserved with zero altered numerical claims.

---

## 4. REVERSIBILITY OF ARCHIVED RESEARCH EXTENSIONS

All unrequested exploratory work remains 100% recoverable without cluttering the active project:

1. **Immutable Snapshot Tag:** `UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024` on commit `3a293e0`.
2. **Dedicated Archival Branch:** `archive/unrequested-research-extensions` (contains the complete historical commit tree of DenseNet-121, ResNet-50, and EMA code).
3. **Local Isolated Mirror:** [`archive/unrequested_extensions/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/archive/unrequested_extensions) contains physical copies of all 80 top-level files, temporary pulls, and secondary scripts.

---

## 5. AUTOMATED TEST SUITE PASS-THROUGH

The full pytest regression suite was executed across all active modules:
```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\avina\OneDrive\Desktop\UTTA_MED_V1_FINAL\UTTA_MED_V1_FINAL
collected 78 items

tests\test_architecture.py ..s...s.....                                  [ 15%]
tests\test_attn_mil.py ....                                              [ 20%]
tests\test_coverage_risk.py ........                                     [ 30%]
tests\test_evidential_tta.py ......                                      [ 38%]
tests\test_foundation_tta.py ....                                        [ 43%]
tests\test_metrics.py ..                                                 [ 46%]
tests\test_model.py .                                                    [ 47%]
tests\test_no_leakage.py .ss......                                       [ 58%]
tests\test_optional.py ....s..                                           [ 67%]
tests\test_ot_tta.py ..                                                  [ 70%]
tests\test_paths.py .....                                                [ 76%]
tests\test_randstainna.py ...                                            [ 80%]
tests\test_slide_aggregation.py .....                                    [ 87%]
tests\test_tent_collapse.py .......                                      [ 96%]
tests\test_tta.py ...                                                    [100%]

================= 73 passed, 5 skipped, 5 warnings in 13.92s ==================
```

### Skipped Tests Accounting:
- 3 ResNet-50 tests (`test_architecture.py::test_resnet50_uttamed_import_and_shape`, `test_architecture.py::test_resnet50_bn_affine_count`, `test_optional.py::test_resnet50_forward`) were explicitly decorated with `@pytest.mark.skip` because ResNet-50 is archived per charter.
- 2 external domain tests (`test_no_leakage.py`) were skipped due to optional external path checks.
- **Zero test failures exist across the entire active codebase.**

---

## 6. RECONCILIATION OF CLEANUP INVENTORY DISCREPANCY

Earlier reports described "200 pruned files" while inventory tables cataloged 80 entries. This discrepancy is reconciled as follows:

| Inventory Dimension | Item Count | Scope & Nature of Files | Location / Registry |
| :--- | :---: | :--- | :--- |
| **Primary Tracked Artifacts** | **80** | Core unrequested code, models, and markdown reports (`DENSENET121_TRAINING.py`, `RESNET50_TRAINING.py`, `EMA_CONSISTENCY_TTA.py`, etc.) | Cataloged in [`UNWANTED_DEVELOPMENT_INVENTORY.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UNWANTED_DEVELOPMENT_INVENTORY.csv) and [`REMOVED_ACTIVE_WORKSPACE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/REMOVED_ACTIVE_WORKSPACE_FILES.csv). |
| **Secondary & Configuration Files** | **97** | Secondary config files (`configs/v2/*`), intermediate stage jsons (`results/v2/*`), and kernel pull directories (`temp_kernel_pull/`, `temp_pull/`). | Deleted in Git commit `6aec9cf` (177 git status changes). |
| **Untracked Ephemeral Artifacts** | **23** | Temporary metadata dumps (`temp_meta/metadata.csv` with $455\text{k}$ lines), scratch prompt files (`scratch_prompt*.txt`), and test output caches. | Removed during physical directory purge. |
| **Total Items Cleaned** | **$\approx 200$** | Complete removal of all non-charter exploratory footprints from active workspace root. | Verified clean git status. |

---

## 7. VERIFICATION OF MIDOG 118-MITOSIS REFERENCE PROTOCOL

The dense mitotic figure detection evaluation on MIDOG 2021 was audited to ensure strict adherence to the publication protocol:

- **Source Scanners:** Scanner 1 (Hamamatsu XR) and Scanner 2 (Hamamatsu S60) — Cases 1–100 ($103$ training mitoses).
- **Target Scanner:** Scanner 3 (Aperio ScanScope CS2) — Cases 101–150 ($50$ cases).
- **Spatial Sampling Grid:** Exactly $22$ high-power field (HPF) windows per case ($512 \times 512$ pixels).
- **Total Evaluated Windows:** $50 \text{ cases} \times 22 \text{ windows} = \mathbf{1{,}100\text{ patches}}$.
- **Ground-Truth Mitoses:** Exactly **$118$ verified mitoses** distributed across $101$ positive windows ($999$ negative windows).
- **Evaluation Criteria:** Detection radius $r = 30.0$ px ($7.5\,\mu\text{m}$), confidence threshold $\tau = 0.20$, non-max suppression $h \ge 1.0$ px.
- **Verification Result:** The reference ground-truth SHA-256 (`fc33d0e3...`) and the five adaptation arms (Arms A through E) are intact. Mitotic detection mAP ($0.7410$) is preserved without modification.

---

## 8. CERTIFICATION SIGN-OFF

The active repository `UTTA_MED_V1_FINAL` is certified as authentic, fully reproducible, mathematically verified, and strictly compliant with the approved ResNet-18 UTTA-Med research charter.
