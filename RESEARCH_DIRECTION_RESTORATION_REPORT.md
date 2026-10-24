# RESEARCH DIRECTION RESTORATION REPORT
## Formal Restoration of Original UTTA-Med Research Scope & Clean Repository Architecture

**Author:** Senior Software Architect, Research Reproducibility Engineer & Deep Learning Specialist  
**Project:** UTTA-Med Project 2 (`research-project-2`)  
**Date:** October 2026  
**Status:** FULLY RESTORED & CERTIFIED REVERSIBLE  
**Restoration Checkpoint Tag:** `UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024`  
**Archival Git Branch:** `archive/unrequested-research-extensions`  
**Pre-Cleanup Commit Hash:** `3a293e0`  

---

## 1. Locked Official Research Identity

The official, approved, and authoritative title of this research is:

> **"Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift"**

This is the **sole authoritative title** for the study. All derivative titles, unrequested rebrandings, and replacement project names have been permanently removed from the active research workspace.

### Authoritative Reference
- **Primary Manuscript Draft:** [`UTTA_MED_PAPER_DRAFT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_PAPER_DRAFT.pdf) (Preserved without alteration)
- **Verified IEEE Camera-Ready:** [`UTTA_MED_FINAL_COMPACT_PAPER.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.pdf) (SHA-256: `9bbe2632d7ccccebf78de2996e0d738da37647e0df12b21bccf2380f4d3e4744`)
- **Camera-Ready Companion:** [`FINAL_MANUSCRIPT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FINAL_MANUSCRIPT.pdf) (SHA-256: `8eb312679b4028f2e4032e3a7559e17ac16d86b0d5f28082ea98e5d2ba255271`)

---

## 2. Research Direction Restoration Scope

During recent development, the project was unintentionally expanded into exploratory directions that departed from the approved research charter:
1. **Backbone Replacement:** Introduction of DenseNet-121 and ResNet-50 architectures as proposed replacements for the original ResNet-18 baseline.
2. **Adaptation Methodology Replacement:** Introduction of Exponential Moving Average (EMA) Teacher–Student consistency training (E0/E1/E2 framework) as a replacement for the core uncertainty-aware gating mechanism.
3. **Exploratory Classification Expansions:** Introduction of C0–C7 multi-signal centroid classification matrices not documented in the original manuscript.
4. **Exploratory Detection Expansions:** Introduction of D8–D9 experimental branches and Bio-SafeAdapt variations outside the certified MIDOG Arms A–E.
5. **Documentation Proliferation:** Creation of numerous overlapping Phase 5–11 progress reports, task matrices, and speculative performance projections.

**Restoration Action:**
All unrequested development has been **safely archived** in version control and removed from the active workspace. The active repository is now restored strictly to the original UTTA-Med research charter:
- **Core Backbone:** Genuine ResNet-18 across 5 independent random seeds (`42, 123, 2024, 7, 99`).
- **Core TTA Strategy:** Selective test-time adaptation with Monte Carlo Dropout uncertainty estimation, deterministic Confidence Gating, Tent, and EATA baselines.
- **Fail-Closed Safeguards:** Bounded parameter drift monitoring, slide-level entropy thresholds, and atomic rollback mechanisms.
- **Dense Detection:** MIDOG 2021 mitosis detection framework strictly covering original **Arms A–E**.
- **Authentic Datasets:** Camelyon17-WILDS multicenter histopathology ($N = 85,054$ target stream) and MIDOG 2021 multicenter tissue ($N = 117$ ground-truth mitoses).

---

## 3. Complete Reversible Backup Infrastructure

To ensure that no experimental work, code, or historical artifacts are permanently lost, a multi-tier reversible backup was established prior to pruning:

```mermaid
gitGraph
   commit id: "4d6f1d8" tag: "pre-audit"
   commit id: "3a293e0" tag: "UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024"
   branch archive/unrequested-research-extensions
   checkout archive/unrequested-research-extensions
   commit id: "archive-head"
   checkout research-project-2
   commit id: "active-restoration"
```

1. **Working Tree Snapshot:** All untracked and modified files were staged and committed under commit hash **`3a293e0`**.
2. **Restoration Tag:** Created lightweight Git tag **`UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024`** pointing directly to `3a293e0`.
3. **Archival Git Branch:** Created permanent archival branch **`archive/unrequested-research-extensions`** containing the complete, unpruned tree.
4. **Physical Provenance Directory:** Essential code and configuration files for unrequested extensions were copied to [`archive/unrequested_extensions/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/archive/unrequested_extensions) for transparent local inspection without needing Git branch switching.

### How to Revert the Cleanup (One-Command Recovery)
If any archived component is ever required in the future:
```bash
# To view the complete unpruned repository state:
git checkout archive/unrequested-research-extensions

# To restore a specific archived file into the active workspace:
git checkout archive/unrequested-research-extensions -- path/to/file

# To restore the entire pre-cleanup state:
git checkout UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024
```

---

## 4. Preservation of Key Scientific Assets

The integrity of all foundational scientific assets was verified via cryptographic SHA-256 checksums before and after cleanup:

### 4.1 Certified Model Checkpoints
| Checkpoint File | Role | Size (Bytes) | SHA-256 Hash |
| :--- | :--- | :---: | :--- |
| [`checkpoints/camelyon17_resnet18_source_s42_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s42_BEST.pt) | Seed 42 Source | 45,839,231 | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| [`checkpoints/camelyon17_resnet18_source_s123_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s123_BEST.pt) | Seed 123 Source | 45,841,153 | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` |
| [`checkpoints/camelyon17_resnet18_source_s2024_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s2024_BEST.pt) | Seed 2024 Source | 45,840,579 | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` |
| [`checkpoints/camelyon17_resnet18_source_s7_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s7_BEST.pt) | Seed 7 Source | 45,840,509 | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` |
| [`checkpoints/camelyon17_resnet18_source_s99_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/camelyon17_resnet18_source_s99_BEST.pt) | Seed 99 Source | 45,840,319 | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` |
| [`checkpoints/midog_resnet18_s42_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/midog_resnet18_s42_BEST.pt) | MIDOG Detection | 44,785,291 | `409a9c7d3b9d10523221affc477f5a85660009fabd6359999a5b9b95a07c0287` |

### 4.2 Certified Prediction Arrays & Manuscripts
- **Target Stream Bootstrap Predictions:** All 35 `.npz` files in [`results/data/wsi_bootstrap/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/results/data/wsi_bootstrap) ($N = 85,054$ patches per file) remain completely untouched and verified.
- **Original IEEE Paper:** [`UTTA_MED_FINAL_COMPACT_PAPER.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.pdf) (437,811 bytes, verified identical).
- **Original Manuscript LaTeX:** [`UTTA_MED_FINAL_COMPACT_PAPER.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.tex) and [`20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/20_FINAL_MANUSCRIPT.tex) remain completely preserved.

---

## 5. Summary of Pruned Components

A total of **200 unrequested and obsolete files** were cleanly removed from the active workspace root and development paths, while **1,322 active research files** were certified and retained:

1. **DenseNet-121 Replacement Suite:** Removed standalone training scripts, kernel pull directories, and verification reports.
2. **ResNet-50 Replacement Suite:** Removed `RESNET50_TRAINING.py`, `src/models/resnet50.py`, and related centroid backups.
3. **EMA Teacher-Student Suite:** Removed `EMA_CONSISTENCY_TTA.py` and derivative comparison tables.
4. **Unrequested C0–C7 & D8–D9 Extensions:** Removed secondary matrix evaluations not required by the paper.
5. **Redundant Progress Reports:** Cleaned 38 redundant markdown status reports and scratch files from the root.

All removed items are cataloged in [`UNWANTED_DEVELOPMENT_INVENTORY.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UNWANTED_DEVELOPMENT_INVENTORY.csv) and [`REMOVED_ACTIVE_WORKSPACE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/REMOVED_ACTIVE_WORKSPACE_FILES.csv).

---

## 6. Current Kaggle Kernel Status

In accordance with Step 5, we inspected remote execution telemetry for:
- **Kernel Name:** `avinashreddy7676/utta-med-densenet121-training`
- **Current Status:** `RUNNING` (on 2 $\times$ Nvidia Tesla T4 GPUs)
- **Policy Adherence:**
  - **No duplicate submissions:** Zero redundant training jobs were submitted.
  - **Safe Execution:** The running container was not forcefully terminated, preserving its runtime logs and ensuring any produced artifacts are recoverable upon natural session completion.
  - **Decoupling:** The active repository no longer depends on this remote execution to fulfill its research objectives.

---

## 7. Restoration Certification

The active repository `UTTA_MED_V1_FINAL` is hereby certified as:
1. **Restored to Original Charter:** Focused exclusively on ResNet-18, MC-Dropout selective adaptation, fail-closed safeguards, and MIDOG Arms A–E.
2. **Reproducible & Executable:** Verified by 5 automated regression test suites passing with zero errors.
3. **Reversibly Cleaned:** Guaranteed by Git commit `3a293e0`, tag `UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024`, branch `archive/unrequested-research-extensions`, and directory `archive/unrequested_extensions/`.
