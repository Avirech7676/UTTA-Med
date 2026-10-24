# ARCHIVED EXPERIMENTS REGISTER
## Comprehensive Registry of Unrequested Extensions, Code, Configurations, and Provenance

**Project:** UTTA-Med Project 2 (`research-project-2`)  
**Archival Date:** October 2026  
**Archival Branch:** `archive/unrequested-research-extensions`  
**Archival Commit Hash:** `3a293e0`  
**Restoration Tag:** `UTTA_MED_BEFORE_DIRECTION_CLEANUP_20261024`  
**Physical Archive Directory:** [`archive/unrequested_extensions/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/archive/unrequested_extensions)  

---

## 1. Archival Policy & Guarantees

All experimental branches and files documented in this register were introduced as part of unrequested research direction expansions. To keep the active workspace clean, focused, and aligned with the original approved study, these components have been removed from the default development workflow.

### Preservation Guarantees:
1. **Zero Data Loss:** All source code, data tables, configurations, raw execution logs, and documentation remain 100% preserved in Git history on branch `archive/unrequested-research-extensions`.
2. **Local File Availability:** Essential scripts, configurations, and verification reports are also mirrored in [`archive/unrequested_extensions/`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/archive/unrequested_extensions) for immediate local reference.
3. **Reversible Restoration:** Any individual experiment or file can be restored into the active workspace in seconds using standard Git commands.

---

## 2. Register of Archived Experimental Extensions

### Experiment Suite 1: DenseNet-121 Backbone Replacement
- **Motivation for Archival:** Introduced as an unrequested replacement for the original ResNet-18 classification backbone.
- **Key Files Archived:**
  - `DENSENET121_TRAINING.py` (Local training script for DenseNet-121)
  - `DENSENET121_FIVE_SEED_CHECKPOINTS.md` (Multi-seed checkpoint plan)
  - `DENSENET121_FIVE_SEED_TRAINING_LOG.csv` (Training log schema)
  - `DENSENET121_FIVE_SEED_VALIDATION.csv` (Validation metrics)
  - `ACTUAL_DENSENET121_CHECKPOINTS.md` (Verification report)
  - `FINAL_DENSENET_CHECKPOINT_VERIFICATION.md` (Forensic verification)
  - `FINAL_DENSENET_FIVE_SEED_RESULTS.csv` (Result matrix)
  - `FINAL_DENSENET_RESULTS.csv` (Result matrix)
  - `VERIFIED_DENSENET_CHECKPOINTS.md` (Checkpoint status)
  - `scripts/kaggle_densenet121/train_densenet121_kaggle.py` (Kaggle deployment code)
  - `scripts/retrieve_and_finalize_densenet121.py` (Remote artifact retrieval)
  - `checkpoints/densenet121/` (Directory)
  - `kaggle_output_densenet121/` (Directory)
  - `utta-med-densenet121-training/` (Directory)
- **Status in Active Workspace:** Removed from active execution; code mirrored in `archive/unrequested_extensions/`.

---

### Experiment Suite 2: ResNet-50 Backbone Replacement
- **Motivation for Archival:** Exploratory scale-up study replacing ResNet-18 with ResNet-50, departing from the original paper's low-resource deployment objective.
- **Key Files Archived:**
  - `RESNET50_TRAINING.py` (Supervised training script for ResNet-50)
  - `src/models/resnet50.py` (ResNet-50 model definition with head dropout)
  - `checkpoints/source_centroids_s42_resnet50_BACKUP.pt` (ResNet-50 feature centroids)
- **Status in Active Workspace:** Removed from active execution; code mirrored in `archive/unrequested_extensions/`.

---

### Experiment Suite 3: EMA Teacher–Student Adaptation (E0/E1/E2 Suite)
- **Motivation for Archival:** Introduced as an unrequested adaptation paradigm replacing MC-Dropout uncertainty gating.
- **Key Files Archived:**
  - `EMA_CONSISTENCY_TTA.py` (Exponential Moving Average Teacher-Student implementation)
  - `FINAL_E0_E1_E2_COMPARISON.csv` (E0/E1/E2 result comparison table)
  - `FINAL_E0_E1_E2_RESULTS.csv` (Telemetry output table)
  - `FINAL_ADAPTATION_SUPERIORITY_ANALYSIS.md` (E2 vs E1 superiority audit)
- **Status in Active Workspace:** Removed from active execution; code mirrored in `archive/unrequested_extensions/`.

---

### Experiment Suite 4: Extended Classification Experiments (C0–C7 Series)
- **Motivation for Archival:** Exploratory multi-signal gating and feature-distance configurations not part of the certified manuscript.
- **Key Files Archived:**
  - `VERIFIED_C0_C7_RESULTS.csv` (C0 through C7 benchmark table)
  - `MATCHED_COVERAGE_C7_RESULTS.csv` (C7 matched-coverage table)
  - `FEATURE_DISTANCE_ABLATIONS.csv` (Source-centroid distance ablations)
  - `CLASS_AWARE_GATING_RESULTS.csv` (Class-asymmetric gating ablations)
  - `scripts/evaluate_c_series.py` (C-series evaluation runner)
  - `scripts/run_c7_matched_coverage.py` (Matched-coverage script)
  - `scripts/tune_confidence_feature_gating.py` (Hyperparameter search)
  - `src/v2/` (UTTA-Med 2.0 experimental module directory)
  - `configs/v2/` (UTTA-Med 2.0 configuration directory)
  - `results/v2/` (UTTA-Med 2.0 results directory)
- **Status in Active Workspace:** Removed from active execution; code mirrored in `archive/unrequested_extensions/`.

---

### Experiment Suite 5: Extended Detection Experiments (D8–D9 Series & Bio-SafeAdapt Expansions)
- **Motivation for Archival:** Mitotic detection expansions introduced after the completion of the original MIDOG Arms A–E study.
- **Key Files Archived:**
  - `D8_VS_D9_VERIFICATION.md` (D8 vs D9 comparative audit)
  - `D8_D9_INCREMENTAL_ABLATION_AUDIT.md` (Incremental ablation report)
  - `BIO_SAFEADAPT_ABLATIONS.csv` (Bio-SafeAdapt ablation table)
  - `BIO_SAFEADAPT_ABLATION_RESULTS.csv` (Detailed ablation results)
  - `BIO_SAFEADAPT_COMPARISON.csv` (Cross-arm comparison table)
  - `BIO_SAFEADAPT_IMPLEMENTATION_AUDIT.md` (Implementation audit)
  - `BIO_SAFEADAPT_MULTI_SCANNER_RESULTS.csv` (Multi-scanner generalization)
  - `BIO_SAFEADAPT_REAL_RESULTS.csv` (Real WSI run results)
  - `BIO_SAFEADAPT_VERIFIED_PREDICTIONS.csv` (Detection predictions)
  - `BIO_SAFEADAPT_VERIFIED_RESULTS.csv` (Verified detection table)
  - `08_BIO_SAFEADAPT_IMPLEMENTATION_AUDIT.md` (Phase report)
  - `10_BIO_SAFEADAPT_ABLATION_RESULTS.csv` (Phase report)
  - `12_BIO_SAFEADAPT_OPTIMIZATION_REPORT.md` (Phase report)
  - `scripts/generate_bio_safeadapt_ablations.py` (Script)
  - `scripts/generate_bio_safeadapt_multi_scanner.py` (Script)
  - `scripts/generate_bio_safeadapt_verified_predictions.py` (Script)
  - `scripts/generate_full_bio_safeadapt_ablations.py` (Script)
- **Status in Active Workspace:** Removed from active execution; original MIDOG Arms A–E preserved in active project (`09_VERIFIED_DETECTION_BASELINES.csv`, `VERIFIED_DETECTION_RESULTS.csv`).

---

## 3. How to Inspect or Restore Archived Experiments

### Option 1: Instant Local Inspection
To inspect any archived script or configuration without switching Git branches:
```bash
# Navigate to the local physical archive directory:
cd archive/unrequested_extensions/

# Inspect archived files:
ls -la
```

### Option 2: Selective File Restoration via Git
To pull any specific archived file back into your working directory:
```bash
# Example: restore the DenseNet training script
git checkout archive/unrequested-research-extensions -- DENSENET121_TRAINING.py

# Example: restore the ResNet-50 model
git checkout archive/unrequested-research-extensions -- src/models/resnet50.py
```

### Option 3: Full Branch Exploration
To switch into the complete unpruned research environment:
```bash
# Checkout the archival branch (read-only exploration recommended):
git checkout archive/unrequested-research-extensions

# Return to the active restored branch:
git checkout research-project-2
```
