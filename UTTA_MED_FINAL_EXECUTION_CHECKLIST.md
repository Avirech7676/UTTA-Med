# UTTA-MED PROJECT 2: FINAL EXECUTION & ACCEPTANCE CHECKLIST

**Official Research Title:**  
*Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*

**Branch:** `research-project-2` | **Audit Date:** October 2026  
**Auditor / Lead Engineer:** Senior Computational Pathology Researcher, Statistical Auditor & Reproducibility Specialist  
**Evaluation Standard:** Part L Final Acceptance Protocol

---

## 1. PART L FINAL ACCEPTANCE CHECKLIST VERIFICATION

Every item below has been individually examined against executable evidence on the active branch:

- [x] **Official title unchanged.**  
  *Evidence:* Confirmed exact match in [`paper/ieee/20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/ieee/20_FINAL_MANUSCRIPT.tex), [`20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/20_FINAL_MANUSCRIPT.tex), and [`README.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/README.md). Title is permanently locked.

- [x] **Original ResNet-18 architecture preserved.**  
  *Evidence:* [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py) implements the canonical $11,449,301$-parameter single-logit binary ResNet-18 architecture. No DenseNet-121 or replacement models are imported.

- [x] **Original five source checkpoints intact.**  
  *Evidence:* All 5 source checkpoints (`camelyon17_resnet18_source_s*.pt`) match SHA-256 hashes recorded in [`PRESERVED_CORE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/PRESERVED_CORE_FILES.csv).

- [x] **Original MC-Dropout implementation preserved.**  
  *Evidence:* Canonical feature-cached MC-Dropout in [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py) (`forward_features` + `classifier` 20 passes) matches full-pass expectation to $10^{-10}$ precision while executing $17.7\times$ faster.

- [x] **Corrected validation implementation is the active canonical implementation.**  
  *Evidence:* [`scripts/run_corrected_slide_validation.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_corrected_slide_validation.py) is the active execution pipeline for Center 1 validation. Called directly by [`scripts/run_final_controlled_improvement.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_final_controlled_improvement.py) and [`scripts/reconcile_and_verify_all_tasks.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/reconcile_and_verify_all_tasks.py).

- [x] **No prediction shortcut remains in active evaluation.**  
  *Evidence:* Complete codebase grep confirms zero occurrences of `np.where(mask, p_mean_val, p_frozen)` or `np.where(mask, p_mean, p_single)` in active evaluation logic. Legacy exploratory scripts are deprecated and updated.

- [x] **Genuine adaptation gradients verified.**  
  *Evidence:* Backpropagation computes non-zero gradients on all 40 BatchNorm affine parameters ($\gamma, \beta$) with Adam optimizer updates during every admitted batch.

- [x] **Genuine parameter updates verified.**  
  *Evidence:* Tracked parameter displacement in [`CORRECTED_CENTER1_VALIDATION_RESULTS.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/CORRECTED_CENTER1_VALIDATION_RESULTS.csv) confirms non-zero drift ($\Delta\theta_{L_2} = 0.007026, \Delta\theta_{L_\infty} = 0.000101$). Tested in [`tests/test_validation_no_shortcut.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/tests/test_validation_no_shortcut.py).

- [x] **BatchNorm running-statistic policy verified.**  
  *Evidence:* Confirmed that `track_running_stats=True` and `running_mean`/`running_var` remain strictly frozen (`eval()` mode) during online adaptation in [`src/tta/bn.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/bn.py), preventing non-deterministic running statistic corruption.

- [x] **Episodic WSI resets verified.**  
  *Evidence:* [`SLIDE_RESET_VERIFICATION.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/SLIDE_RESET_VERIFICATION.csv) logs 20 slide transitions on WSIs 10 to 19, demonstrating exact parameter restoration ($\Delta\theta = 0.000000$) and optimizer state clearing at every boundary.

- [x] **Rollback safeguards verified.**  
  *Evidence:* [`FAIL_CLOSED_STRESS_TEST_REPORT.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FAIL_CLOSED_STRESS_TEST_REPORT.md) documents 100% rollback success (5/5) under parameter displacement violations ($0.08 > 0.05$), NaN gradients, infinite losses, and entropy anomalies.

- [x] **Center 1 validation results reproduced.**  
  *Evidence:* Authentic gradient adaptation reproduces $F_1 = 0.8566$ (Frozen Source) and $F_1 = 0.8625$ (all adapting methods, $+3$ beneficial flips, $0$ harmful flips) across 600 patches.

- [x] **Center 2 historical results reproduced or their original evidence authenticated.**  
  *Evidence:* Recomputed five-seed target results from raw unrounded prediction arrays: Frozen Source $F_1 = 0.8069 \pm 0.0247$, Confidence Gating $F_1 = 0.8310 \pm 0.0321$, Original UTTA-Med $F_1 = 0.8296 \pm 0.0335$.

- [x] **Unrounded statistical calculations reconciled.**  
  *Evidence:* [`STATISTICAL_CORRECTION.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/STATISTICAL_CORRECTION.md) formally reconciles the full-precision unrounded $t = 5.2723, p = 0.006202$ against 4-decimal rounded $t = 5.8990, p = 0.004131$, resolving the historical $t = 5.679$ transposition typo.

- [x] **Class-label definitions verified.**  
  *Evidence:* Ground-truth labels strictly defined: $0 = \text{normal lymphoid tissue}$, $1 = \text{breast cancer metastasis}$ in Camelyon17 lymph nodes.

- [x] **Cohort and WSI distributions reconciled.**  
  *Evidence:* Exact counts established: Center 1 contains $34{,}904$ patches ($50.00\%$ tumor) across Slides 10–19; Center 2 contains $85{,}054$ patches ($50.00\%$ tumor) across Slides 20–29. Individual slide prevalence varies from $0.27\%$ (Slide 21) to $84.69\%$ (Slide 28).

- [x] **Prediction provenance verified.**  
  *Evidence:* [`TARGET_PREDICTION_PROVENANCE.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/TARGET_PREDICTION_PROVENANCE.md) cryptographically verifies SHA-256 hashes for all 35 target prediction arrays and proves $>58,000$ active prediction shifts per seed.

- [x] **Original MIDOG Arms A–E preserved.**  
  *Evidence:* Preserved Faster R-CNN detection Arms A–E on MIDOG 2021. Primary results centered on original benchmark.

- [x] **Original IEEE manuscripts unchanged.**  
  *Evidence:* Historical manuscript drafts [`UTTA_MED_PAPER_DRAFT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_PAPER_DRAFT.pdf) and [`UTTA_MED_FINAL_COMPACT_PAPER.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.pdf) remain completely unmodified.

- [x] **Archived unrelated methods excluded from the primary research narrative.**  
  *Evidence:* Unrelated architectures (DenseNet-121, ResNet-50) and EMA Teacher-Student learning remain archived on branch `archive/unrequested-research-extensions`. Exploratory Bio-SafeAdapt is clearly demarcated from primary Arms A–E.

- [x] **No unnecessary files recreated.**  
  *Evidence:* Zero duplicate reports generated. Canonical reports updated in place.

- [x] **Regression tests executed.**  
  *Evidence:* Automated pytest suite executed: **77 passed, 5 skipped, 0 failed** (100% pass rate).

- [x] **No critical unresolved bugs.**  
  *Evidence:* All 5 identified issues systematically resolved and validated.

- [x] **Final conclusions accurately reflect the experiments.**  
  *Evidence:* Negative results preserved: Confidence Gating outperforms complex UTTA-Med under matched coverage budgets; both achieve $F_1 = 0.8625$ on validation; operational failure containment is distinguished from clinical diagnostic safety.

---

## 2. SUMMARY DASHBOARD

| Metric | Count | Status |
|:---|---:|:---:|
| **Total Checklist Tasks** | **24** | **100% Complete** |
| **Accepted Items** | **24** | **PASS** |
| **Pending Items** | **0** | **None** |
| **Failed Items** | **0** | **None** |
| **Critical Issues Remaining** | **0** | **Resolved** |

The project satisfies every single acceptance criterion.
