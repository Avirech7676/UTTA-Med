# UTTA-MED PROJECT 2: FINAL EXECUTION & ACCEPTANCE CHECKLIST

**Official Research Title:**  
*Selective Test-Time Adaptation and Fail-Closed Safeguards for Digital Pathology: Multicenter Classification and Dense Detection Under Domain Shift*

**Branch:** `research-project-2` | **Audit Date:** October 2026  
**Auditor / Lead Engineer:** Senior Computational Pathology Researcher, Statistical Auditor & Reproducibility Specialist  
**Evaluation Standard:** Part L Final Acceptance Protocol & Tasks 1–14 Reconciliation

---

## 1. PART L & FINAL RECONCILIATION ACCEPTANCE CHECKLIST

Every condition below has been verified against executable code, raw dataset parquet files, cryptographic hashes, and unrounded evaluation arrays:

- [x] **Official title unchanged.**  
  *Evidence:* Exact match in [`paper/ieee/20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/paper/ieee/20_FINAL_MANUSCRIPT.tex), [`20_FINAL_MANUSCRIPT.tex`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/20_FINAL_MANUSCRIPT.tex), and [`README.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/README.md). Title is permanently locked.

- [x] **Original ResNet-18 architecture preserved.**  
  *Evidence:* [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py) implements the canonical $11,449,301$-parameter single-logit binary ResNet-18. No DenseNet-121 or replacement models are used.

- [x] **Original five source checkpoints intact.**  
  *Evidence:* All 5 source checkpoints (`camelyon17_resnet18_source_s*.pt`) match SHA-256 hashes recorded in [`PRESERVED_CORE_FILES.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/PRESERVED_CORE_FILES.csv).

- [x] **Original MC-Dropout implementation preserved.**  
  *Evidence:* Canonical feature-cached MC-Dropout in [`src/models/resnet18.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/models/resnet18.py) (`forward_features` + `classifier` 20 passes) matches full-pass expectation to $10^{-10}$ precision while executing $17.7\times$ faster.

- [x] **Corrected validation implementation is active canonical implementation.**  
  *Evidence:* [`scripts/run_corrected_slide_validation.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_corrected_slide_validation.py) is the active execution pipeline for Center 1 validation. Called directly by [`scripts/run_final_controlled_improvement.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/run_final_controlled_improvement.py) and [`scripts/reconcile_and_verify_all_tasks.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/reconcile_and_verify_all_tasks.py).

- [x] **No prediction shortcut remains in active evaluation.**  
  *Evidence:* Complete codebase grep confirms zero occurrences of `np.where(mask, p_mean_val, p_frozen)` in active evaluation logic. Legacy exploratory scripts are deprecated.

- [x] **Genuine adaptation gradients verified.**  
  *Evidence:* Backpropagation computes non-zero gradients on all 40 BatchNorm affine parameters ($\gamma, \beta$) with Adam optimizer updates during every admitted batch.

- [x] **Genuine parameter updates verified.**  
  *Evidence:* Tracked parameter displacement in [`CORRECTED_CENTER1_VALIDATION_RESULTS.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/CORRECTED_CENTER1_VALIDATION_RESULTS.csv) confirms non-zero drift ($\Delta\theta_{L_2} = 0.007026, \Delta\theta_{L_\infty} = 0.000101$). Tested in [`tests/test_validation_no_shortcut.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/tests/test_validation_no_shortcut.py).

- [x] **BatchNorm running-statistic policy verified.**  
  *Evidence:* Confirmed that `track_running_stats=True` and `running_mean`/`running_var` remain strictly frozen (`eval()` mode) during online adaptation in [`src/tta/bn.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/src/tta/bn.py), preventing running statistic corruption.

- [x] **Episodic WSI resets verified.**  
  *Evidence:* [`SLIDE_RESET_VERIFICATION.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/SLIDE_RESET_VERIFICATION.csv) logs 20 slide transitions on WSIs 10 to 19, demonstrating exact parameter restoration ($\Delta\theta = 0.000000$) and optimizer state clearing at every boundary.

- [x] **Rollback safeguards verified.**  
  *Evidence:* [`FAIL_CLOSED_STRESS_TEST_REPORT.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FAIL_CLOSED_STRESS_TEST_REPORT.md) documents 100% rollback success (5/5) under parameter displacement violations ($0.08 > 0.05$), NaN gradients, infinite losses, and entropy anomalies.

- [x] **Center 1 validation results reproduced.**  
  *Evidence:* Authentic gradient adaptation reproduces $F_1 = 0.8566$ (Frozen Source) and $F_1 = 0.8625$ (all adapting methods, $+3$ beneficial flips, $0$ harmful flips) across 600 patches.

- [x] **Center 2 target results reproduced from raw prediction arrays.**  
  *Evidence:* Recomputed five-seed target results from raw unrounded prediction arrays: Frozen Source $F_1 = 0.8069 \pm 0.0247$, Confidence Gating $F_1 = 0.8310 \pm 0.0321$, Original UTTA-Med $F_1 = 0.8296 \pm 0.0325$, Improved UTTA-Med $F_1 = 0.8303 \pm 0.0322$.

- [x] **Unrounded statistical calculations reconciled.**  
  *Evidence:* [`STATISTICAL_CORRECTION.md`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/STATISTICAL_CORRECTION.md) formally reconciles the full-precision unrounded $t = 5.2723, p = 0.006202$ against 4-decimal rounded $t = 5.8990, p = 0.004131$, resolving the historical $t = 5.679$ transposition typo.

- [x] **Patient and slide counts verified from raw metadata (Problem 1 resolved).**  
  *Evidence:* Audited raw parquet files (`c:/Users/avina/OneDrive/Desktop/Camelyon17-Data/data`). Center 2 contains **10 WSIs from exactly 9 distinct patients** (`{40, 41, 42, 44, 45, 46, 48, 51, 52}`). Patient 46 contributes both Slide 25 (node 3) and Slide 26 (node 4). Earlier reports mistakenly showed 10 distinct patient IDs due to typos (typing 44 as 43, 46 as 48 on slide 26, and 48 as 49 on slide 27). This discrepancy is completely resolved.

- [x] **Clinical staging claims audited and unverified labels excised (Problem 2 resolved).**  
  *Evidence:* Official Camelyon17-WILDS patch benchmark metadata contains binary patch labels and coordinates, but NO physical metastasis diameter measurements or clinical nodal staging annotations (pN0(i+), pN1mi, pN1a). All unsupported clinical staging claims are removed from canonical tables and classified as unverified surrogate proxies.

- [x] **Adaptation harm denominators mathematically reconciled (Problem 3 resolved).**  
  *Evidence:* Delineated both metrics: Cohort Harm Rate ($H_{\text{flips}} / N = 497.4 / 85,054 = 0.585\% \pm 0.224\%$) vs Accepted Harm Rate ($H_{\text{flips}} / k = 497.4 / 78,998 = 0.630\% \pm 0.241\%$ under unconstrained $\tau=0.85$, and $595 / 49,416 = 1.204\%$ under matched-budget $58.1\%$ coverage). Net flip benefit proves Confidence Gating yields $+1,741.8$ net correct patches vs $+1,659.8$ for UTTA-Med.

- [x] **Experiment identities restored; A3 disentangled from CAD-Gate (Problem 4 resolved).**  
  *Evidence:* In [`FIVE_SEED_ORIGINAL_UTTA_RESULTS.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FIVE_SEED_ORIGINAL_UTTA_RESULTS.csv), $A_3$ is Improved UTTA-Med (Feature-Cached MC-Dropout, AUROC $0.9351 \pm 0.0217$, F1 $0.8303 \pm 0.0322$, latency $7.82$ ms/batch). CAD-Gate was evaluated exclusively on Center 1 validation ($F_1 = 0.8625$, AUROC $0.9606$) and was never evaluated on Center 2 target benchmark.

- [x] **Coverage definitions verified across operating points (Problem 5 resolved).**  
  *Evidence:* $58.1\%$ coverage represents the matched-budget operating constraint ($\tau \approx 0.9912$ or fixed admission budget $k$). $92.88\%$ represents unconstrained online mini-batch admission under $\tau=0.85$. $87.95\%$ represents static offline source confidence filtering at $\tau=0.85$.

- [x] **Inference and adaptation latencies reconciled (Problem 4 & Task 7 resolved).**  
  *Evidence:* Measured on batch size 64: $A_0$ Frozen Source = $6.74$ ms/batch ($0.105$ ms/patch); $A_1$ Confidence Gating = $7.68$ ms/batch ($0.120$ ms/patch); $A_3$ Improved UTTA-Med = $7.82$ ms/batch ($0.122$ ms/patch); $A_2$ Original UTTA-Med = $136.29$ ms/batch ($2.130$ ms/patch) due to 20 un-cached stochastic forward passes.

- [x] **Original MIDOG Arms A–E preserved.**  
  *Evidence:* Preserved Faster R-CNN detection Arms A–E on MIDOG 2021. Primary results centered on original benchmark; exploratory extensions quarantined.

- [x] **Original IEEE manuscripts unchanged.**  
  *Evidence:* Historical manuscript drafts [`UTTA_MED_PAPER_DRAFT.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_PAPER_DRAFT.pdf) and [`UTTA_MED_FINAL_COMPACT_PAPER.pdf`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/UTTA_MED_FINAL_COMPACT_PAPER.pdf) remain completely unmodified.

- [x] **Archived unrelated methods excluded from primary research narrative.**  
  *Evidence:* Unrelated architectures (DenseNet-121, ResNet-50) and EMA Teacher-Student learning remain archived on branch `archive/unrequested-research-extensions`.

- [x] **Regression tests executed and verified.**  
  *Evidence:* Automated pytest suite executed: **77 passed, 5 skipped, 0 failed** in 27.20 seconds.

- [x] **Scientific readiness claims calibrated to peer-review standards (Problem 6 resolved).**  
  *Evidence:* Replaced hyperbolic claims ("100% production certified", "clinically proven") with calibrated scientific statements separating implementation correctness from prospective clinical trial requirements.

---

## 2. SUMMARY DASHBOARD

| Metric | Count | Status |
|:---|---:|:---:|
| **Part L & Reconciliation Acceptance Tasks** | **25** | **100% Complete** |
| **Accepted Items** | **25** | **PASS** |
| **Pending Items** | **0** | **None** |
| **Failed Items** | **0** | **None** |
| **Critical Issues Remaining** | **0** | **Resolved** |

Every acceptance condition is verified by authentic computational and dataset evidence.
