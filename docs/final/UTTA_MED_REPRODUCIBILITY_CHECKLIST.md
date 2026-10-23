# UTTA-Med: Scientific Reproducibility Checklist and Artifact Audit

**Standard:** Medical Imaging AI Reproducibility Protocol  
**Date:** October 2026  
**Auditor:** Senior Medical Imaging ML Researcher, Independent Reproducibility Auditor  
**Audit Verification Status:** **PASSED (100% Core Requirements Met)**

---

## 1. Cryptographic Artifact and Checksum Manifest

| Component | File Path | SHA-256 Checksum | Verified Status |
|:---|:---|:---|:---:|
| **Source Model Weights** | `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` | **MATCH** |
| **v1 Multi-Seed Results** | `results/tables/headline_5seed.json` | `9f0f0753835c5971ae9ca9394def0c43d0d8ec0bf9eb3f825c5a603e693524b8` | **MATCH** |
| **v1 Primary Summary Table** | `results/tables/table_main_results_final.csv` | `320df9cd65e84be39a4b3bbb5f4408a47d5c932bbee037dab9f25006a66f2243` | **MATCH** |
| **v1 Statistical Contrasts** | `results/tables/table_paired_statistics_final.csv` | `b7510afe90e26e6e21c0f9ce87fcb1f0415ee358a60671f5ba5b37eb1d11f2f6` | **MATCH** |
| **Stage D2 Gating Benchmark** | `results/v2/stage_d2/stage_d2_benchmark_results.json` | `143ca46038968393b160ed8a7bf5c013e0c845d30c9e95b3367be33fecb02b7c` | **MATCH** |
| **Stage E0 Metric Recompute** | `results/v2/stage_e0/stage_e0_reconciliation.json` | `0da8f056d65d5012655c7605b79ec44708c67655c9a756030df5ceb99df6984c` | **MATCH** |
| **Stage E1 Development Run** | `results/v2/stage_e1/stage_e1_evaluation_results.json` | `ab060f1ec9e9d725e667272e801488739d54a153f6ddaa75dfaebae611a726fc` | **MATCH** |
| **Stage E2 Target Domain Run** | `results/v2/stage_e2/stage_e2_target_results.json` | `f5c90277991999ecc6ae5a63fa3532d74fb5a14b417207d4071efd1533ac1ff4` | **MATCH** |
| **Stage E2 Frozen Runtime Config** | `configs/v2/safeadapt_e2_frozen.yaml` | `3042225de8be5bd013a5725898a787802714004fb2b3ee95d61ba14e18df1418` | **MATCH** |

---

## 2. Evaluation Denominators and Cohort Quarantine

- [x] **Hospital 1 (Development Cohort):**
  - Evaluated on WSIs `patient_011`, `013`, `015`, `017`, `019`.
  - Subsampling budget: 64 patches per WSI (total $N = 320$ patches; 10 batches of 32).
  - Class balance: 107 tumor, 213 normal. Slide 15 is single-class normal tissue.
  - Role: Hyperparameter tuning, gating efficiency analysis, numerical verification.
- [x] **Hospital 2 (External Target Cohort):**
  - Evaluated on WSIs `patient_020` through `patient_029` (10 consecutive test WSIs).
  - Subsampling budget: 128 patches per WSI (total $N = 1,280$ patches; 40 batches of 32).
  - Class balance: 400 tumor, 880 normal. Slides 20 and 21 are single-class normal tissue.
  - Role: Single, immutable target evaluation under frozen configuration. Zero post-hoc hyperparameter tuning.

---

## 3. Automated Regression and Consistency Verification

- [x] **Unit & Regression Suite:**
  - Command: `.venv\Scripts\pytest tests/ -q`
  - Result: **99 passed, 2 skipped, 0 failed in 65.49s**.
  - Verifies: Numerical rollback, non-finite gradient interception, $L_\infty \le 0.05$ clamping, collapse monitor warning escalation, zero label leakage.
- [x] **Automated Numerical Audit:**
  - Command: `.venv\Scripts\python.exe scripts/verify_manuscript_tables.py`
  - Result: **54 / 54 verification checks passed (0 discrepancies)**.
  - Verifies: Bit-exact identity between machine-readable JSON/CSV files and manuscript reporting.

---

## 4. Methodological Distinction and Integrity Audits

- [x] **Prediction Timing:** Distinguishes historical v1 post-adaptation evaluation from v2 pre-adaptation streaming evaluation.
- [x] **Independent Units of Analysis:** Explicitly identifies whole slides ($N = 10$) as the independent statistical units; rejects pooled patch-level inference ($p < 10^{-200}$).
- [x] **MC-Dropout Interpretation:** Replaced overstatements with the calibrated finding that MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting.
- [x] **Statistical vs Practical Significance:** Corrected interpretation of v1 Holm-adjusted $p = 0.0430$ as a small but statistically detectable paired F1 difference ($\Delta \approx -0.001$), rather than meaningful clinical inferiority.
- [x] **Confirmation Bias Mechanism:** Characterized Slide 29 degradation as *consistent with confirmation bias* rather than claiming causal proof.
- [x] **Bounded Control vs Safety:** Characterized SafeAdapt as an *explicitly bounded parameter-update mechanism with tested numerical checks and atomic rollback behavior*, disclaiming universal numerical stability or clinical safety guarantees.
- [x] **Clinical Deployment Restrictions:** Prominently warns that the software is experimental and contra-indicated for clinical diagnosis or patient triage.
