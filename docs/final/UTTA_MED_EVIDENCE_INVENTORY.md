# UTTA-Med: Comprehensive Evidence Inventory and Scientific Audit Ledger

**Audit Protocol:** UTTA-Med v2 — Stage F Consolidation  
**Auditor Roles:** Independent Scientific Auditor, Senior Medical Imaging ML Researcher, Reproducibility Specialist  
**Date of Audit:** October 2026  
**Evaluation Environments:** PyTorch 2.x CPU / CUDA, Torchvision, Scikit-learn, WILDS Camelyon17 Benchmark  
**Canonical Source Checkpoint SHA-256:** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` (`checkpoints/camelyon17_resnet18_source_s42_BEST.pt`)

---

## 1. Executive Summary and Scope Demarcation

This inventory reconciles all frozen experimental artifacts, datasets, code revisions, and numerical outputs spanning the **UTTA-Med project lifecycle (v1 to v2 Stage E2)**. In accordance with strict medical imaging machine learning auditing standards, this inventory establishes a categorical barrier between:

1. **Historical Exploratory Benchmarking (v1):** 5-seed multi-center evaluations investigating test-time adaptation (TTA) instabilities, empirical entropy collapse in histopathology, and heuristic uncertainty gating.
2. **Methodological Development and Calibration (v2 Stages B through E1.5):** Safeguard controller construction, metric forensics, collapse monitor calibration, and gating efficiency analysis strictly confined to **Center 1 (Hospital 1)** development data.
3. **External Target Validation (v2 Stage E2):** A single, immutable, pre-registered test-time adaptation evaluation on the isolated external target domain **Center 2 (Hospital 2)** under frozen hyperparameters.

All previously identified bugs, transcription discrepancies, and implementation defects are documented with root-cause forensics and verification proofs.

---

## 2. Cryptographic Manifest of Authoritative Artifacts

The following machine-readable files constitute the authoritative scientific evidence base for the UTTA-Med project. Manual transcriptions are rejected in favor of bit-level validation against these files.

| Artifact Path | SHA-256 Checksum | Phase / Domain | Description / Denominator |
|:---|:---|:---|:---|
| `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` | Source Checkpoint | Pre-trained ResNet-18 base model (Seed 42) |
| `results/tables/headline_5seed.json` | `f3893c5d6e246e7fcae8c56fa68c5b969e46a788876c1f8a8cf38c114ce9bf14` | v1 Multi-Center | 5-seed Camelyon17 performance metrics across methods |
| `results/tables/table_main_results_final.csv` | `320df9cd65e84be39a4b3bbb5f4408a47d5c932bbee037dab9f25006a66f2243` | v1 Multi-Center | Aggregated 5-seed summary table (Mean ± SD) |
| `results/tables/table_paired_statistics_final.csv` | `b7510afe90e26e6e21c0f9ce87fcb1f0415ee358a60671f5ba5b37eb1d11f2f6` | v1 Multi-Center | Paired t-tests, Holm-adjusted p-values, Cohen's d |
| `results/v2/stage_c2_calibration_results.json` | `7332da7d1b326dafe68307db1fe8da82ee0a9057864f84c8a24ca18a7c13c7bb` | v2 Hospital 1 | Stage C2 collapse monitor calibration (320 patches) |
| `results/v2/stage_d2/stage_d2_benchmark_results.json` | `143ca46038968393b160ed8a7bf5c013e0c845d30c9e95b3367be33fecb02b7c` | v2 Hospital 1 | Gating comparison (Conf vs MC vs Hybrid, 320 patches) |
| `results/v2/stage_e0/stage_e0_reconciliation.json` | `0da8f056d65d5012655c7605b79ec44708c67655c9a756030df5ceb99df6984c` | v2 Hospital 1 | Forensic reconciliation of Brier metric & patch ordering |
| `results/v2/stage_e1/stage_e1_evaluation_results.json` | `ab060f1ec9e9d725e667272e801488739d54a153f6ddaa75dfaebae611a726fc` | v2 Hospital 1 | 4-arm streaming evaluation on Hospital 1 (320 patches) |
| `results/v2/stage_e2/stage_e2_target_results.json` | `f5c90277991999ecc6ae5a63fa3532d74fb5a14b417207d4071efd1533ac1ff4` | v2 Hospital 2 | 4-arm target evaluation on Hospital 2 (1,280 patches) |
| `configs/v2/safeadapt_e2_frozen.yaml` | `3042225de8be5bd013a5725898a787802714004fb2b3ee95d61ba14e18df1418` | v2 Target Spec | Frozen configuration executed for Stage E2 evaluation |

---

## 3. Cohort Split and Evaluation Denominators

A major vulnerability in medical imaging machine learning literature is denominator ambiguity (e.g., claiming full whole-slide image evaluation when assessing subsampled patch sets). Below is the exact inventory of patient cohorts, slide identifiers, and patch denominators across all project stages:

```
+---------------------------------------------------------------------------------------------------+
| CAMELYON17-WILDS HISTOPATHOLOGY DATASET (5 HOSPITALS / CENTERS)                                   |
+----------------------------------+----------------------------------+-----------------------------+
| Centers 0, 3, 4                  | Center 1 (Hospital 1)            | Center 2 (Hospital 2)       |
| Training & In-Domain Validation  | Development / Calibration Cohort | External Target Validation  |
| Source Model Training (Seeds 7,  | WSIs: slide_011, 013, 015,       | WSIs: slide_020, 021, 022,  |
| 42, 99, 123, 2024)               |       017, 019                   |       023, 024, 025, 026,   |
|                                  | Total Patches Avail: 23,987      |       027, 028, 029         |
|                                  | Stage C2-E1 Subsample: 320       | Stage E2 Subsample: 1,280   |
|                                  | (64 patches/WSI, 5 slides)       | (128 patches/WSI, 10 slides)|
+----------------------------------+----------------------------------+-----------------------------+
```

### Detailed Cohort Specifications

1. **Source Training Cohort (Centers 0, 3, 4):**
   - Architecture: ResNet-18 (and exploratory ResNet-50).
   - Standard Cross-Entropy loss training with random cropping, horizontal/vertical flips, and color jitter.
   - 5 independent random seeds evaluated in v1: 7, 42, 99, 123, 2024.
2. **Hospital 1 (Center 1) Development Cohort:**
   - Slide Identifiers: `patient_011_node_4` (11), `patient_013_node_1` (13), `patient_015_node_2` (15), `patient_017_node_1` (17), `patient_019_node_1` (19).
   - Class Distribution (320-patch canonical evaluation set): 107 tumor patches (33.44%), 213 normal patches (66.56%). Slide 15 contains 0 tumor patches (purely normal tissue).
   - Sampling Budget: Uniform random subsampling of 64 patches per WSI (total $N = 320$).
   - Prediction Timing:
     - Stage D1 / D2: Post-adaptation evaluation (predictions evaluated on current batch after adapting on it).
     - Stage E0 / E1: Pre-adaptation streaming evaluation (predictions evaluated before model update on current batch).
3. **Hospital 2 (Center 2) Target Cohort:**
   - Slide Identifiers: `patient_020` through `patient_029` (10 consecutive test WSIs).
   - Total Available Patches: 33,560 patches.
   - Evaluated Budget (Stage E2 Frozen Protocol): 128 patches per WSI across 10 slides (total $N = 1,280$ patches; 40 batches of size 32).
   - Class Balance: 400 tumor patches (31.25%), 880 normal patches (68.75%). Slides 20 and 21 contain 0 tumor patches (single-class normal tissue).
   - Prediction Timing: Pre-adaptation streaming evaluation. Model updates occur after prediction. Parameters and monitor reset at each WSI boundary.

---

## 4. Methodological Progression: v1 to v2

### 4.1 Historical v1 Architecture and Empirical Findings
- **Techniques Evaluated:** Source-only, Tent (Wang et al., 2021), EATA (Niu et al., 2022), Random-Gated TTA, Confidence-Gated TTA, and UTTA-Med (v1).
- **Primary Mechanism:** Unsupervised entropy minimization targeting batch normalization affine parameters $(\gamma, \beta)$ with learning rate $\eta = 10^{-4}$.
- **Core v1 Observation:** Tent and Random-Gated TTA suffered catastrophic collapse on ResNet-50 and severe instability on ResNet-18, driving batch predictions to all-tumor or all-normal states.
- **v1 Flaws Identified During Audit:**
  - Reliance on stochastic Monte Carlo (MC) dropout forward passes for epistemic uncertainty estimation, introducing heavy computational latency ($>3\times$).
  - Post-adaptation evaluation masking potential single-step degradation.
  - Absence of hard mathematical parameter displacement boundaries, relying entirely on loss thresholding.

### 4.2 SafeAdapt v2 Controller Innovations (Stage B & Stage C)
To convert empirical adaptation into a reliable clinical pipeline, v2 introduced deterministic runtime safeguards:
1. **Parameter Displacement Constraint ($L_\infty \le \epsilon$):**
   - Strict clipping of all adapted parameters to $\|\theta_t - \theta_0\|_\infty \le 0.05$.
   - Any parameter update violating this bound is rejected or clamped, guaranteeing that the model cannot drift indefinitely from the validated source model.
2. **Numerical Gradient Interception:**
   - Immediate detection and rejection of candidate updates containing `NaN` or `Inf` gradients or activations.
3. **Atomic State Rollback:**
   - Pre-update parameter snapshots, optimizer states, and rolling statistics are stored in host memory. If an update is rejected, the model reverts to the previous state atomically without corrupting subsequent inference batches.
4. **Sliding-Window Collapse Monitor:**
   - Tracks entropy, maximum softmax probability (MSP), and predicted class distribution over a rolling window.
   - Two-stage escalation: Warning state triggers temperature scaling and threshold tightening; Alarm state triggers immediate fallback to the immutable source model.
5. **WSI-Boundary Invariance Reset:**
   - Complete reset of model parameters, optimizer buffers, and monitoring states at each whole-slide boundary, eliminating cross-slide contamination and batch-order hysteresis.

---

## 5. Comprehensive Audit Ledger: Documented Historical Corrections

During the v2 auditing lifecycle, multiple technical discrepancies and latent bugs were discovered, forensically audited, and definitively corrected:

### Issue 1: Stage C2 / C2.5 Subsampling Denominator Ambiguity
- **Discrepancy:** Initial Stage C2 documentation referenced 23,987 available patches on Hospital 1, but evaluation tables reported exactly 320 patches.
- **Root Cause:** A smoke-test cap of 64 patches per WSI ($5 \times 64 = 320$) had been implemented in DataLoader sampling to facilitate rapid iteration, but the documentation had not explicitly separated the available pool from the active evaluation set.
- **Audit Correction (Stage C2.5):** Canonical evaluation denominator was formally frozen at $N = 320$ patches. The exact patch IDs, coordinates, and ground-truth labels were serialized into `results/v2/stage_e0/stage_e0_reconciliation.json`. Evaluated metric equality between Frozen Source and Active SafeAdapt on Hospital 1 was proven to stem from micro-updates ($\Delta \approx 2 \times 10^{-5}$) having zero impact on argmax threshold classifications.

### Issue 2: Stage D2 Brier Score Reporting Discrepancy
- **Discrepancy:** The Stage C2.5 report documented a true frozen source Brier score of $0.082867$ on Hospital 1, whereas `docs/v2/STAGE_D2_REPORT.md` stated a source Brier score of approximately $0.0573$.
- **Forensic Investigation (Stage E0):** Recomputed all calibration metrics directly from bit-level predictions. The machine-readable JSON file `stage_d2_benchmark_results.json` actually recorded $0.082866569$, exactly matching Stage C2.5. The number $0.0573$ was a human transcription typo copied from an earlier v1 coverage-risk summary table.
- **Audit Correction (Stage E0):** Machine-readable JSON was reaffirmed as the sole source of truth. Automated assertion tests were integrated into the CI suite to prevent markdown manual transcription errors.

### Issue 3: Stage E1 Collapse Monitor Muting Defect
- **Discrepancy:** Stage E1 reported 0 collapse alarms triggered across all arms on Hospital 1.
- **Root Cause:** An audit of `safeadapt/monitor.py` revealed that the monitor parameter `min_samples_for_alarm` defaulted to 100, while the rolling window size was set to $W = 10$. Because each WSI only evaluated 64 patches (2 batches of 32) and the monitor reset at every WSI boundary, the sample count never exceeded 64. Consequently, `is_sufficient_samples` evaluated permanently to `False`, rendering the collapse detector mathematically inert.
- **Audit Correction (Stage E1.5):** The detector was recalibrated: window size $W = 50$, minimum observation threshold $\min = 20$, and evaluation batch count expanded to $N = 128$ patches per slide in Stage E2. Under this verified configuration, 40 eligible decisions were registered on Hospital 2 with zero false alarms.

### Issue 4: Prediction Timing Semantics Shift
- **Discrepancy:** Early v1 experiments and Stage D1 used post-adaptation evaluation timing (predictions evaluated on current batch after adapting on it).
- **Scientific Impact:** Post-adaptation evaluation allows a batch to benefit from self-adaptation, which can mask immediate boundary degradation.
- **Audit Correction (Stage E1 / E2):** Protocols were frozen to strict **pre-adaptation prediction timing**. Every batch is first evaluated by the model in its current state; adaptation occurs subsequently to update weights for future batches within the slide. This represents an authentic online streaming clinical deployment simulation.

---

## 6. Audit Verdict

All historical artifacts and codebases are categorized as:
- **Bit-level verified against raw machine-readable JSON/CSV logs.**
- **Statistically cross-referenced against exact slide identifiers.**
- **Strictly separated between exploratory development (Hospital 1) and external target evaluation (Hospital 2).**
