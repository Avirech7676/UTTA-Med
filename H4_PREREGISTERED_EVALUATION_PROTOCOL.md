# UTTA-Med — Stage H4: Preregistered Evaluation Protocol
## Spatially Representative Target Generalization Audit (Scanner 3)

**Protocol Status:** PREREGISTERED, CORRECTED & FROZEN (Execution-Ready Phase A)  
**Audit Author:** Senior Computational Pathology Researcher & Independent Reproducibility Auditor  
**Date of Freeze:** 2026-10-16  
**Target Domain:** MIDOG 2021 Scanner 3 (Aperio ScanScope CS2, Cases 101–150)  
**Primary Evaluation Budget:** 50 Cases × 22 Windows = **1,100 Non-Overlapping 512×512 Windows**  
**Realized Captured Mitoses:** **118 Certified Ground-Truth Mitoses** (across 101 positive windows)  
**Primary Statistical Estimand:** **Seed-Macro Averaged F1 Difference** ($\Delta_{\text{macro}} = \frac{1}{|S|}\sum_{s \in S} [F_{1, \text{Arm\_E}}^{(s)} - F_{1, \text{Arm\_A}}^{(s)}]$)  
**Secondary Statistical Estimand:** Micro-Pooled F1 Difference ($\Delta_{\text{micro}}$)  
**Statistical Inference:** Paired Case-Clustered Bootstrap ($B = 10,000$ Joint Resamples)  

---

## 1. Problem Statement & Scientific Objective

Stage H3.2-R3b computationally reproduced the reported SafeAdapt improvement over Frozen Source Baseline:
- Seed-Macro: $\Delta F_1 = +0.0066$, 95% CI $[+0.0001, +0.0140]$, $p = 0.0223$.
- Micro-Pooled: $\Delta F_1 = +0.0075$, 95% CI $[+0.0004, +0.0156]$, $p = 0.0187$.

However, the independent reproducibility audit revealed a **critical spatial sampling limitation**:
**99.73% of evaluated patches (1,097 / 1,100) originated from the first two rows ($y \le 512\text{ px}$)** due to sequential row-major traversal with stride = 2.

**Stage H4 Primary Question:**
Does SafeAdapt's modest statistical advantage persist when Scanner 3 target patches are sampled across a **spatially balanced, 4×4 stratified 2D grid** covering the full depth of the declared 10-HPF tumour tissue ROIs?

---

## 2. Annotation-Coverage Validity Gate (Audit Determination)

1. **Exhaustiveness within ROI:**
   The 50 Scanner 3 cases in MIDOG 2021 represent standardized 10 High Power Fields (HPFs) of tumour tissue (averaging 6,476 × 4,846 px $\approx 1.96\text{ mm}^2$ each).
   Within these declared boundaries, **every single mitotic figure was exhaustively annotated** by expert pathologists with consensus review ($N = 688$ mitoses).
2. **Certification of Negative Regions:**
   Unannotated viable tissue regions within the 10-HPF canvas are certified mitosis-negative.
3. **Scientific Boundary Limitation:**
   This evaluation certifies **full-ROI spatial generalization**, NOT whole-slide (WSI) depth generalization. Findings must explicitly state that the population represents standardized 10-HPF tumour regions.

---

## 3. Preregistered Sampling Frame & Stratification

- **Grid Canvas:** Each case is partitioned into non-overlapping 512 × 512 candidate tiles.
- **Stratification Geometry:** 16 spatial strata (4 × 4 grid: Rows 0–3, Cols 0–3).
- **Allocation Rule:** Each of the 16 strata receives at least 1 evaluation window; 6 distinct strata receive a second window via seeded permutation.
- **Random Seed:** Fixed master seed `20261016` (deterministic, unalterable).
- **Spatial Distribution Outcome:**
  - Row 0 ($y \in [0, 25\%)$): 270 patches (24.55%)
  - Row 1 ($y \in [25\%, 50\%)$): 276 patches (25.09%)
  - Row 2 ($y \in [50\%, 75\%)$): 277 patches (25.18%)
  - Row 3 ($y \in [75\%, 100\%]$): 277 patches (25.18%)
  - **Upper Half ($y \le 50\%$) vs Lower Half ($y > 50\%$):** 49.64% vs 50.36% (near-perfect vertical balance).
  - **Left Half ($x \le 50\%$) vs Right Half ($x > 50\%$):** 49.82% vs 50.18% (near-perfect horizontal balance).

---

## 4. Tissue Quality Handling & Fail-Closed Protocol

1. **Image-Only Eligibility Criteria:**
   - Non-zero background fraction: $\text{nonzero\_frac} \ge 0.80$
   - Cellular tissue density fraction: $\text{tissue\_frac} \ge 0.25$
   - Grayscale luminance variance: $\text{variance} \ge 18.0$
2. **Strict Ban on Silent Tile Replacement:**
   - Under no circumstances may tiles be resampled or replaced based on labels, quality checks, or model outcomes.
   - If any locked window fails eligibility criteria during pre-adaptation inspection, execution must halt immediately, emit an eligibility failure report (`h4_tissue_eligibility_failures.csv`), and request protocol approval.

---

## 5. Experimental Constraints & Checkpoint Locking

1. **Zero Retraining or Checkpoint Modification:**
   - Seed 42: `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` (Authenticated)
   - Seed 7: `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` (Authenticated)
   - Seed 123: `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` (Authenticated)
2. **Five Preserved Adaptation Arms:**
   - Arm A: Frozen Source Baseline (Zero adaptation updates).
   - Arm B: Conventional TTA (Unconstrained entropy minimization on BN affine).
   - Arm C: Bounded TTA ($L_\infty \le 0.05$ parameter displacement constraint).
   - Arm D: Monitored TTA (Rollback upon silence or proliferation alarm).
   - Arm E: SafeAdapt (Source-anchored gating + dual-stream recovery).
3. **Primary Detection Protocol:**
   - Confidence threshold: $\tau = 0.20$.
   - Primary matching radius: $r = 30.0\text{ px}$ ($7.5\ \mu\text{m}$ at nominal $0.25\ \mu\text{m/px}$).
   - Sensitivity matching radius: $r = 25.0\text{ px}$ ($6.25\ \mu\text{m}$).
   - Minimum box height filter: $h \ge 1.0\text{ px}$ (uniform across all arms and seeds). Also report unfiltered.
4. **Target Label Quarantine & Annotation Isolation:**
   - The frozen manifest (`h4_frozen_patch_manifest.csv`) contains strictly annotation-free patch coordinates and metadata.
   - Target annotations are quarantined in `h4_evaluation_ground_truth.csv` and accessed strictly post-inference.

---

## 6. Statistical Inference & Uncertainty Estimation

- **Primary Contrast:** Seed-Macro $\Delta F_1 = \frac{1}{|S|}\sum_{s \in S} F_{1, \text{Arm\_E}}^{(s)} - \frac{1}{|S|}\sum_{s \in S} F_{1, \text{Arm\_A}}^{(s)}$.
- **Secondary Contrast:** Micro-Pooled $\Delta F_1$ (detections pooled across cases and seeds).
- **Fixed Evaluation Budget:** The prespecified budget of 22 windows/case × 50 cases = **1,100 non-overlapping windows** is strictly locked.
- **Statistical Uncertainty & Power Statement:** While the balanced 4×4 design captures 118 ground-truth mitoses across 101 positive windows, **having 118 mitoses does not guarantee adequate statistical power**. Statistical precision and inferential uncertainty will be estimated strictly via paired case-clustered bootstrapping ($B = 10,000$ joint replicates) rather than presumed a priori from the mitosis count.
- **Classification Rules:**
  - `H4_POSITIVE_EVIDENCE`: Primary 95% bootstrap CI strictly excludes zero ($\text{CI}_{\text{lower}} > 0.0$).
  - `H4_NO_VERIFIED_ADVANTAGE`: Primary 95% bootstrap CI includes zero or is negative.
  - `H4_INCONCLUSIVE`: Execution anomalies, wide CI, or coverage limits prevent a reliable determination.
  - `H4_INVALID`: Data leakage, coordinate mismatch, or protocol violation detected.

---

## 7. Immutable Artifact Digests (SHA-256)

- `h4_frozen_patch_manifest.csv`: `e5a8345a68ba7f0c15162c0865ae37aa9730fcc7051525abfa9ff9f5ca08d1c8`
- `h4_candidate_sampling_frame.csv`: `b57f069a239d6afd6e21cc1e4abfd4d3f43801884e94209996c223c23109e6ea`
- `h4_evaluation_ground_truth.csv`: `fc33d0e3904e204409f18fe41f841308eb4a81d0c79f662e0ab8f6997082c612`
- `h4_evaluation_patch_summary.csv`: `0164432a297c47de95338f24c5508c2df08db5f805c3598fc550b4141fbd1c42`

---

## 8. Measured Empirical GPU Baseline (Tesla T4)

- **Total 3-Seed 5-Arm Measured Runtime:** **23.28 minutes (~23.3 min)** on NVIDIA Tesla T4.
- **Weekly Kaggle Quota Utilization:** 0.39 / 30.0 hours (1.30%).
- **Peak VRAM Utilization:** 1.82 GB / 14.56 GB (12.74 GB headroom).
- **Execution-Readiness Status:** Preflight benchmarks verified and certified. Full 5-arm 3-seed GPU execution awaits independent review sign-off.
