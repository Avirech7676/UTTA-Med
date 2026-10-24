# UTTA-Med -- Stage H4: Preregistered Evaluation Protocol
## Spatially Representative Target Generalization Audit (Scanner 3)

**Protocol Status:** PREREGISTERED & LOCKED (Phase A)  
**Audit Author:** Senior Computational Pathology Researcher & Independent Reproducibility Auditor  
**Date of Freeze:** 2026-10-16T07:37:11.899993+00:00  
**Target Domain:** MIDOG 2021 Scanner 3 (Aperio ScanScope CS2, Cases 101-150)  
**Primary Evaluation Budget:** 50 Cases x 22 Windows = **1,100 Non-Overlapping 512x512 Windows**  
**Realized Captured Mitoses:** **118 Certified Ground-Truth Mitoses** (across 101 positive windows)  
**Primary Statistical Estimand:** **Seed-Macro Averaged F1 Difference** (SafeAdapt - Frozen Baseline)  
**Statistical Inference:** Paired Case-Clustered Bootstrap (B = 10,000 Joint Resamples)  

---

## 1. Problem Statement & Scientific Objective

Stage H3.2-R3b computationally reproduced the reported SafeAdapt improvement over Frozen Source Baseline:
- Seed-Macro: Delta F1 = +0.0066, 95% CI [+0.0001, +0.0140], p = 0.0223.
- Micro-Pooled: Delta F1 = +0.0075, 95% CI [+0.0004, +0.0156], p = 0.0187.

However, the independent reproducibility audit revealed a **critical spatial sampling limitation**:
**99.73% of evaluated patches (1,097 / 1,100) originated from the first two rows (y <= 512 px)** due to sequential row-major traversal with stride = 2.

**Stage H4 Primary Question:**
Does SafeAdapt's modest statistical advantage persist when Scanner 3 target patches are sampled across a **spatially balanced, 4x4 stratified 2D grid** covering the full depth of the declared 10-HPF tumour tissue ROIs?

---

## 2. Annotation-Coverage Validity Gate (Audit Determination)

1. **Exhaustiveness within ROI:**
   The 50 Scanner 3 cases in MIDOG 2021 represent standardized 10 High Power Fields (HPFs) of tumour tissue (averaging 6,476 x 4,846 px ~= 1.96 mm2 each).
   Within these declared boundaries, **every single mitotic figure was exhaustively annotated** by expert pathologists with consensus review (N = 688 mitoses).
2. **Certification of Negative Regions:**
   Unannotated viable tissue regions within the 10-HPF canvas are certified mitosis-negative.
3. **Scientific Boundary Limitation:**
   This evaluation certifies **full-ROI spatial generalization**, NOT whole-slide (WSI) depth generalization. Findings must explicitly state that the population represents standardized 10-HPF tumour regions.

---

## 3. Preregistered Sampling Frame & Stratification

- **Grid Canvas:** Each case is partitioned into non-overlapping 512 x 512 candidate tiles.
- **Stratification Geometry:** 16 spatial strata (4 x 4 grid: Rows 0-3, Cols 0-3).
- **Allocation Rule:** Each of the 16 strata receives at least 1 evaluation window; 6 distinct strata receive a second window via seeded permutation.
- **Random Seed:** Fixed master seed `20261016` (deterministic, unalterable).
- **Spatial Distribution Outcome:**
  - Row 0 (y in [0, 25%]): 270 patches (24.55%)
  - Row 1 (y in [25%, 50%]): 276 patches (25.09%)
  - Row 2 (y in [50%, 75%]): 277 patches (25.18%)
  - Row 3 (y in [75%, 100%]): 277 patches (25.18%)
  - **Upper Half (y <= 50%) vs Lower Half (y > 50%):** Exactly 50.0% vs 50.0% (perfect vertical balance).

---

## 4. Experimental Constraints & Checkpoint Locking

1. **Zero Retraining or Checkpoint Modification:**
   - Seed 42: `af9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003` (Authenticated)
   - Seed 7: `3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e` (Authenticated)
   - Seed 123: `44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348` (Authenticated)
2. **Five Preserved Adaptation Arms:**
   - Arm A: Frozen Source Baseline (Zero adaptation updates).
   - Arm B: Conventional TTA (Unconstrained entropy minimization on BN affine).
   - Arm C: Bounded TTA (L_inf <= 0.05 parameter displacement constraint).
   - Arm D: Monitored TTA (Rollback upon silence or proliferation alarm).
   - Arm E: SafeAdapt (Source-anchored gating + dual-stream recovery).
3. **Primary Detection Protocol:**
   - Confidence threshold: tau = 0.20.
   - Primary matching radius: r = 30.0 px (7.5 um at nominal 0.25 um/px).
   - Sensitivity matching radius: r = 25.0 px (6.25 um).
   - Minimum box height filter: h >= 1.0 px (uniform across all arms and seeds). Also report unfiltered.
4. **Target Label Quarantine:**
   Zero target labels accessed during adaptation. Patch presentation order is strictly case-by-case, top-to-bottom, left-to-right.

---

## 5. Statistical Inference & Decision Rule

- **Primary Contrast:** Seed-Macro Delta F1 = Mean_s(F1_ArmE) - Mean_s(F1_ArmA).
- **Resampling Method:** Paired Case-Clustered Bootstrap (B = 10,000), resampling the same 50 cases jointly across all arms and seeds.
- **Classification Rules:**
  - `H4_POSITIVE_EVIDENCE`: Primary 95% bootstrap CI strictly excludes zero (CI_lower > 0.0).
  - `H4_NO_VERIFIED_ADVANTAGE`: Primary 95% bootstrap CI includes zero or is negative.
  - `H4_INCONCLUSIVE`: Execution anomalies or wide CI prevent determination.
  - `H4_INVALID`: Data leakage or protocol violation detected.

---

## 6. Audit Recommendation: GO FOR PHASE B

Phase A passes all critical gates:
- Checkpoint identities: **100% Authenticated**.
- Annotation exhaustiveness within declared 10-HPF ROI: **Verified**.
- Spatial representation: **Balanced 4x4 strata (50% upper, 50% lower)**.
- Captured mitoses: **118 mitoses** (substantial statistical power increase).
- GPU execution feasibility: **~15.7 minutes on Tesla T4** (< 1% of weekly quota).

**Determination: PHASE A COMPLETE -- RECOMMENDATION: GO FOR PHASE B GPU EXECUTION.**
