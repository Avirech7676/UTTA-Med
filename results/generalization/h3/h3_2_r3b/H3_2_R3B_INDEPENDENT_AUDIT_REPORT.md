# UTTA-Med — Stage H3.2-R3b: Independent Reproducibility & Statistical Audit Report

**Audit Timestamp:** 2026-10-16T05:28:51.388471+00:00  
**Compute Platform:** Python 3.14.6 (Windows-11-10.0.26300-SP0)  
**Final Audit Determination:** `SUPERIOR_VERIFIED`  

## 1. Executive Summary & Verification Verdict

This independent audit verified the empirical claims of Stage H3.2-R3a directly from raw prediction files and ground-truth coordinates without GPU inference or model retraining. Under an exhaustive 10,000-replicate paired case-clustered bootstrap, the determination is **`SUPERIOR_VERIFIED`**.

### Key Audit Determinations:
1. **Reconstruction Fidelity:** All 15 seed/arm combinations match H3.2-R3a records bit-for-bit (True).
2. **Statistical Significance Verified ($B = 10,000$):**
   - Micro-Pooled Estimand: $\Delta F_1 = +0.0075$ [95% CI: $+0.0004, +0.0156$], $p = 0.0187$.
   - Seed-Macro Estimand: $\Delta F_1 = +0.0066$ [95% CI: $+0.0001, +0.0140$], $p = 0.0223$.
   - In both cases, the entire 95% bootstrap confidence interval is strictly positive.
3. **Discrepancy Explained:** The $+0.0066$ value is the arithmetic mean of seed-level differences (Macro), while $+0.0077$ was the pooled micro-F1 difference across seeds. Both are valid and strictly positive.
4. **True Positive Audit (160 -> 158):** SafeAdapt loses exactly two True Positives on Seed 42 ($57 \to 55$), while eliminating 19 False Positives ($212 \to 193$). SafeAdapt does NOT preserve 100% of TPs.
5. **Sampling Distribution Limitation:** 99.73% of tiles originate from y <= 512 px due to sequential grid traversal with stride=2. Findings apply strictly to the evaluated 1,100-window cohort.

## 2. Reconstructed Detection Performance Matrix

| Seed | Arm | TP | FP | FN | Precision | Recall | Reconstructed F1 | R3a F1 | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 | Arm_A | 57 | 101 | 60 | 0.3608 | 0.4872 | **0.4145** | 0.4145 | [MATCH] |
| 42 | Arm_B | 51 | 60 | 66 | 0.4595 | 0.4359 | **0.4474** | 0.4474 | [MATCH] |
| 42 | Arm_C | 51 | 60 | 66 | 0.4595 | 0.4359 | **0.4474** | 0.4474 | [MATCH] |
| 42 | Arm_D | 51 | 60 | 66 | 0.4595 | 0.4359 | **0.4474** | 0.4474 | [MATCH] |
| 42 | Arm_E | 55 | 85 | 62 | 0.3929 | 0.4701 | **0.4280** | 0.4280 | [MATCH] |
| 7 | Arm_A | 53 | 58 | 64 | 0.4775 | 0.4530 | **0.4649** | 0.4649 | [MATCH] |
| 7 | Arm_B | 46 | 47 | 71 | 0.4946 | 0.3932 | **0.4381** | 0.4381 | [MATCH] |
| 7 | Arm_C | 46 | 47 | 71 | 0.4946 | 0.3932 | **0.4381** | 0.4381 | [MATCH] |
| 7 | Arm_D | 46 | 47 | 71 | 0.4946 | 0.3932 | **0.4381** | 0.4381 | [MATCH] |
| 7 | Arm_E | 53 | 58 | 64 | 0.4775 | 0.4530 | **0.4649** | 0.4649 | [MATCH] |
| 123 | Arm_A | 50 | 53 | 67 | 0.4854 | 0.4274 | **0.4545** | 0.4545 | [MATCH] |
| 123 | Arm_B | 40 | 40 | 77 | 0.5000 | 0.3419 | **0.4061** | 0.4061 | [MATCH] |
| 123 | Arm_C | 40 | 40 | 77 | 0.5000 | 0.3419 | **0.4061** | 0.4061 | [MATCH] |
| 123 | Arm_D | 40 | 40 | 77 | 0.5000 | 0.3419 | **0.4061** | 0.4061 | [MATCH] |
| 123 | Arm_E | 50 | 50 | 67 | 0.5000 | 0.4274 | **0.4608** | 0.4608 | [MATCH] |

## 3. Comprehensive Sensitivity Grid (4 Configurations)

| Boundary Filter ($h \ge 1.0\text{ px}$) | Matching Radius | Arm A Mean F1 | Arm E Mean F1 | $\Delta F_1$ | Superior? |
|:---:|:---:|:---:|:---:|:---:|:---:|
| True | 30.0 px | 0.4447 | 0.4513 | **+0.0066** | True |
| True | 25.0 px | 0.4447 | 0.4513 | **+0.0066** | True |
| False | 30.0 px | 0.3785 | 0.3821 | **+0.0036** | True |
| False | 25.0 px | 0.3785 | 0.3808 | **+0.0023** | True |

## 4. Programmatic Consistency Checks (9/9 PASS)

| Check Name | Status | Result |
|---|:---:|:---:|
| `check_raw_metrics_reconstructed_exactly` | PASS | True |
| `check_bootstrap_10k_micro_strictly_positive` | PASS | True |
| `check_bootstrap_10k_macro_strictly_positive` | PASS | True |
| `check_tost_strict_margin_satisfied` | PASS | True |
| `check_true_positive_difference_documented` | PASS | True |
| `check_false_positive_reduction_documented` | PASS | True |
| `check_spatial_sampling_limitation_documented` | PASS | True |
| `check_matching_radius_invariance_verified` | PASS | True |
| `check_decision_rule_applied_strictly` | PASS | True |
