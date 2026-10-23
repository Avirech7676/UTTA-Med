# UTTA-Med — Stage H3.2-R3a: Expanded Controlled Target Adaptation Audit Report

**Execution Timestamp:** 2026-10-16T13:18:29.938996+00:00  
**Compute Platform:** Tesla T4 (14.56 GB VRAM)  
**Git Revision:** `UNKNOWN_NOT_A_GIT_CHECKOUT`  
**Evaluated Target Windows:** 1100 tiles across 50 Scanner 3 cases  
**Evaluated Target Tissue Area:** 18.0224 mm² (76.0 HPF equivalents)  
**Target Mitoses Captured:** 131 certified mitoses across 111 positive tiles  
**Primary Matching Radius:** 30.0 px (7.5 µm at nominal 0.25 µm/px)  
**Primary Operating Threshold:** $\tau = 0.20$  
**Scientific Certification Decision:** `EQUIVALENT_STRICT_MARGIN`  

## 1. Executive Summary & Core Determinations

Stage H3.2-R3a addresses all four critical methodological concerns from previous target audits:
1. **Resolved Target Statistical Power:** Expanded annotation-independent 2D sampling from 100 tiles (8 mitoses) to 1100 tiles (131 mitoses), providing robust sample size.
2. **Uniform Boundary Slit Filter:** Applied the prespecified $h \ge 1.0\text{ px}$ filter across all 5 arms and 3 seeds, eliminating Seed 42's degenerate boundary proposals.
3. **Resolved Matching-Radius Discrepancy:** Standardized on official MIDOG $30.0\text{ px}$ ($7.5\ \mu\text{m}$) matching while reporting 25.0 px sensitivity side-by-side.
4. **Rigorous Equivalence vs. Inconclusive Mandate:** Replaced `CERTIFIED_NEUTRAL_EQUIVALENCE` with `EQUIVALENT_STRICT_MARGIN` under formal TOST analysis.

## 2. Five-Arm Detection Performance Matrix (Primary $r = 30.0\text{ px}$, $\tau = 0.20$)

| Arm | Protocol Description | Seed 42 F1 | Seed 7 F1 | Seed 123 F1 | Mean F1 | Mean Precision | Mean Recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Arm_A | Frozen Source Baseline | 0.4497 | 0.4609 | 0.4649 | **0.4585** | 0.4732 | 0.4555 |
| Arm_B | Conventional TTA (Unconstrained) | 0.4777 | 0.4505 | 0.4381 | **0.4554** | 0.5468 | 0.3944 |
| Arm_C | Bounded TTA ($L_\infty \le 0.05$) | 0.4777 | 0.4505 | 0.4381 | **0.4554** | 0.5468 | 0.3944 |
| Arm_D | Monitored TTA (Rollback) | 0.4839 | 0.4505 | 0.4381 | **0.4575** | 0.5482 | 0.3969 |
| Arm_E | SafeAdapt (Source-Anchored + Dual Recovery) | 0.4632 | 0.4609 | 0.4649 | **0.4630** | 0.4823 | 0.4529 |

## 3. Matching-Radius Sensitivity Analysis (30 px vs 25 px)

| Arm | F1 ($r=30\text{ px}$) | F1 ($r=25\text{ px}$) | $\Delta F_1$ | TP ($30\text{ px}$) | TP ($25\text{ px}$) | Unfiltered F1 ($30\text{ px}$) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Arm_A | 0.4585 | 0.4559 | -0.0026 | 179 | 178 | 0.3854 |
| Arm_B | 0.4554 | 0.4524 | -0.0030 | 155 | 154 | 0.3680 |
| Arm_C | 0.4554 | 0.4524 | -0.0030 | 155 | 154 | 0.3678 |
| Arm_D | 0.4575 | 0.4545 | -0.0030 | 156 | 155 | 0.3690 |
| Arm_E | 0.4630 | 0.4604 | -0.0026 | 178 | 177 | 0.3893 |

## 4. Case-Clustered Bootstrap Contrasts ($B = 1000$)

| Contrast | Mean $\Delta F_1$ | 95% Bootstrap CI | TOST Equiv ($\delta=0.03$) | TOST Equiv ($\delta=0.05$) | Scientific Determination |
|---|:---:|:---:|:---:|:---:|:---:|
| `Delta_Arm_E_vs_Arm_A` | +0.0053 | [-0.0022, +0.0132] | True | True | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_E_vs_Arm_D` | +0.0044 | [-0.0217, +0.0329] | False | True | `EQUIVALENT_STANDARD_MARGIN` |
| `Delta_Arm_D_vs_Arm_A` | +0.0009 | [-0.0290, +0.0285] | True | True | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_B_vs_Arm_A` | -0.0011 | [-0.0295, +0.0261] | True | True | `EQUIVALENT_STRICT_MARGIN` |
| `Delta_Arm_C_vs_Arm_B` | +0.0000 | [+0.0000, +0.0000] | True | True | `EQUIVALENT_STRICT_MARGIN` |

## 5. Proposal Geometry & Collapse Verification

| Arm | Seed 42 Mean Width | Seed 7 Mean Width | Seed 123 Mean Width | Frac > 25% Tile Area | Slits Filtered |
|---|:---:|:---:|:---:|:---:|:---:|
| Arm_A | 50.2 px | 46.6 px | 49.7 px | 0.00% | 869 |
| Arm_B | 50.9 px | 46.4 px | 49.4 px | 0.00% | 1057 |
| Arm_C | 50.9 px | 46.4 px | 49.4 px | 0.00% | 1057 |
| Arm_D | 50.9 px | 46.4 px | 49.4 px | 0.00% | 1060 |
| Arm_E | 50.4 px | 46.6 px | 49.7 px | 0.00% | 890 |

---
### Auditor Summary:
The empirical analysis establishes that the comparison between SafeAdapt (Arm E) and Frozen Source (Arm A) yields decision `EQUIVALENT_STRICT_MARGIN`. The complete experimental artifacts and audit traces are packaged in `utta_med_h3_2_r3a_target_adaptation.zip` for transparent external peer review.
