# UTTA-Med: Authoritative Consolidated Results Tables

**Evaluation Lifecycle:** UTTA-Med v1 to v2 Stage E2  
**Validation Integrity:** 100% Machine-Validated against Bit-Exact JSON & CSV Artifacts  
**Authoritative Hash Audit:** Script `scripts/generate_all_final_tables.py` verified against test suite (99 passed)

---

## Overview

This document presents the eight authoritative, validated tables consolidating the findings of the UTTA-Med investigation. Metrics are partitioned between historical multi-seed benchmarks (v1), developmental gating benchmarks on Hospital 1 (v2 Stage D2 / E1), and frozen external validation on Hospital 2 (v2 Stage E2).

---

## Table 1: Historical v1 Multi-Seed Performance (Camelyon17-WILDS)

Aggregated results over five random source seeds (7, 42, 99, 123, 2024). Source checkpoint ResNet-18. Evaluation conducted under the v1 evaluation protocol across test centers.

| Method | AUROC (Mean ± SD) | F1 (Mean ± SD) | ECE (Mean ± SD) | Harm Rate (Mean ± SD) | Patch Coverage (Mean ± SD) |
|:---|:---|:---|:---|:---|:---|
| **Source-Only (Frozen)** | 0.930 ± 0.009 | 0.807 ± 0.025 | 0.128 ± 0.026 | — | — |
| **Tent (Full TTA)** | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.284 ± 0.179 | 18.7% ± 17.9% | 100.0% ± 0.0% |
| **EATA** | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.152 ± 0.024 | 2.5% ± 1.5% | 88.5% ± 3.4% |
| **Random-Gated TTA** | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.298 ± 0.166 | 19.4% ± 17.0% | 70.0% ± 0.1% |
| **Confidence-Gated TTA** | 0.936 ± 0.022 | 0.831 ± 0.032 | 0.130 ± 0.030 | 0.7% ± 0.3% | 58.1% ± 4.7% |
| **UTTA-Med (v1 Hybrid)** | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.131 ± 0.031 | 0.7% ± 0.3% | 57.5% ± 4.7% |

*Data Source: `results/tables/table_main_results_final.csv` (SHA-256: `320df9cd65e84be39a4b3bbb5f4408a47d5c932bbee037dab9f25006a66f2243`)*

---

## Table 2: Historical v1 Paired Statistical Comparisons

Paired two-tailed t-tests across 5 random seeds comparing UTTA-Med against baseline methods. Holm-Bonferroni correction applied for family-wise error rate control.

| Contrast | Mean $\Delta$ | 95% Confidence Interval | Nominal $p$-value | Holm-Adjusted $p$ | Cohen's $d$ | Wins ($N_+$) | Losses ($N_-$) |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **UTTA vs Source (AUROC)** | +0.0058 | [-0.0121, +0.0237] | 0.4199 | 0.8397 | 0.402 | 3 | 2 |
| **UTTA vs Tent (AUROC)** | +0.2373 | [-0.1411, +0.6158] | 0.1566 | 0.5396 | 0.779 | 5 | 0 |
| **UTTA vs EATA (AUROC)** | +0.0145 | [+0.0019, +0.0271] | 0.0330 | 0.1979 | 1.430 | 5 | 0 |
| **UTTA vs Random (AUROC)** | +0.2388 | [-0.1158, +0.5934] | 0.1349 | 0.5396 | 0.836 | 5 | 0 |
| **UTTA vs Conf-Gated (AUROC)** | -0.00005 | [-0.0008, +0.0007] | 0.8688 | 0.8688 | -0.079 | 1 | 4 |
| **UTTA vs Source (F1)** | +0.0231 | [+0.0104, +0.0358] | 0.0072 | 0.0505 | 2.260 | 5 | 0 |
| **UTTA vs Conf-Gated (F1)** | -0.0010 | [-0.0015, -0.0005] | 0.0054 | **0.0430\*** | -2.454 | 0 | 5 |
| **UTTA vs Tent (Harm Rate)** | -0.1802 | [-0.3994, +0.0390] | 0.0846 | 0.4230 | -1.021 | 0 | 5 |

*\* Statistically detectable at $\alpha = 0.05$ under Holm adjustment ($\Delta = -0.0010, p = 0.0430$). The large effect size ($d = -2.454$) reflects exceptional cross-seed consistency despite the absolute difference being practically tiny ($\sim 0.001$ F1). This represents a small statistically detectable difference rather than meaningful clinical inferiority.*  
*Data Source: `results/tables/table_paired_statistics_final.csv` (SHA-256: `b7510afe90e26e6e21c0f9ce87fcb1f0415ee358a60671f5ba5b37eb1d11f2f6`)*


---

## Table 3: Stage D2 Uncertainty Gating Benchmark (Hospital 1 Development Cohort)

Natural policy evaluation across 320 patches from Hospital 1 (WSIs 11, 13, 15, 17, 19). Evaluates accuracy, calibration, and computational burden across candidate gating functions.

| Gating Policy | AUROC | AUPRC | F1 Score | ECE | Brier Score | Patch Coverage | MC Passes | Total Latency |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Frozen Source** | 0.9736 | 0.9542 | 0.8632 | 0.0786 | 0.082867 | 0.0% | 0 | 48.31s |
| **Confidence-Only ($\tau=0.90$)** | 0.9738 | 0.9545 | 0.8596 | 0.0789 | 0.082927 | 87.5% | 0 | 168.73s |
| **MC-Only ($K=20, \tau_u=8.88\times 10^{-6}$)** | 0.9737 | 0.9544 | 0.8596 | 0.0846 | 0.083038 | 68.1% | 200 | 598.21s |
| **Hybrid Policy ($\tau=0.65/0.95, K=20$)** | 0.9738 | 0.9545 | 0.8632 | 0.0788 | 0.082953 | 83.8% | 200 | 182.17s |

*Key Findings: MC-dropout predictive variance provided limited incremental patch-selection information beyond confidence in this evaluated setting ($\rho = -0.9917$). MC-only forward passes take $3.55\times$ longer than confidence gating while providing zero incremental discriminatory accuracy. Note that because patches are clustered within five WSIs, pooled patch-level significance values are not treated as independent inferential evidence.*  
*Data Source: `results/v2/stage_d2/stage_d2_benchmark_results.json` (SHA-256: `143ca46038968393b160ed8a7bf5c013e0c845d30c9e95b3367be33fecb02b7c`)*


---

## Table 4: Stage E1 Four-Arm Streaming Evaluation (Hospital 1 Development Cohort)

Pre-adaptation streaming evaluation over 320 patches (64 patches/WSI, 5 slides). Model updates occur after prediction on the current batch; full model, optimizer, and monitor reset at each WSI boundary.

| Arm ID | Architectural Arm Description | Macro AUROC (Valid Slides) | Pooled AUROC | Pooled AUPRC | Pooled F1 Score | Expected Calibration Error (ECE) | Brier Score | Runtime (s) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Arm A** | Frozen Source Baseline | 0.9706 | 0.9736 | 0.9542 | 0.8632 | 0.0786 | 0.082867 | 30.13s |
| **Arm B** | Confidence TTA (Unsafeguarded) | 0.9706 | 0.9736 | 0.9543 | 0.8632 | 0.0786 | 0.082858 | 89.42s |
| **Arm C** | Stage B SafeAdapt (Bounds & Numerical) | 0.9706 | 0.9736 | 0.9543 | 0.8632 | 0.0786 | 0.082858 | 123.40s |
| **Arm D** | Stage C2 Active SafeAdapt (Dual Safeguards) | 0.9706 | 0.9736 | 0.9543 | 0.8632 | 0.0786 | 0.082858 | 152.00s |

*Note: Macro AUROC is computed across the 4 slides containing both positive and negative tissue (Slide 15 is purely normal tissue).*  
*Data Source: `results/v2/stage_e1/stage_e1_evaluation_results.json` (SHA-256: `ab060f1ec9e9d725e667272e801488739d54a153f6ddaa75dfaebae611a726fc`)*

---

## Table 5: Stage E2 Four-Arm Target Evaluation (Hospital 2 External Target Cohort)

Frozen pre-adaptation evaluation over 1,280 patches across 10 unseen whole-slide images (WSIs 20–29).

| Arm ID | Experimental Arm Description | Macro AUROC (Valid Slides) | Pooled AUROC | Pooled AUPRC | Pooled F1 Score | Expected Calibration Error (ECE) | Brier Score | Runtime (s) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **Arm A** | **Frozen Source Baseline** | **0.9308** | **0.9364** | **0.8897** | **0.7840** | **0.0650** | **0.091709** | 223.85s |
| **Arm B** | Confidence TTA (Unsafeguarded) | 0.9299 | 0.9360 | 0.8891 | 0.7822 | 0.0655 | 0.092051 | 354.10s |
| **Arm C** | Stage B SafeAdapt (Bounds & Numerical) | 0.9300 | 0.9359 | 0.8891 | 0.7822 | 0.0655 | 0.092053 | 389.90s |
| **Arm D** | Stage C2 Active SafeAdapt (Dual Safeguards) | 0.9300 | 0.9359 | 0.8891 | 0.7822 | 0.0655 | 0.092053 | 495.85s |

*Primary Target Finding: Test-time adaptation resulted in negative adaptation transfer across all evaluation metrics. Frozen source is the strictly superior model on Hospital 2.*  
*Data Source: `results/v2/stage_e2/stage_e2_target_results.json` (SHA-256: `f5c90277991999ecc6ae5a63fa3532d74fb5a14b417207d4071efd1533ac1ff4`)*

---

## Table 6: Per-WSI Granular Performance on Hospital 2 (WSIs 20–29)

Disaggregated evaluation by slide identifier across Arms A, B, and D ($N = 128$ patches per slide). Slides 20 and 21 are single-class normal tissue.

| Whole Slide Image | Total Patches ($N$) | Arm A AUROC | Arm B AUROC | Arm D AUROC | Arm A F1 | Arm B F1 | Arm D F1 | Arm A Brier | Arm B Brier | Arm D Brier |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `slide_020` | 128 | N/A (Single-class) | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0136 | 0.0135 | 0.0135 |
| `slide_021` | 128 | N/A (Single-class) | N/A | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0544 | 0.0538 | 0.0538 |
| `slide_022` | 128 | 0.9655 | 0.9655 | 0.9655 | 0.7742 | 0.7742 | 0.7742 | 0.1683 | 0.1694 | 0.1694 |
| `slide_023` | 128 | 0.9914 | 0.9914 | 0.9914 | 0.9353 | 0.9353 | 0.9353 | 0.0527 | 0.0527 | 0.0526 |
| `slide_024` | 128 | 0.7707 | 0.7707 | 0.7707 | 0.0000 | 0.0000 | 0.0000 | 0.0802 | 0.0800 | 0.0800 |
| `slide_025` | 128 | 0.9861 | 0.9854 | 0.9854 | 0.9355 | 0.9355 | 0.9355 | 0.0318 | 0.0318 | 0.0318 |
| `slide_026` | 128 | 0.8376 | 0.8354 | 0.8359 | 0.3571 | 0.3571 | 0.3571 | 0.1121 | 0.1124 | 0.1124 |
| `slide_027` | 128 | 0.9813 | 0.9787 | 0.9787 | 0.5000 | 0.5000 | 0.5000 | 0.0221 | 0.0221 | 0.0221 |
| `slide_028` | 128 | 0.9843 | 0.9843 | 0.9843 | 0.9619 | 0.9619 | 0.9619 | 0.0572 | 0.0570 | 0.0570 |
| `slide_029` | 128 | **0.9293** | 0.9281 | 0.9281 | **0.3659** | 0.3457 | 0.3457 | **0.3247** | 0.3279 | 0.3279 |

*Analysis: Slide 29's performance degradation is consistent with confirmation bias under domain shift. When high-confidence false positive patches were accepted by confidence gating, slide AUROC dropped from 0.9293 to 0.9281 and F1 dropped from 0.3659 to 0.3457.*  
*Data Source: `results/v2/stage_e2/stage_e2_target_results.json`*


---

## Table 7: Safeguard Interventions and Numerical Telemetry

Telemetry collected across all 50 streaming evaluation batches (10 in E1, 40 in E2).

| Evaluation Cohort | Arm | Selected Patches | Updates Attempted | Updates Accepted | Updates Rejected | Eligible Decisions | Alarms Triggered | Rollbacks Executed | Non-Finite Failures | Bound Clamps | Max Observed Displacement ($L_\infty$) |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **E1 (Hospital 1)** | Arm A | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00$ |
| **E1 (Hospital 1)** | Arm B | 280 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | $2.00 \times 10^{-5}$ |
| **E1 (Hospital 1)** | Arm C | 280 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | $2.00 \times 10^{-5}$ |
| **E1 (Hospital 1)** | Arm D | 280 | 10 | 10 | 0 | 0 | 0 | 0 | 0 | 0 | $2.00 \times 10^{-5}$ |
| **E2 (Hospital 2)** | Arm A | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | $0.00$ |
| **E2 (Hospital 2)** | Arm B | 1,069 | 40 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | $4.01 \times 10^{-5}$ |
| **E2 (Hospital 2)** | Arm C | 1,069 | 40 | 40 | 0 | 0 | 0 | 0 | 0 | 0 | $4.01 \times 10^{-5}$ |
| **E2 (Hospital 2)** | Arm D | 1,069 | 40 | 40 | 0 | **40** | **0** | **0** | **0** | **0** | **$4.01 \times 10^{-5}$** |

*Interpretation: The maximum parameter displacement ($L_\infty = 4.01 \times 10^{-5}$) utilized only 0.080% of the allowed bound ($0.05$), proving that micro-step updates under $\eta=10^{-4}$ with 4 updates per slide remained within a tight local perturbation regime. Zero false alarms occurred across 40 eligible monitoring evaluations.*

---

## Table 8: Runtime Latency and Computational Overhead

Comparative wall-clock runtime across methods, environments, and cohort sizes.

| Evaluation Benchmark & Environment | Frozen Source Baseline | Confidence-Only Adaptation | Hybrid Gating Adaptation | MC-Dropout Only Adaptation | Active SafeAdapt (Dual Safeguards) |
|:---|:---|:---|:---|:---|:---|
| **Stage D2 Benchmark** (Hospital 1, 320 patches, 20 MC samples) | 72.09s (1.00×) | 204.88s (2.84×) | 201.60s (2.80×) | 624.86s (8.67×) | — |
| **Stage E1 Streaming** (Hospital 1, 320 patches, 5 WSIs) | 30.13s (1.00×) | 89.42s (2.97×) | — | — | 152.00s (5.04×) |
| **Stage E2 Streaming** (Hospital 2, 1,280 patches, 10 WSIs) | 223.85s (1.00×) | 354.10s (1.58×) | — | — | 495.85s (2.22×) |

*Computational Overhead Summary: Active SafeAdapt adds approximately +40% runtime latency over unsafeguarded confidence TTA to maintain rolling window state, evaluate collapse metrics, and verify parameter displacement bounds. In contrast, MC dropout multiplies total inference latency by >3×.*
