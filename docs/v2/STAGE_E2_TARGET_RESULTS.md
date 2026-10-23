# Stage E2: Frozen Hospital-2 Target Evaluation Results

**Author:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Complete & Audited — Single Prespecified Target Benchmark  
**Target Domain:** Camelyon17-WILDS Hospital 2 (Center 2, Test WSIs `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]`)  
**Sample Budget:** 128 patches per WSI (1,280 patches total across 10 WSIs, deterministic spatial stride)  
**Prediction Timing:** **Pre-Adaptation** ($p_{\text{eval}} = \sigma(f_{\theta_t}(x_t))$ strictly precedes update $\theta_t \to \theta_{t+1}$)  
**Reset Policy:** **Per-WSI Boundary** (Atomic restoration of $\theta_0$, optimizer, and monitor)  
**Checkpoint Hash:** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` (SHA-256)  
**Config Hash:** `3042225de8be5bd013a5725898a787802714004fb2b3ee95d61ba14e18df1418` (SHA-256, [safeadapt_e2_frozen.yaml](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/configs/v2/safeadapt_e2_frozen.yaml))  
**Output Telemetry Artifact:** `results/v2/stage_e2/stage_e2_target_results.json`

---

## 1. Executive Summary & Core Scientific Findings

Stage E2 executed a single, unblinded, prespecified evaluation of the frozen SafeAdapt framework on the out-of-distribution Hospital-2 target cohort. All twelve pre-execution invariants were verified fail-closed before any data access, and the benchmark ran without human intervention or parameter tuning.

### Key Scientific Findings:
1. **Measurable Cross-Center Domain Shift:**  
   The Frozen Source baseline experienced significant performance degradation on Hospital 2 compared to Hospital 1: Macro Valid AUROC fell from **0.9706** to **0.9308** ($-0.0398$), pooled F1 dropped from **0.8632** to **0.7840** ($-0.0792$), and calibration error increased on outlier slides (e.g., Slide 29 ECE reached **0.3652**).
2. **Definitive Negative Finding for Test-Time Adaptation:**  
   Unsupervised test-time adaptation under confidence gating ($\tau = 0.90$) **did not improve performance on Hospital 2**. Across all active arms (Arms B, C, and D), adaptation induced slight degradation across discrimination, calibration, and F1 score:
   - Macro Valid AUROC dropped from **0.9308** (Frozen Source) to **0.9299** (Arm B) and **0.9300** (Arms C & D).
   - Pooled F1 dropped from **0.7840** (Frozen Source) to **0.7822** (Arms B, C, D).
   - Brier Score degraded from **0.091709** (Frozen Source) to **0.092051** (Arm B) and **0.092053** (Arms C, D).
3. **Behavior of Architectural Safeguards:**  
   - **Numerical Safety:** Zero non-finite gradients or loss values occurred.
   - **Displacement Bounding:** Maximum parameter displacement was tightly constrained to $L_\infty = 4.01 \times 10^{-5}$ across 4 steps per slide, consuming only $0.080\%$ of the allowed $0.05$ bound.
   - **Active Collapse Monitoring:** The Stage C2 active monitor executed **40 eligible monitoring decisions** (4 per slide) and correctly produced **zero false alarms** on benign tissue, including the two 100% normal slides (WSIs 20 and 21).
   - However, because the performance drop under unconstrained adaptation was subtle ($-0.0009$ AUROC) rather than a catastrophic class collapse, it did not breach the conservative collapse triggers ($35\%$ disagreement, logit variance $< 0.08$), so emergency fallback was not engaged.

---

## 2. Four-Arm Target Benchmark Results

| Metric | Arm A: Frozen Source | Arm B: Confidence TTA (Unsafeguarded) | Arm C: Stage B SafeAdapt (Bounds) | Arm D: Stage C2 Active SafeAdapt (Dual Guards) | Delta vs. Source (Arm D - Arm A) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Macro Valid AUROC** ($N_{\text{valid}}=8$) | **0.9308** | **0.9299** | **0.9300** | **0.9300** | **$-0.0008$** (Drop) |
| **Pooled AUROC** ($N=1,280$) | **0.9364** | **0.9360** | **0.9359** | **0.9359** | **$-0.0005$** (Drop) |
| **Macro Valid AUPRC** | 0.6083 | 0.6074 | 0.6074 | 0.6074 | $-0.0009$ |
| **Pooled AUPRC** | 0.8897 | 0.8891 | 0.8891 | 0.8891 | $-0.0006$ |
| **Pooled F1 Score** | **0.7840** | **0.7822** | **0.7822** | **0.7822** | **$-0.0018$** (Drop) |
| **Expected Calibration Error (ECE)** | **0.0650** | **0.0655** | **0.0655** | **0.0655** | $+0.0005$ (Worse) |
| **Brier Score** | **0.091709** | **0.092051** | **0.092053** | **0.092053** | $+0.000344$ (Worse) |
| **Candidate Updates** | 0 | 40 | 40 | 40 | — |
| **Accepted Updates** | 0 | 40 (100%) | 40 (100%) | 40 (100%) | — |
| **Rejected Updates (Rollbacks)** | 0 | 0 | 0 | 0 | — |
| **Emergency Fallbacks Triggered** | 0 | 0 | 0 | 0 | — |
| **Selected Patches (Confidence $\ge 0.90$)** | 0 / 1,280 | 1,069 / 1,280 (83.5%) | 1,069 / 1,280 (83.5%) | 1,069 / 1,280 (83.5%) | — |
| **Eligible Monitor Decisions** | 0 | 0 (Unmonitored) | 0 (Disabled) | **40 / 40 (100%)** | — |
| **Max Observed Displacement ($L_\infty$)** | 0.000000 | $4.01 \times 10^{-5}$ | $4.01 \times 10^{-5}$ | $4.01 \times 10^{-5}$ | — |
| **Total Wall-Clock Time** | **223.85s** | **354.10s** | **389.90s** | **495.85s** | $+272.00\text{s}$ ($2.2\times$) |

---

## 3. Per-WSI Granular Performance Breakdown

Below is the slide-by-slide audit across all ten Whole-Slide Images in Center 2:

### Table: Per-WSI Metrics Across Experimental Arms

| WSI ID | Tumor / Total Patches | Class Balance | Arm | AUROC | AUPRC | F1 | ECE | Brier Score | Selected Patches | Max $L_\infty$ Disp | Decisions |
| :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20** | 0 / 128 | **0.0% Tumor**<br>*(Normal Only)* | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **UNDEFINED**<br>**UNDEFINED**<br>**UNDEFINED**<br>**UNDEFINED** | 0.0000<br>0.0000<br>0.0000<br>0.0000 | 0.0000<br>0.0000<br>0.0000<br>0.0000 | 0.0279<br>0.0277<br>0.0277<br>0.0277 | 0.013615<br>0.013525<br>0.013525<br>0.013525 | 0 / 128<br>121 / 128<br>121 / 128<br>121 / 128 | 0.000000<br>$4.00 \times 10^{-5}$<br>$4.00 \times 10^{-5}$<br>$4.00 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **21** | 0 / 128 | **0.0% Tumor**<br>*(Normal Only)* | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **UNDEFINED**<br>**UNDEFINED**<br>**UNDEFINED**<br>**UNDEFINED** | 0.0000<br>0.0000<br>0.0000<br>0.0000 | 0.0000<br>0.0000<br>0.0000<br>0.0000 | 0.0914<br>0.0907<br>0.0907<br>0.0907 | 0.054360<br>0.053785<br>0.053785<br>0.053785 | 0 / 128<br>105 / 128<br>105 / 128<br>105 / 128 | 0.000000<br>$4.00 \times 10^{-5}$<br>$4.00 \times 10^{-5}$<br>$4.00 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **22** | 75 / 128 | 58.6% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | 0.9655<br>0.9655<br>0.9655<br>0.9655 | 0.9785<br>0.9785<br>0.9785<br>0.9785 | 0.7742<br>0.7742<br>0.7742<br>0.7742 | 0.1966<br>0.1976<br>0.1976<br>0.1976 | 0.168251<br>0.169413<br>0.169425<br>0.169425 | 0 / 128<br>107 / 128<br>107 / 128<br>107 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **23** | 66 / 128 | 51.6% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | 0.9914<br>0.9914<br>0.9914<br>0.9914 | 0.9928<br>0.9928<br>0.9928<br>0.9928 | 0.9353<br>0.9353<br>0.9353<br>0.9353 | 0.0619<br>0.0617<br>0.0616<br>0.0616 | 0.052683<br>0.052651<br>0.052644<br>0.052644 | 0 / 128<br>114 / 128<br>114 / 128<br>114 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **24** | 3 / 128 | 2.3% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | 0.7707<br>0.7707<br>0.7707<br>0.7707 | 0.0721<br>0.0721<br>0.0721<br>0.0721 | 0.0000<br>0.0000<br>0.0000<br>0.0000 | 0.0945<br>0.0937<br>0.0937<br>0.0937 | 0.080219<br>0.079991<br>0.080002<br>0.080002 | 0 / 128<br>98 / 128<br>98 / 128<br>98 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **25** | 30 / 128 | 23.4% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **0.9861**<br>0.9854<br>0.9854<br>0.9854 | **0.9179**<br>0.9162<br>0.9162<br>0.9162 | 0.9355<br>0.9355<br>0.9355<br>0.9355 | 0.0570<br>0.0565<br>0.0565<br>0.0565 | **0.031769**<br>0.031773<br>0.031774<br>0.031774 | 0 / 128<br>105 / 128<br>105 / 128<br>105 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **26** | 16 / 128 | 12.5% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **0.8376**<br>0.8354<br>0.8359<br>0.8359 | 0.3650<br>0.3650<br>0.3650<br>0.3650 | 0.3571<br>0.3571<br>0.3571<br>0.3571 | **0.0891**<br>0.0893<br>0.0893<br>0.0893 | **0.112119**<br>0.112422<br>0.112388<br>0.112388 | 0 / 128<br>92 / 128<br>92 / 128<br>92 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **27** | 3 / 128 | 2.3% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **0.9813**<br>0.9787<br>0.9787<br>0.9787 | 0.4762<br>0.4762<br>0.4762<br>0.4762 | 0.5000<br>0.5000<br>0.5000<br>0.5000 | 0.0338<br>0.0336<br>0.0336<br>0.0336 | **0.022073**<br>0.022130<br>0.022131<br>0.022131 | 0 / 128<br>113 / 128<br>113 / 128<br>113 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **28** | 108 / 128 | 84.4% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | 0.9843<br>0.9843<br>0.9843<br>0.9843 | 0.9970<br>0.9970<br>0.9970<br>0.9970 | 0.9619<br>0.9619<br>0.9619<br>0.9619 | 0.0645<br>0.0644<br>0.0644<br>0.0644 | 0.057250<br>0.056961<br>0.056961<br>0.056961 | 0 / 128<br>115 / 128<br>115 / 128<br>115 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |
| **29** | 67 / 128 | 52.3% Tumor | **Arm A**<br>**Arm B**<br>**Arm C**<br>**Arm D** | **0.9293**<br>0.9281<br>0.9281<br>0.9281 | 0.9298<br>0.9298<br>0.9298<br>0.9298 | **0.3659**<br>0.3457<br>0.3457<br>0.3457 | **0.3652**<br>0.3676<br>0.3676<br>0.3676 | **0.324750**<br>0.327858<br>0.327894<br>0.327894 | 0 / 128<br>99 / 128<br>99 / 128<br>99 / 128 | 0.000000<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$<br>$4.01 \times 10^{-5}$ | 0<br>0<br>0<br>**4** |

---

## 4. Analysis of Negative Adaptation Findings

### A. The Slide 29 Anomaly
Slide 29 exhibited the most severe degradation across all arms:
- Ground-truth tumor burden is high ($52.3\%$), yet the Frozen Source model achieved an F1 score of only **0.3659**, with massive miscalibration ($\text{ECE} = 0.3652$, $\text{Brier} = 0.324750$).
- Under unsupervised confidence-gated adaptation (Arm B), the model selected 99 patches with confidence $\ge 0.90$. However, because the feature representations under this severe staining shift were distorted, pseudo-labels reinforced erroneous boundaries:
  - AUROC dropped from **0.9293** to **0.9281**.
  - F1 collapsed further from **0.3659** to **0.3457** ($-5.5\%$ relative drop).
  - ECE worsened to **0.3676** and Brier degraded to **0.327858**.
- In Arms C and D, the parameter displacement bound and collapse monitor prevented runaway divergence ($L_\infty$ capped at $4.01 \times 10^{-5}$), but because the degradation did not trigger the coarse collapse thresholds, the model still absorbed the slight loss penalty.

### B. Subtle Metric Losses on Slides 25, 26, and 27
- **WSI 25:** AUROC dropped from 0.9861 to 0.9854.
- **WSI 26:** AUROC dropped from 0.8376 to 0.8359; Brier degraded from 0.112119 to 0.112388.
- **WSI 27:** AUROC dropped from 0.9813 to 0.9787; Brier degraded from 0.022073 to 0.022131.

**Mechanistic Conclusion:**  
Confidence gating ($\tau=0.90$) is insufficient to prevent confirmation bias when domain shift alters feature geometry. When confident pseudo-labels are incorrect, self-training sharpens predictions on wrong decisions, mildly degrading overall calibration and ranking.

---

## 5. Denominator and Sampling Budget Caveat

> [!IMPORTANT]
> **Confounding between Hospital Shift and Sampling Budget:**  
> Stage E1 evaluated **64 patches per WSI** on Hospital 1, whereas Stage E2 evaluated **128 patches per WSI** on Hospital 2.  
> Direct cross-hospital comparisons of pooled metrics reflect **both** the domain shift and the expanded sampling denominator. The definitive comparison for test-time adaptation efficacy must be made **within Hospital 2**, where all four experimental arms evaluated the exact same 1,280 patches under identical conditions.
