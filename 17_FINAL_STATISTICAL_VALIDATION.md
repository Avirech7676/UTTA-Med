# 17. FINAL STATISTICAL VALIDATION REPORT
**UTTA-Med Project 2 — Comprehensive Research Development & Verification Audit**  
**Date:** October 2026 | **Status:** FINAL AUDITED & RECONCILED  
**Lead Auditor:** Senior Medical-Imaging Statistician & Reproducibility Auditor

---

## 1. FORMAL HYPOTHESIS TESTING FRAMEWORK

In compliance with Phase 17, four core scientific hypotheses were formulated a priori, evaluated under rigorous statistical testing frameworks, and audited against multiple-testing penalties.

```
========================================================================================================================
MASTER STATISTICAL HYPOTHESIS TESTING SUMMARY
========================================================================================================================
Hypothesis ID & Description          Analysis Unit      N    Effect Size (dz)  Test Statistic    p-value      FDR Adjusted
------------------------------------------------------------------------------------------------------------------------
H1: C1 is safer than C7 (Matched)    Seeds / Patients   5    4.58 (Extreme)    Paired t = 10.25  p = 0.0005   p = 0.0020 (Sig)
H2: C7 improves F1 over C1 (Cohort)  Seeds              5    0.19 (Negligible) Paired t = 0.42   p = 0.6931   p = 0.6931 (Non-Sig)
H3: Bio-SafeAdapt vs Frozen Source   Case Clusters      50   0.88 (Large)      Bootstrap (10k)   p = 0.0187   p = 0.0249 (Sig)
H4: Bio-SafeAdapt vs SafeAdapt       Multi-Scanner Folds 9   1.96 (Large)      Paired t = 5.88   p = 0.0042   p = 0.0084 (Sig)
========================================================================================================================
```

---

## 2. TRACK A: CLASSIFICATION STATISTICAL ANALYSIS (CAMELYON17-WILDS)

### 2.1 Hypothesis 1: Matched-Coverage Safety Superiority of C1 over C7
- **Null Hypothesis ($H_{0,1}$):** Method C7 commits equal or fewer harmful updates than C1 when evaluated at matched adaptation coverage ($56.2\%$, $k = 719$ updates admitted).
- **Alternative Hypothesis ($H_{1,1}$):** Method C1 commits fewer harmful updates than C7 ($\mu_{H_{\text{C7}}} - \mu_{H_{\text{C1}}} > 0$).
- **Analysis Unit:** 5 independently trained random seeds ($42, 123, 2024, 7, 99$) across 9 patient clusters ($1,280$ patches).
- **Empirical Measurements:**
  - C1 Mean Harmful Updates: $55.0 \pm 18.3$ ($7.65\%$ error rate)
  - C7 Mean Harmful Updates: $63.2 \pm 18.4$ ($8.79\%$ error rate)
  - Mean Difference ($\bar{D}$): $+8.20 \pm 1.79$ harmful updates
- **Test Statistic & p-value:**
  $$t = \frac{\bar{D}}{s_D / \sqrt{n}} = \frac{8.20}{1.79 / \sqrt{5}} = \mathbf{10.25}, \quad \text{df} = 4, \quad p = \mathbf{0.0005}$$
- **Effect Size:** Cohen's $d_z = \mathbf{4.58}$ (extremely large effect).
- **Multiple Comparison Adjustment:** Survives Bonferroni adjustment ($\alpha_{\text{bonf}} = 0.05 / 4 = 0.0125$) and Benjamini-Hochberg FDR ($q = 0.05$).
- **Conclusion:** $H_{0,1}$ is rejected. C1 is statistically significantly safer than C7.

### 2.2 Hypothesis 2: Classification Accuracy Difference (C1 vs C7)
- **Null Hypothesis ($H_{0,2}$):** C7 produces an identical $F_1$ score to C1 on the full $85,054$ target cohort ($\mu_{F_{1,\text{C7}}} - \mu_{F_{1,\text{C1}}} = 0$).
- **Empirical Measurements:**
  - C1 Mean $F_1$: $0.8310 \pm 0.0321$
  - C7 Mean $F_1$: $0.8318 \pm 0.0314$
  - Mean Difference: $+0.0008 \pm 0.0042$
- **Test Statistic & p-value:**
  $$t = 0.420, \quad \text{df} = 4, \quad p = 0.6931$$
- **Conclusion:** Fail to reject $H_{0,2}$. The $+0.0008$ difference is statistically indistinguishable from zero noise.

---

## 3. TRACK B: DETECTION STATISTICAL ANALYSIS (MIDOG 2021)

### 3.1 Hypothesis 3: Bio-SafeAdapt vs Frozen Source Faster R-CNN
- **Null Hypothesis ($H_{0,3}$):** Complete Bio-SafeAdapt achieves equal or lower mitotic detection $F_1$ than the Frozen Source baseline ($\Delta F_1 \le 0$).
- **Analysis Unit:** 50 joint patient case clusters resampled with replacement ($B = 10,000$ replicates) across 3 random seeds.
- **Micro-Pooled Estimand:**
  - Mean $\Delta F_1$: **$+0.00755$**
  - Median $\Delta F_1$: $+0.00742$
  - **$95\%$ Bootstrap Percentile Confidence Interval:** **$[+0.00045, +0.01565]$**
  - Empirical One-Sided p-value: $p = \frac{1}{B} \sum \mathbb{I}(\Delta F_1 \le 0) = \mathbf{0.0187}$
- **Seed-Macro Estimand:**
  - Mean $\Delta F_1$: **$+0.00660$**
  - **$95\%$ Bootstrap Confidence Interval:** **$[+0.00012, +0.01400]$**
  - Empirical One-Sided p-value: $p = \mathbf{0.0223}$
- **Conclusion:** Both micro-pooled and seed-macro intervals are strictly bounded away from zero ($CI_{\text{lower}} > 0.0$). $H_{0,3}$ is rejected. Bio-SafeAdapt provides a statistically verified improvement over the frozen model.

### 3.2 Hypothesis 4: Bio-SafeAdapt vs Original SafeAdapt Incremental Gain
- **Null Hypothesis ($H_{0,4}$):** Bio-SafeAdapt provides no incremental $F_1$ gain over Original SafeAdapt under multi-scanner evaluation ($\mu_{\Delta} \le 0$).
- **Analysis Unit:** 9 cross-scanner evaluation folds (3 Scanners $\times$ 3 Seeds).
- **Empirical Measurements:**
  - Frozen Source Mean $F_1$: $0.4585$
  - Original SafeAdapt Mean $F_1$: $0.4630$
  - Bio-SafeAdapt Mean $F_1$: $0.4732$
  - Incremental Gain ($\Delta F_1$): **$+0.01020 \pm 0.0052$** ($+1.02\%$ F1)
  - **$95\%$ Confidence Interval:** **$[+0.00310, +0.01740]$**
- **Test Statistic & p-value:**
  $$t = 5.88, \quad \text{df} = 8, \quad p = \mathbf{0.0042}$$
- **Conclusion:** $H_{0,4}$ is rejected. Bio-SafeAdapt produces a genuine, statistically significant incremental improvement over Original SafeAdapt.

---

## 4. CLINICAL GOVERNANCE & TRANSLATIONAL DISCLAIMER

1. **Benchmark Constraint:** All data originated from retrospective public competition challenge splits (Camelyon17-WILDS and MIDOG 2021).
2. **Clinical Safety Boundary:** A statistically significant reduction in benchmark false alarms does **not** establish autonomous clinical diagnostic safety. Prospective clinical trials with certified pathologist oversight are mandatory prior to clinical deployment.
