# UTTA-Med 2.0: Comprehensive Statistical Analysis & Cluster Verification

**Document Type:** Formal Statistical Hypothesis Testing, Clustered Bootstrap, and Multi-Level Variance Audit  
**Auditor:** Senior ML Research Engineer & Reproducibility Auditor  
**Date:** October 22, 2026  
**Primary Dataset:** Camelyon17-WILDS Center 2 (Target Test Cohort, 85,054 patches, 10 WSIs, 9 Patients)  
**Seeds Evaluated:** 42, 123, 2024, 7, 99  

---

## 1. Executive Summary of Statistical Truth

This document establishes the definitive, peer-review-defensible statistical evaluation of test-time adaptation methods in UTTA-Med.

### Primary Conclusions:
1. **Confidence Gating is the Superior Benchmark:**  
   Single-pass Confidence Gating (A1) statistically significantly outperforms 20-pass Original UTTA-Med (A2) in discriminative ranking (AUROC: $+0.00047, p = 0.0161$), classification balance (Binary F1: $+0.00132, p = 0.0062$; Macro F1: $+0.00103, p = 0.0050$), and patient safety (Harm Rate: $-0.0268\%, p = 0.0155$).
2. **Absolute vs. Relative Improvement Distinction:**  
   The absolute empirical F1 gain of Confidence Gating over UTTA-Med is $+0.0013$ ($0.8310$ vs. $0.8296$). Because within-seed paired variance is exceptionally small ($s_D = 0.00056$), the paired-differences Cohen's $d_z$ is **$+2.36$** (Extremely Large). However, the pooled marginal effect size across independent patients is $d_s = \mathbf{+0.041}$ (Small). Both must be reported to avoid misleading claims.
3. **Clustering Bounds:**  
   The evaluation cohort contains 10 whole-slide images from **9 distinct patients** (Patient 46 contributed both Slides 25 and 26). Treating patches as independent observations deflates standard errors by over $200\times$. Genuine patient-clustered bootstrap intervals demonstrate wide biological variance across patients ($[0.77, 0.97]$ for AUROC).

---

## 2. Paired Hypothesis Tests Across 5 Independent Training Seeds

All hypothesis tests were computed across the 5 independent model training seeds using genuine patch predictions from `results/data/wsi_bootstrap/`.

### 2.1 Confidence Gating (A1) vs. Original UTTA-Med (A2)

$$\Delta = \text{Metric}_{\text{A1}} - \text{Metric}_{\text{A2}}$$

| Metric | Mean A1 | Mean A2 | Paired Mean Shift ($\bar{D}$) | Within-Pair SD ($s_D$) | Paired $t$-stat ($df=4$) | Exact $p$-value | Paired $d_z$ | Marginal $d_s$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | $0.9355$ | $0.9351$ | **+0.000469** | $0.000262$ | $+4.0052$ | **$p = 0.01606$** | $+1.791$ | $+0.021$ | **Statistically Significant ($p < 0.05$)** |
| **Binary F1** | $0.8310$ | $0.8296$ | **+0.001322** | $0.000561$ | $+5.2723$ | **$p = 0.00620$** | $+2.358$ | $+0.041$ | **Statistically Significant ($p < 0.01$)** |
| **Macro F1** | $0.8515$ | $0.8505$ | **+0.001028** | $0.000410$ | $+5.6112$ | **$p = 0.00496$** | $+2.509$ | $+0.042$ | **Statistically Significant ($p < 0.01$)** |
| **Accuracy** | $0.8545$ | $0.8535$ | **+0.000958** | $0.000431$ | $+4.9702$ | **$p = 0.00768$** | $+2.223$ | $+0.042$ | **Statistically Significant ($p < 0.01$)** |
| **Harm Rate** | $0.585\%$ | $0.612\%$ | **-0.0268%** | $0.0148\%$ | $-4.0470$ | **$p = 0.01551$** | $-1.810$ | $-0.116$ | **Statistically Significant ($p < 0.05$)** |

*Consistency Across Seeds:*  
In **100% of seeds (5/5)**, Confidence Gating achieves strictly higher F1, higher AUROC, and lower harm than Original UTTA-Med.

---

### 2.2 Adaptation Methods vs. Frozen Source Baseline (A0)

| Contrast | Metric | Source (A0) Mean | Method Mean | Mean Shift ($\bar{D}$) | Paired $t$-stat | Two-Tailed $p$-value | Clinical Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Confidence vs. Source** | AUROC | $0.9297$ | $0.9355$ | $+0.00583$ | $+0.8891$ | $p = 0.4242$ | Non-significant ranking shift |
| | Binary F1 | $0.8069$ | $0.8310$ | **+0.02411** | **+5.4135** | **$p = 0.0056$** | **Dependable positive adaptation ($+2.41\%$)** |
| | Harm Rate | $0.000\%$ | $0.585\%$ | $+0.00585$ | $+5.8311$ | $p = 0.0043$ | Minor harm incurred ($<0.6\%$) |
| **UTTA-Med vs. Source** | AUROC | $0.9297$ | $0.9351$ | $+0.00536$ | $+0.8217$ | $p = 0.4574$ | Non-significant ranking shift |
| | Binary F1 | $0.8069$ | $0.8296$ | **+0.02279** | **+4.9521** | **$p = 0.0078$** | Positive adaptation, but inferior to A1 |
| | Harm Rate | $0.000\%$ | $0.612\%$ | $+0.00612$ | $+5.7303$ | $p = 0.0046$ | Higher harm than Confidence Gating |
| **Tent vs. Source** | AUROC | $0.9297$ | $0.6981$ | **-0.23157** | $-1.5540$ | $p = 0.1950$ | Severe instability; variance exploded |
| | Binary F1 | $0.8069$ | $0.5057$ | **-0.30120** | $-1.6210$ | $p = 0.1790$ | Catastrophic collapse on Seeds 7 & 99 |
| | Harm Rate | $0.000\%$ | $15.482\%$ | **+0.15482** | $+2.3580$ | $p = 0.0779$ | Unacceptable clinical harm ($>30\%$) |

---

## 3. Mathematical Clarification of Effect Size Definitions

The manuscript must explicitly declare which Cohen's $d$ equation is reported:

### 1. Paired Differences Effect Size ($d_z$):
$$d_z = \frac{\bar{D}}{s_D} = \frac{0.001322}{0.000561} = \mathbf{2.3578}$$
This indicates that the observed improvement is **$2.36$ standard deviations above zero variance in paired comparisons**.

### 2. Pooled Standardized Mean Difference ($d_s$):
$$d_s = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2 + s_2^2}{2}}} = \frac{0.831022 - 0.829700}{\sqrt{\frac{0.0321^2 + 0.0325^2}{2}}} = \mathbf{0.0409}$$
This indicates that the absolute shift is modest ($0.04$ pooled standard deviations) relative to patient-to-patient variance.

---

## 4. Multi-Level Patient Clustered Bootstrap Analysis

### 4.1 Resampling Protocol
* **Resampling Unit:** Patient level ($N=9$).
* **Cluster Mapping:**
  - Cluster 1: Patient 40 (Slide 20, 3,810 patches)
  - Cluster 2: Patient 41 (Slide 21, 3,694 patches)
  - Cluster 3: Patient 42 (Slide 22, 7,210 patches)
  - Cluster 4: Patient 44 (Slide 23, 5,288 patches)
  - Cluster 5: Patient 45 (Slide 24, 7,727 patches)
  - **Cluster 6: Patient 46 (Slides 25 & 26, 8,149 patches combined)**
  - Cluster 7: Patient 48 (Slide 27, 4,556 patches)
  - Cluster 8: Patient 51 (Slide 28, 31,878 patches)
  - Cluster 9: Patient 52 (Slide 29, 12,742 patches)
* **Replications:** 10,000 paired resamples with replacement.

### 4.2 Empirical Confidence Intervals
From 10,000 patient-clustered resamples:
* **Frozen Source AUROC:** Mean = $0.9297$, 95% CI: $[0.827, 0.964]$
* **Confidence Gating AUROC:** Mean = $0.9355$, 95% CI: $[0.770, 0.971]$
* **Original UTTA-Med AUROC:** Mean = $0.9351$, 95% CI: $[0.769, 0.971]$
* **Paired Shift ($\Delta \text{AUROC}$ Confidence vs. UTTA-Med):**
  - Mean shift = $+0.00047$, 95% CI: $[+0.00012, +0.00084]$ (Strictly excludes zero).
* **Paired Shift ($\Delta F_1$ Confidence vs. UTTA-Med):**
  - Mean shift = $+0.00132$, 95% CI: $[+0.00062, +0.00201]$ (Strictly excludes zero).

---

## 5. Methodological Guidance for Manuscript Authors

1. **Do not claim training seeds represent patient cohorts:** Seeds $42, 123, 2024, 7, 99$ test optimizer and parameter initialization stability. They do not increase the biological sample size beyond $N=9$ patients.
2. **Report $d_z$ alongside $d_s$:** Explicitly state: *"Confidence gating improves binary F1 by $+0.00132$ ($p = 0.0062$, paired $d_z = 2.36$, pooled $d_s = 0.041$)."*
3. **Acknowledge finite cohort clustering:** Disclose that Patients 51 and 52 constitute $52.5\%$ of the patch volume, explaining the wide clustered confidence intervals.
