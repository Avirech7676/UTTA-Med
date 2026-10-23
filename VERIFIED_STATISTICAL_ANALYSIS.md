# UTTA-Med 2.0: Verified Statistical Analysis & Hypothesis Testing Audit

**Document Type:** Formal Statistical Audit & Methodology Verification  
**Auditor:** Independent Machine Learning Research Auditor & Reproducibility Engineer  
**Date:** October 22, 2026  
**Target Evaluation Cohort:** Camelyon17-WILDS Center 2 (Target Test Domain, 85,054 patches, 10 Whole-Slide Images from 10 distinct patients)  
**Seeds Evaluated:** 42, 123, 2024, 7, 99  
**Audit Status:** Complete & Verified  

---

## 1. Executive Summary: Statistical Reality vs. Stated Claims

This audit independently scrutinizes the statistical methodology, paired hypothesis tests, effect sizes, and bootstrap resampling protocols utilized in the UTTA-Med 2.0 evaluation.

### Key Audit Findings:
1. **The F1 $p$-value Discrepancy ($p = 0.0961$):**  
   The widely cited $p = 0.0961$ in prior reports was derived **not** from real histological patch streaming, but from a paired $t$-test across $N=5$ synthetic random perturbations of A9 vs. A1 in `UTTA_MED_V2_RESULTS.csv`. Because the synthetic generator produced a negative difference on Seed 42 ($-0.0029$) alongside four positive differences ($+0.0074, +0.0090, +0.0099, +0.0025$), the standard error was inflated ($s_D = 0.005345$ vs. $\bar{D} = 0.005180$), resulting in $t = 2.1671$ and a two-tailed $p = 0.0961$ (degrees of freedom $= 4$). The previous report improperly claimed statistical significance by cherry-picking an unadjusted bootstrap percentile interval that excluded zero.
2. **Empirical Truth Across All 5 Seeds:**  
   In the genuine empirical data (85,054 patches evaluated across all 5 seeds in `results/data/wsi_bootstrap/`), **Confidence Gating (A1) strictly and statistically significantly outperforms Original UTTA-Med (A2)**:
   - **Binary F1:** $\Delta = +0.001322$, $t = +5.2723$, $p = 0.00620$ ($p < 0.01$).
   - **Macro F1:** $\Delta = +0.001028$, $t = +5.6112$, $p = 0.00496$ ($p < 0.01$).
   - **AUROC:** $\Delta = +0.000469$, $t = +4.0052$, $p = 0.01606$ ($p < 0.05$).
   - **Harm Reduction:** $\Delta = -0.000268$, $t = -4.0470$, $p = 0.01551$ ($p < 0.05$).
   - Confidence Gating achieves this superior performance with **$20\times$ lower forward pass complexity** (1 pass vs. 20 passes).
3. **Resampling Unit & Clustering Limitations:**  
   The target test domain consists of 10 Whole-Slide Images from 10 patients. Resampling patches as independent observations severely artificially deflates standard errors. Genuine slide-level clustered bootstrap demonstrates wide slide-to-slide variance.

---

## 2. Independent Recomputation of Paired Statistical Contrasts

All tests below are evaluated on the authentic 5-seed empirical dataset from `results/data/wsi_bootstrap/`. Each seed represents the exact classification of 85,054 patches from Center 2.

### 2.1 Confidence Gating (A1) vs. Original UTTA-Med (A2)

Confidence Gating employs a deterministic threshold ($\tau_{\text{conf}} = 0.90$) using a single forward pass. Original UTTA-Med uses a 20-pass Monte Carlo Dropout predictive variance gate ($\tau_{\text{unc}} = 8.875 \times 10^{-6}$).

| Metric | Confidence (A1) Mean $\pm$ SD | UTTA-Med (A2) Mean $\pm$ SD | Paired Diff ($\bar{D}$) | Diff SD ($s_D$) | Paired $t$-stat | Degrees of Freedom | Two-Tailed $p$-value | Significance ($\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AUROC** | $0.9355 \pm 0.0218$ | $0.9351 \pm 0.0218$ | **+0.000469** | $0.000262$ | **+4.0052** | 4 | **$p = 0.01606$** | **Significant ($p < 0.05$)** |
| **Binary F1** | $0.8310 \pm 0.0321$ | $0.8296 \pm 0.0325$ | **+0.001322** | $0.000561$ | **+5.2723** | 4 | **$p = 0.00620$** | **Significant ($p < 0.01$)** |
| **Macro F1** | $0.8515 \pm 0.0244$ | $0.8505 \pm 0.0247$ | **+0.001028** | $0.000410$ | **+5.6112** | 4 | **$p = 0.00496$** | **Significant ($p < 0.01$)** |
| **Accuracy** | $0.8545 \pm 0.0227$ | $0.8535 \pm 0.0230$ | **+0.000958** | $0.000431$ | **+4.9702** | 4 | **$p = 0.00768$** | **Significant ($p < 0.01$)** |
| **Harm Rate** | $0.585\% \pm 0.224\%$ | $0.612\% \pm 0.239\%$ | **-0.0268%** | $0.0148\%$ | **-4.0470** | 4 | **$p = 0.01551$** | **Significant ($p < 0.05$)** |

#### Per-Seed Empirical Breakdown ($\text{A1} - \text{A2}$):
* **Seed 42:** $\Delta \text{F1} = +0.000995$, $\Delta \text{AUROC} = +0.000605$, $\Delta \text{Harm} = -0.0071\%$
* **Seed 123:** $\Delta \text{F1} = +0.000963$, $\Delta \text{AUROC} = +0.000054$, $\Delta \text{Harm} = -0.0188\%$
* **Seed 2024:** $\Delta \text{F1} = +0.001036$, $\Delta \text{AUROC} = +0.000720$, $\Delta \text{Harm} = -0.0259\%$
* **Seed 7:** $\Delta \text{F1} = +0.002292$, $\Delta \text{AUROC} = +0.000581$, $\Delta \text{Harm} = -0.0435\%$
* **Seed 99:** $\Delta \text{F1} = +0.001325$, $\Delta \text{AUROC} = +0.000383$, $\Delta \text{Harm} = -0.0388\%$

**Critical Auditor Conclusion:**  
In **5 out of 5 random seeds (100% of trials)**, Confidence Gating achieves strictly higher AUROC, higher Binary F1, higher Macro F1, and lower harm than Original UTTA-Med. There is **zero empirical support** for the hypothesis that 20-pass MC Dropout provides superior adaptation over single-pass confidence gating.

---

### 2.2 Adaptation Methods vs. Frozen Source Baseline (A0)

| Contrast | Metric | Source (A0) Mean | Method Mean | Mean Diff ($\bar{D}$) | Paired $t$-stat | Two-Tailed $p$-value | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Confidence vs. Source** | AUROC | $0.9297$ | $0.9355$ | $+0.005831$ | $+0.8891$ | $p = 0.42416$ | Not Significant ($p > 0.05$) |
| | Binary F1 | $0.8069$ | $0.8310$ | $+0.024107$ | $+5.4135$ | $p = 0.00564$ | **Significant ($p < 0.01$)** |
| | Harm Rate | $0.000\%$ | $0.585\%$ | $+0.005848$ | $+5.8311$ | $p = 0.00431$ | **Significant ($p < 0.01$)** |
| **UTTA-Med vs. Source** | AUROC | $0.9297$ | $0.9351$ | $+0.005362$ | $+0.8217$ | $p = 0.45742$ | Not Significant ($p > 0.05$) |
| | Binary F1 | $0.8069$ | $0.8296$ | $+0.022785$ | $+4.9521$ | $p = 0.00775$ | **Significant ($p < 0.01$)** |
| | Harm Rate | $0.000\%$ | $0.612\%$ | $+0.006116$ | $+5.7303$ | $p = 0.00459$ | **Significant ($p < 0.01$)** |
| **Tent vs. Source** | AUROC | $0.9297$ | $0.6981$ | $-0.231572$ | $-1.5540$ | $p = 0.19500$ | Severe Catastrophic Drops |
| | Binary F1 | $0.8069$ | $0.5057$ | $-0.301198$ | $-1.6210$ | $p = 0.17900$ | Collapse on Seeds 7 & 99 |
| | Harm Rate | $0.000\%$ | $15.482\%$ | $+0.154820$ | $+2.3580$ | $p = 0.07790$ | Catastrophic Harm ($>30\%$) |

*Note on Tent Collapse:*  
On Seeds 7 and 99, unconstrained Tent collapses catastrophically (AUROC drops from $\sim 0.93$ to $0.3429$ and $0.3443$; F1 drops to $0.0544$ and $0.0576$; Harm exceeds $30\%$).

---

## 3. Explicit Definitions and Recomputation of Cohen's $d$

In scientific literature, "Cohen's $d$" is frequently reported ambiguously. There are two distinct, non-interchangeable mathematical definitions in paired experimental designs:

### Definition 1: Paired-Differences Effect Size ($d_z$)
$$d_z = \frac{\bar{D}}{s_D} = \frac{\frac{1}{N}\sum_{i=1}^N (X_{1,i} - X_{2,i})}{\sqrt{\frac{1}{N-1}\sum_{i=1}^N (D_i - \bar{D})^2}}$$
*Interpretation:* Measures the mean paired shift relative to the standard deviation of within-subject differences.

### Definition 2: Standardized Mean Difference using Pooled Variance ($d_s$)
$$d_s = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}, \quad \text{where } s_{\text{pooled}} = \sqrt{\frac{s_1^2 + s_2^2}{2}}$$
*Interpretation:* Measures the absolute separation of the two marginal distributions.

### Comparative Table of Effect Sizes (Confidence Gating vs. Original UTTA-Med)

| Metric | Mean Difference ($\bar{D}$) | Paired $d_z$ ($d_{\text{paired}}$) | Pooled $d_s$ ($d_{\text{pooled}}$) | Audit Commentary |
| :--- | :---: | :---: | :---: | :--- |
| **AUROC** | $+0.000469$ | **$+1.7912$** (Large) | **$+0.0215$** (Negligible) | High consistency across seeds yields large $d_z$, but absolute gain is small relative to seed variance. |
| **Binary F1** | $+0.001322$ | **$+2.3578$** (Very Large) | **$+0.0409$** (Small) | Every seed improves, producing massive paired $t$ and $d_z$, while marginal distributions overlap heavily. |
| **Macro F1** | $+0.001028$ | **$+2.5094$** (Very Large) | **$+0.0419$** (Small) | Identical phenomenon: $d_z$ reflects paired reliability; $d_s$ reflects marginal effect. |
| **Harm Rate** | $-0.000268$ | **$-1.8099$** (Large) | **$-0.1158$** (Small) | Paired harm reduction is statistically dependable across all 5 seeds. |

**Audit Recommendation for IEEE Manuscript:**  
Authors must explicitly declare: *"We report the paired-difference effect size $d_z = \bar{D}/s_D$. Because within-seed variance of adaptation difference is small ($s_D = 0.00056$), $d_z = 2.36$, whereas the pooled marginal effect size is $d_s = 0.041$."* Omitting this distinction constitutes statistical reporting opacity.

---

## 4. Forensic Investigation of the Reported F1 $p$-value ($p = 0.0961$)

In `STATISTICAL_VALIDATION.md` (lines 20–25), the previous report stated:
> *"Pooled F1 Score: Paired Difference (+0.0051), 95% Bootstrap CI: [+0.0007, +0.0090], Paired t-test: p = 0.0961, Cohen's d: 0.969"*

### 4.1 Exact Mathematical Derivation
From `UTTA_MED_V2_RESULTS.csv`, the five reported seed values for A1 and A9 were:
* **A1 (Confidence Gating):** $0.8345, 0.8258, 0.8270, 0.8272, 0.8308 \implies \bar{X}_1 = 0.82906, s_1 = 0.00349$
* **A9 (Complete UTTA-Med 2.0):** $0.8316, 0.8332, 0.8360, 0.8371, 0.8333 \implies \bar{X}_2 = 0.83424, s_2 = 0.00225$

Computing paired differences ($D_i = \text{A9}_i - \text{A1}_i$):
$$D = [-0.0029, +0.0074, +0.0090, +0.0099, +0.0025]$$
* **Sample Mean:** $\bar{D} = +0.005180$
* **Sample Standard Deviation ($N-1=4$):** $s_D = 0.005345$
* **Standard Error:** $\text{SE} = \frac{s_D}{\sqrt{5}} = \frac{0.005345}{2.236068} = 0.002390$
* **Paired $t$-statistic:**
  $$t = \frac{\bar{D}}{\text{SE}} = \frac{0.005180}{0.002390} = 2.1671$$
* **Two-Tailed $p$-value ($df=4$):**
  $$p = 2 \times (1 - F_{\text{Student-}t}(2.1671, df=4)) = \mathbf{0.0961}$$
* **Paired Cohen's $d$ ($d_z$):**
  $$d_z = \frac{0.005180}{0.005345} = \mathbf{0.9692}$$

### 4.2 Why the Results Conflicted and Why It Was Misreported
1. **The Negative Seed:** Seed 42 had a negative difference ($-0.0029$), meaning Confidence Gating beat UTTA-Med 2.0 on Seed 42 in that synthetic run.
2. **High Sample Variance:** Because $s_D = 0.005345 > \bar{D} = 0.005180$, the test lacks statistical power at $N=5$.
3. **Selective Reporting:** The authors observed that the parametric paired $t$-test failed to reach significance ($p = 0.0961 > 0.05$). Rather than accepting the null hypothesis, they executed a non-parametric bootstrap resampling that yielded a positive 95% interval ($[+0.0007, +0.0090]$) and claimed the result was significant ("Non-zero CI ($CI > 0$), Marginal $p$").
4. **Synthetic Nature:** As established in `VERIFIED_GPU_EXECUTION.md`, both sets of numbers were synthetically generated in `run_kaggle.py`. In real empirical data, A9 was never run, and A1 beats A2 with $p = 0.0062$.

---

## 5. Clustered Bootstrapping & Multi-Level Resampling Limitations

### 5.1 Resampling Unit Analysis
In medical whole-slide imaging, observations are structured hierarchically:
$$\text{Patches} \subset \text{Whole-Slide Images (WSIs)} \subset \text{Patients} \subset \text{Medical Centers}$$

In Camelyon17-WILDS Center 2:
* **Total Patches:** 85,054
* **Total Slides:** 10 slides (Slides 20, 21, 22, 23, 24, 25, 26, 27, 28, 29)
* **Total Patients:** Exactly 10 patients (1 slide per patient)

#### Statistical Implications:
1. **Patch-Level Resampling Fallacy:**  
   Treating 85,054 patches as independent identically distributed (i.i.d.) observations yields artificially shrunken standard errors ($\approx 1/\sqrt{85054}$). Slide 28 alone contains 31,878 patches ($37.5\%$ of the cohort), and Slide 29 contains 12,742 patches ($15.0\%$). Together, two slides represent $52.5\%$ of the evaluation set. Any patch-level bootstrap is heavily dominated by these two patients.
2. **Slide-Level / Patient-Level Resampling:**  
   When the resampling unit is the slide/patient ($N=10$), sampling with replacement means high-burden slides (28 and 29) are sampled 0, 1, 2, or 3 times, reflecting true clinical variance.
3. **10-Slide Empirical Confidence Intervals:**  
   From the verified 10-slide bootstrap distribution across all 5 seeds (`results/statistical/wsi_bootstrap/wsi_5seed_combined.json`):
   - **Confidence Gating AUROC 95% CI:** $[0.770, 0.971]$
   - **UTTA-Med AUROC 95% CI:** $[0.769, 0.971]$
   - **Frozen Source AUROC 95% CI:** $[0.827, 0.964]$
   The wide intervals demonstrate that slide-to-slide staining and patient biology introduce substantial variance, further proving that fine differences ($\Delta \text{AUROC} \sim 0.0005$) cannot support claims of clinically game-changing superiority.

---

## 6. Publication Certification Verdict

| Claim Audited | Empirical Evidence Status | Formal Auditor Verdict |
| :--- | :--- | :---: |
| **"UTTA-Med 2.0 achieves statistically significant F1 gain over Confidence Gating ($p < 0.05$)"** | Stated $p = 0.0961$ fails $\alpha=0.05$; based on synthetic CPU data. Real empirical data for A9 does not exist. | **CONTRADICTED** |
| **"Confidence Gating outperforms Original UTTA-Med across seeds"** | Evaluated on 85,054 real patches across 5 seeds: $t=5.27, p=0.0062$. | **VERIFIED** |
| **"UTTA-Med produces lower harm than Confidence Gating"** | Evaluated on 85,054 patches: Confidence gating produces strictly lower harm in all 5 seeds ($t=-4.05, p=0.0155$). | **CONTRADICTED** |
| **"Slide-clustered bootstrap confirms patient-level robustness"** | Resampling $N=10$ slides reveals wide intervals ($[0.77, 0.97]$); claims must disclose cohort clustering bounds. | **PARTIALLY VERIFIED** |
