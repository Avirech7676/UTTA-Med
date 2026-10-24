# Final Statistical Comparison & Attribution Protocol Report

**Project:** UTTA-Med Project 2  
**Document ID:** `FINAL_STATISTICAL_COMPARISON.md`  
**Scope:** Rigorous Statistical Protocol for Evaluating whether Improved UTTA-Med Outperforms Confidence Gating  
**Evaluation Cohort:** Camelyon17-WILDS Center 2 ($N = 85,054$ patches, 10 Whole Slide Images, 9 patients)  
**Seeds Under Evaluation:** 42, 123, 2024, 7, 99  

---

## 1. Statistical Hypothesis Testing Framework

To establish whether EMA Teacher-Student UTTA-Med (E2) genuinely outperforms Confidence Gating (E1), we formulate formal statistical hypotheses on the paired differences across all five authentic seeds:

### Primary Hypothesis (Binary F1 Score)
* **Null Hypothesis ($H_0$):** $\mu_{\Delta F_1} = \mathbb{E}\left[F_1(E2) - F_1(E1)\right] \le 0$  
  *(EMA Teacher-Student UTTA-Med provides no improvement in F1 over deterministic Confidence Gating).*
* **Alternative Hypothesis ($H_1$):** $\mu_{\Delta F_1} = \mathbb{E}\left[F_1(E2) - F_1(E1)\right] > 0$  
  *(EMA Teacher-Student UTTA-Med achieves a statistically significant improvement in F1).*

### Secondary Hypothesis (AUROC)
* **Null Hypothesis ($H_0$):** $\mu_{\Delta \text{AUROC}} = \mathbb{E}\left[\text{AUROC}(E2) - \text{AUROC}(E1)\right] \le 0$
* **Alternative Hypothesis ($H_1$):** $\mu_{\Delta \text{AUROC}} = \mathbb{E}\left[\text{AUROC}(E2) - \text{AUROC}(E1)\right] > 0$

### Test Statistic & Decision Criteria
* **Test:** Two-sided and one-sided paired Student's t-test with $n = 5$ seeds and degrees of freedom $df = n - 1 = 4$:
  $$t = \frac{\bar{D}}{s_D / \sqrt{n}}, \quad \bar{D} = \frac{1}{n}\sum_{i=1}^n D_i, \quad s_D = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (D_i - \bar{D})^2}$$
* **95% Confidence Interval:**
  $$\text{CI}_{95\%} = \left[ \bar{D} - t_{0.975, 4} \cdot \frac{s_D}{\sqrt{5}}, \quad \bar{D} + t_{0.975, 4} \cdot \frac{s_D}{\sqrt{5}} \right], \quad t_{0.975, 4} = 2.7764$$
* **Significance Level:** $\alpha = 0.05$. Significance is achieved if and only if $p < 0.05$ and the $95\%$ CI strictly excludes zero.

---

## 2. Decoupling Backbone Improvements from TTA Algorithm Gains

A critical failure mode in machine learning benchmarks is conflating the baseline uplift from a stronger model architecture with the uplift from a test-time adaptation algorithm. We enforce strict three-way attribution partitioning:

```
[Frozen DenseNet-121 (E0)] ───► Baseline Architecture Gain = E0 - Historical ResNet-18
         │
         ├───► [Confidence Gating (E1)] ───► Gating Adaptation Gain = E1 - E0
         │
         └───► [EMA UTTA-Med (E2)]      ───► Pure Algorithm Uplift = E2 - E1
```

1. **Backbone Contribution:** $\Delta_{\text{Backbone}} = \text{Metric}(E0_{\text{DenseNet}}) - \text{Metric}(\text{ResNet-18}_{\text{Frozen}})$.
2. **Confidence Gating Contribution:** $\Delta_{\text{Gating}} = \text{Metric}(E1) - \text{Metric}(E0)$.
3. **Pure UTTA-Med TTA Contribution:** $\Delta_{\text{Algorithm}} = \text{Metric}(E2) - \text{Metric}(E1)$.

**Scientific Rule:** A claim that "UTTA-Med improved performance" is valid **only if $\Delta_{\text{Algorithm}} = E2 - E1 > 0$** at statistical significance ($p < 0.05$). If $E2 \approx E1$, any higher absolute score over ResNet-18 is entirely attributable to DenseNet-121 representation quality, not adaptation.

---

## 3. Rejection of Synthetic Projection $F_1 = 0.8723$

* **Audit Finding:** An earlier draft projected an F1 of $0.8723$ based on an analytical scaling curve.
* **Empirical Reality:** Real test-time adaptation on histopathology slides faces distribution shift, stain variation, and pseudo-label noise. The projected $0.8723$ is **unsupported by authentic GPU prediction arrays**.
* **Integrity Commitment:** We strictly forbid presenting $0.8723$ as a verified finding. All reporting must reflect the true numbers calculated directly from `s*_predictions.npz`.

---

## 4. Methodological Limitation: Exploratory vs Confirmatory Nature (Issue I13)

We formally acknowledge a central methodological limitation regarding the Camelyon17 benchmark:
* **The Setting:** Camelyon17 Center 2 test data has been repeatedly evaluated across multiple project iterations (ResNet-18 tuning, feature centroid extraction, threshold sweeps).
* **Scientific Classification:** Consequently, any evaluation on Center 2 constitutes an **exploratory benchmark comparison** rather than a pristine, confirmatory double-blind validation.
* **Mitigation:**
  1. We freeze hyperparameters ($\tau = 0.90$, EMA decay $= 0.999$, student learning rate $= 10^{-4}$) prior to running GPU evaluation.
  2. Ground-truth target labels are strictly isolated during adaptation; only unsupervised predictions and optical density augmentations are observed by the adaptation engine.
  3. Ground-truth labels are accessed exclusively post-hoc by the evaluation script for calculating confusion matrices and ROC curves.

---

## 5. Historical ResNet-18 Reference Baseline Matrix

For reference against incoming DenseNet-121 results, the authentic 5-seed ResNet-18 baselines are established below:

| Method ID | Backbone | Method Name | AUROC (Mean $\pm$ SD) | Binary F1 (Mean $\pm$ SD) | Recall (Mean $\pm$ SD) | Precision (Mean $\pm$ SD) | Harm Rate ($H_{\text{err}}/N$) | Coverage ($\tau=0.90$) | Batch Latency ($B=32$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C0** | ResNet-18 | Frozen Source | $0.9351 \pm 0.0218$ | $0.8296 \pm 0.0325$ | $71.81\% \pm 5.41\%$ | $98.54\% \pm 0.93\%$ | $0.0000$ (Baseline) | $100.0\%$ | $6.74\text{ ms}$ |
| **C1** | ResNet-18 | Confidence Gating | $0.9355 \pm 0.0218$ | $0.8310 \pm 0.0321$ | $72.10\% \pm 5.35\%$ | $98.58\% \pm 0.91\%$ | $0.0070 \pm 0.0028$ | $83.74\% \pm 2.23\%$ | $6.77\text{ ms}$ |
| **C2** | ResNet-18 | MC-Dropout (20p) | $0.9350 \pm 0.0220$ | $0.8296 \pm 0.0335$ | $71.79\% \pm 5.45\%$ | $98.52\% \pm 0.94\%$ | $0.0068 \pm 0.0027$ | $54.20\% \pm 4.50\%$ | $136.29\text{ ms}$ |
| **C7** | ResNet-18 | Feature + Conf Gating | $0.9372 \pm 0.0215$ | $0.8338 \pm 0.0315$ | $72.55\% \pm 5.20\%$ | $98.65\% \pm 0.88\%$ | $0.0052 \pm 0.0021$ | $56.20\% \pm 3.80\%$ | $15.05\text{ ms}$ |

**Target for DenseNet-121 (E2 vs E1):**
Upon retrieval of DenseNet-121 predictions, the exact same statistical tests will be executed via `scripts/retrieve_and_finalize_densenet121.py` to populate `FINAL_E0_E1_E2_COMPARISON.csv`.
