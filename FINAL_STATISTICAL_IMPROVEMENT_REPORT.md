# Final Statistical Improvement & Attribution Separation Report

**Project:** UTTA-Med Project 2  
**Document ID:** `FINAL_STATISTICAL_IMPROVEMENT_REPORT.md`  
**Evaluation Cohort:** Camelyon17-WILDS Center 2 ($N = 85,054$ patches across 10 Whole Slide Images, 9 patients)  
**Seeds Under Evaluation:** 42, 123, 2024, 7, 99  
**Execution Node:** 2 $\times$ Nvidia Tesla T4 Tensor Core GPUs (`avinashreddy7676/utta-med-densenet121-training`, Kernel ID: `140433565`)  

---

## 1. Scientific Research Question & Hypotheses

The fundamental inquiry of Project 2 is:
> **Does EMA Teacher–Student UTTA-Med (E2) achieve a statistically significant improvement over Confidence Gating (E1) when evaluated using identical trained DenseNet-121 source checkpoints on authentic histopathology Whole Slide Images?**

### Mathematical Formulation
For each seed $s \in \{42, 123, 2024, 7, 99\}$:
$$\Delta F_{1, s} = F_1(E2, s) - F_1(E1, s)$$
$$\Delta \text{AUROC}_s = \text{AUROC}(E2, s) - \text{AUROC}(E1, s)$$

* **Null Hypothesis ($H_0$):** $\mu_{\Delta F_1} \le 0$  
  *(EMA Teacher–Student UTTA-Med provides no performance improvement over deterministic Confidence Gating).*
* **Alternative Hypothesis ($H_1$):** $\mu_{\Delta F_1} > 0$  
  *(EMA Teacher–Student UTTA-Med achieves a statistically significant improvement in binary F1).*

---

## 2. Decoupling Supervised Backbone Gains from Adaptation Gains

A central failure mode in test-time adaptation literature is **attribution conflation**: attributing performance gains derived from migrating to a superior supervised backbone (e.g., from ResNet-18 to DenseNet-121) to the test-time adaptation algorithm itself.

To prevent this error, we establish strict three-way attribution partitioning:

```
[Frozen ResNet-18 (C0)] ──► Baseline ResNet-18 ($F_1 = 0.8296 \pm 0.0325$, $\text{AUROC} = 0.9351 \pm 0.0218$)
        │
        ▼ (Backbone Transition: ResNet-18 to DenseNet-121)
[Frozen DenseNet-121 (E0)] ──► Backbone Architectural Uplift = Metric(E0) - Metric(C0)
        │
        ▼ (Single-Pass Confidence Gating, tau=0.90)
[Confidence Gating (E1)]   ──► Gating Adaptation Uplift = Metric(E1) - Metric(E0)
        │
        ▼ (Dual-View Student-Teacher Adaptation)
[EMA UTTA-Med (E2)]        ──► Pure TTA Algorithm Uplift = Metric(E2) - Metric(E1)
```

### Scientific Attribution Rule
* **Backbone Uplift:** $\Delta_{\text{Backbone}} = \text{Metric}(E0_{\text{DenseNet}}) - \text{Metric}(C0_{\text{ResNet-18}})$.
* **Gating Uplift:** $\Delta_{\text{Gating}} = \text{Metric}(E1) - \text{Metric}(E0)$.
* **Pure TTA Algorithm Uplift:** $\Delta_{\text{Algorithm}} = \text{Metric}(E2) - \text{Metric}(E1)$.

**Methodological Standard:** If DenseNet-121 reaches an absolute $F_1 > 0.85$, but Method E1 (Confidence Gating) also reaches that same score ($\Delta_{\text{Algorithm}} = E2 - E1 \approx 0$), then **the improvement is 100% attributable to the DenseNet-121 feature backbone**, not the test-time adaptation algorithm. Improved UTTA-Med can only claim algorithmic superiority if $\Delta_{\text{Algorithm}} = E2 - E1 > 0$ at statistical significance ($p < 0.05$).

---

## 3. Strict Rejection of Projected $F_1 = 0.8723$

* **Audit Finding:** Previous progress summaries presented an analytical projection ($F_1 = 0.8723$, AUROC $= 0.9520$) as an expected result.
* **Ground Truth:** This figure was extrapolated from pilot scaling curves and is **unsupported by authentic GPU prediction arrays**.
* **Integrity Standard:** We strictly prohibit reporting $0.8723$ as a verified finding. In [`FINAL_E0_E1_E2_COMPARISON.csv`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/FINAL_E0_E1_E2_COMPARISON.csv), all cells remain marked as `AWAITING_GPU_PREDICTIONS` until authentic inference finishes.
* **Acceptance Protocol:**
  - If empirical $F_1$ exceeds $0.85$, we will report the exact observed score and its 95% confidence interval.
  - If EMA UTTA-Med outperforms Confidence Gating ($\Delta F_1 > 0, p < 0.05$), we will document the paired gain.
  - If EMA UTTA-Med does not outperform Confidence Gating ($\Delta F_1 \le 0$ or $p \ge 0.05$), we will report that result without manipulation.

---

## 4. Paired Statistical Testing Protocol ($n=5, df=4$)

When raw target prediction arrays (`s*_predictions.npz`) are downloaded, the automated verification script ([`scripts/retrieve_and_finalize_densenet121.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/retrieve_and_finalize_densenet121.py)) executes:

$$\bar{D} = \frac{1}{5}\sum_{s=1}^5 \Delta F_{1, s}, \quad s_D = \sqrt{\frac{1}{4}\sum_{s=1}^5 (\Delta F_{1, s} - \bar{D})^2}$$
$$t = \frac{\bar{D}}{s_D / \sqrt{5}}, \quad \text{CI}_{95\%} = \left[ \bar{D} - 2.7764 \cdot \frac{s_D}{\sqrt{5}}, \quad \bar{D} + 2.7764 \cdot \frac{s_D}{\sqrt{5}} \right]$$

Significance requires two conditions:
1. Two-sided $p$-value $< 0.05$.
2. The 95% confidence interval strictly excludes zero ($\text{CI}_{\text{lower}} > 0$).

---

## 5. Methodological Disclosure: Exploratory Nature of Center 2 Benchmark

We formally disclose an unavoidable methodological limitation:
* **Repeated Benchmark Inspection:** The Camelyon17 Center 2 test cohort has been evaluated repeatedly across multiple research phases (ResNet-18 baseline tuning, feature centroid calculation, confidence sweeps).
* **Scientific Classification:** As a result, subsequent evaluations on Center 2 constitute an **exploratory benchmark comparison** rather than a fresh confirmatory double-blind validation.
* **Mitigation Protocol:**
  1. All adaptation hyperparameters ($\tau = 0.90$, EMA decay $= 0.999$, $\lambda_{\text{anchor}} = 0.01$, $\text{lr} = 10^{-4}$) were frozen before launching training.
  2. Ground-truth target labels are strictly withheld during adaptation; the engine observes only unlabeled patches.
  3. Ground-truth labels are evaluated exclusively post-hoc by evaluation scripts for computing final confusion matrices.
