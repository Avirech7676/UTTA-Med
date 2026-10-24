# Final Adaptation Superiority Analysis & Attribution Separation Report

**Project:** UTTA-Med Project 2  
**Document ID:** `FINAL_ADAPTATION_SUPERIORITY_ANALYSIS.md`  
**Scope:** Establishing Whether Improved EMA UTTA-Med (E2) Genuinely Outperforms Confidence Gating (E1)  
**Evaluation Cohort:** Camelyon17-WILDS Center 2 ($N = 85,054$ patches, 10 Whole Slide Images, 9 patients)  
**Trained Seeds:** 42, 123, 2024, 7, 99  

---

## 1. Scientific Research Question

The core objective of Project 2 is to answer the following research question:
> **Does the improved EMA Teacher–Student UTTA-Med algorithm achieve a statistically significant performance gain over deterministic Confidence Gating when evaluated using identical trained checkpoints on authentic histopathology data?**

### Statistical Formulation
We define the paired difference between Method E2 (EMA Teacher–Student UTTA-Med) and Method E1 (Confidence Gating) for seed $s \in \{42, 123, 2024, 7, 99\}$:
$$\Delta F_{1, s} = F_1(E2, s) - F_1(E1, s)$$
$$\Delta \text{AUROC}_s = \text{AUROC}(E2, s) - \text{AUROC}(E1, s)$$

* **Null Hypothesis ($H_0$):** $\mu_{\Delta F_1} \le 0$  
  *(Improved UTTA-Med provides no advantage over Confidence Gating).*
* **Alternative Hypothesis ($H_1$):** $\mu_{\Delta F_1} > 0$  
  *(Improved UTTA-Med genuinely outperforms Confidence Gating).*

---

## 2. Decoupling Backbone Gains from Adaptation Gains

A central methodological risk in test-time adaptation research is **attribution conflation**: attributing performance gains from a superior supervised backbone architecture to the test-time adaptation algorithm. We enforce strict three-way attribution partitioning:

```
[Frozen ResNet-18 (C0)] ───► Baseline ResNet-18 Reference ($F_1 = 0.8296 \pm 0.0325$)
        │
        ▼ (Backbone Transition: ResNet-18 to DenseNet-121)
[Frozen DenseNet-121 (E0)] ──► Pure Architecture Gain = Metric(E0) - Metric(C0)
        │
        ▼ (Single-Pass Confidence Gating)
[Confidence Gating (E1)]   ──► Gating Adaptation Gain = Metric(E1) - Metric(E0)
        │
        ▼ (Dual-View Student-Teacher Adaptation)
[EMA UTTA-Med (E2)]        ──► Pure TTA Algorithm Uplift = Metric(E2) - Metric(E1)
```

### Attribution Rules
1. **Backbone Contribution:** $\Delta_{\text{Backbone}} = \text{Metric}(E0_{\text{DenseNet}}) - \text{Metric}(C0_{\text{ResNet-18}})$.
2. **Confidence Gating Contribution:** $\Delta_{\text{Gating}} = \text{Metric}(E1) - \text{Metric}(E0)$.
3. **Pure TTA Algorithm Contribution:** $\Delta_{\text{Algorithm}} = \text{Metric}(E2) - \text{Metric}(E1)$.

**Scientific Ruling:** If $E2$ achieves an absolute $F_1 > 0.85$, but $E1$ ALSO achieves an identical $F_1 > 0.85$ ($\Delta_{\text{Algorithm}} \approx 0$), then **the improvement was caused entirely by DenseNet-121 supervised representation learning**, NOT by the test-time adaptation algorithm. Improved UTTA-Med can only be claimed as superior if $\Delta_{\text{Algorithm}} > 0$ at statistical significance ($p < 0.05$).

---

## 3. Strict Rejection of Projected $F_1 = 0.8723$

* **Audit Finding:** Previous project reports cited a projected $F_1$ of $0.8723$ and AUROC of $0.9520$.
* **Ground Truth:** This figure was an analytical extrapolation from pilot scaling curves. It is **unsupported by authentic GPU prediction arrays**.
* **Integrity Commitment:** We strictly forbid citing or assuming $F_1 = 0.8723$. All metrics in `FINAL_E0_E1_E2_RESULTS.csv` will be computed directly from raw binary classification arrays ($p_{\text{pred}} \ge 0.5$ vs $y_{\text{true}}$) without synthetic projection.

---

## 4. Paired Statistical Evaluation Protocol

Upon retrieval of the 5 authentic seed prediction arrays from Kaggle GPU compute, the following automated statistical testing script ([`scripts/retrieve_and_finalize_densenet121.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/retrieve_and_finalize_densenet121.py#L105-L131)) will execute:

```python
def analyze_paired_statistics(df_results: pd.DataFrame) -> Dict[str, Any]:
    diffs = df_results["e2_f1"].to_numpy() - df_results["e1_f1"].to_numpy()
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    se_diff = std_diff / math.sqrt(n)

    t_stat, p_val_two_sided = stats.ttest_1samp(diffs, 0.0)
    t_crit = stats.t.ppf(0.975, df=n - 1)  # 2.7764 for df=4
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff

    return {
        "n": n,
        "df": n - 1,
        "mean_delta_f1": mean_diff,
        "std_delta_f1": std_diff,
        "t_statistic": float(t_stat),
        "p_value_two_sided": float(p_val_two_sided),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "is_significant": bool(p_val_two_sided < 0.05 and ci_lower > 0),
    }
```

---

## 5. Methodological Limitation: Exploratory vs Confirmatory Nature

We explicitly disclose an important methodological constraint:
* **The Center 2 Benchmark Setting:** The Camelyon17 Center 2 test split has been utilized in multiple historical research iterations (ResNet-18 baseline tuning, centroid extraction, threshold sweeps).
* **Scientific Reality:** As a result, subsequent evaluations on Center 2 constitute an **exploratory benchmark comparison** rather than a pristine, confirmatory double-blind clinical validation.
* **Mitigations Implemented:**
  1. All adaptation hyperparameters ($\tau = 0.90$, EMA decay $= 0.999$, $\lambda_{\text{anchor}} = 0.01$, $\text{lr} = 10^{-4}$) were frozen prior to launching GPU training.
  2. Ground-truth target labels are strictly withheld during adaptation; the model observes only unlabeled patches.
  3. Ground-truth labels are accessed exclusively post-hoc by evaluation scripts for computing final confusion matrices.
