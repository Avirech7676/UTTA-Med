# Final Classification Statistical Analysis Framework

**Project:** UTTA-Med Project 2 — Biostatistical Protocol  

---

## 1. Statistical Comparison Protocol: E2 vs E1

To evaluate whether Improved EMA UTTA-Med genuinely outperforms Confidence Gating:
1. **Paired Metric Deltas:**
   $$\Delta F_1 = F_1(E2) - F_1(E1), \quad \Delta \text{AUROC} = \text{AUROC}(E2) - \text{AUROC}(E1)$$
2. **Hypothesis Testing:**
   * Null Hypothesis $H_0$: $\mu_{\Delta F_1} \le 0$
   * Alternative Hypothesis $H_1$: $\mu_{\Delta F_1} > 0$
   * Two-sided paired Student's t-test ($df = 4, \alpha = 0.05$).
3. **Attribution Separation:**
   * Supervised backbone gain: $\Delta_{\text{Backbone}} = E0(\text{DenseNet}) - M0(\text{ResNet})$.
   * Adaptation gain: $\Delta_{\text{TTA}} = E2(\text{EMA}) - E1(\text{Confidence})$.
