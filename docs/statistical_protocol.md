# Statistical Protocol & Significance Guidelines — UTTA-Med

This document defines the authoritative statistical methodology, inference protocols, reported confidence intervals, and reporting guidelines for UTTA-Med on the **Camelyon17-WILDS** benchmark.

---

## 1. Multi-Seed Inference Protocol

- **Fixed Seed Suite**: All methods are evaluated across five fixed, preregistered random seeds:
  $$\mathcal{S} = \{42, 123, 2024, 7, 99\}$$
- **Inference Unit**: Paired seed-level differences ($\Delta = \text{Metric}_{\text{method}} - \text{Metric}_{\text{source}}$).
- **Degrees of Freedom**: $n = 5$ seeds yields $\text{df} = n - 1 = 4$.
- **Critical $t$-value**: Two-sided 95% critical value is $t_{0.975, \text{df}=4} = 2.776$.
- **Paired $t$-Interval**:
  $$\bar{\Delta} \pm t_{0.975, 4} \cdot \frac{s_{\Delta}}{\sqrt{5}}$$
- **Wilcoxon Signed-Rank Floor**: For $n = 5$, the minimum achievable two-sided Wilcoxon $p$-value when all 5 signs agree ($5+/0-$ or $0+/5-$) is:
  $$p_{\min} = 2 \times \left(\frac{1}{2}\right)^5 = \frac{2}{32} = 0.0625$$
  A Wilcoxon $p = 0.0625$ represents the strict mathematical floor of the test at this sample size, not an inconclusive result.
- **Conservative Superiority Language**:
  - There is **no claim of universal superiority** for UTTA-Med across all metrics or baselines.
  - On AUROC, UTTA-Med improves over Source on 3 of 5 seeds (42, 123, 2024) and shows slight decreases on the two collapse-prone seeds (7 and 99); mean paired $\Delta\text{AUROC} = +0.0058$ ($p = 0.42$, not statistically significant).
  - On F1, UTTA-Med improves over Source on 5 of 5 seeds; mean paired $\Delta\text{F1} = +0.0231$ ($95\%\text{ CI: } [+0.0104, +0.0358]$, $p = 0.007$).

---

## 2. Whole-Slide Image (WSI) vs. Patch-Level Inference

- **Primary Unit of Inference**: Whole-Slide Image (WSI) / patient slide level.
  - Test Center 2 contains **10 Whole-Slide Images** ($\text{WSI IDs } \{20, 21, 22, 23, 24, 25, 26, 27, 28, 29\}$), comprising 85,054 patches.
  - Histopathology patches sampled from the same physical tissue slide share slide-level staining, illumination, scanner artifacts, and patient-specific cellular morphology.
  - Treating patches as independent and identically distributed (i.i.d.) severely underestimates standard errors and yields unrealistically narrow confidence intervals.
  - Primary reporting therefore evaluates **cluster-aware WSI-level bootstrap confidence intervals** via paired slide resampling.
- **Secondary Sensitivity Analysis**:
  - Pooled patch-level resampling is reported only as a secondary sensitivity check, never as primary evidence of statistical significance.
- **Slide Resampling Procedure**:
  1. Draw $M = 10$ slide IDs uniformly with replacement from the 10 target slides.
  2. For paired comparisons, evaluate both baseline and adapted predictions on the identical resampled slide set (preserving slide-level covariance).
  3. Concatenate all constituent patches belonging to the resampled slides.
  4. Compute the metric on the concatenated patch set.
  5. Repeat for $B = 1000$ iterations and take the empirical 2.5th and 97.5th percentiles.
- **Seed Coverage**:
  - Seed 42 target dumps (`y, p, slide_id`) serve as the documented primary paired slide bootstrap.
  - Seeds 123, 2024, 7, and 99 use their corresponding per-slide paired bootstrap analysis.

---

## 3. Exact Reported Confidence Intervals (Frozen Artifact Alignment)

All active text, tables, and documentation must exactly mirror the frozen scientific artifacts in `results/`:

### A. Seed-42 Paired WSI Bootstrap ($B = 1000$ slide resamples)
Documented in `results/table_paired_bootstrap.csv` and `archive/docs/STATISTICAL_AUDIT_FINAL.md`:
- **UTTA-Med − Source $\Delta\text{AUROC}$**: $+0.0143$ $[0.0021, 0.0263]$ (excludes 0)
- **UTTA-Med − Source $\Delta\text{F1}$**: $+0.0399$ $[0.0178, 0.0902]$ (excludes 0)
- **UTTA-Med − Confidence $\Delta\text{AUROC}$**: $-0.0007$ $[-0.0012, -0.0004]$
- **UTTA-Med − Confidence $\Delta\text{F1}$**: $-0.0010$ $[-0.0020, -0.0010]$

### B. Five-Seed Paired Seed-Level Contrasts ($n = 5, \text{df} = 4, t_{0.975} = 2.776$)
Documented in `results/statistical/five_seed_final/STATISTICAL_REPORT.md`:

| Contrast | Metric | Mean $\Delta$ | 95% $t$ CI | $t$-test $p$ | Wilcoxon $p$ | Sign Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **UTTA − Source** | **AUROC** | $+0.0058$ | $[-0.0121, +0.0237]$ | $0.420$ | $0.438$ | 3+ / 2− |
| **UTTA − Source** | **F1 Score** | $+0.0231$ | $[+0.0104, +0.0358]$ | $0.007$ | $0.0625$ | 5+ / 0− |
| **UTTA − Confidence** | **AUROC** | $-0.0000$ | $[-0.0008, +0.0007]$ | $0.870$ | $0.625$ | 1+ / 4− |
| **UTTA − Confidence** | **F1 Score** | $-0.0010$ | $[-0.0015, -0.0005]$ | $0.005$ | $0.0625$ | 0+ / 5− |
| **UTTA − EATA** | **AUROC** | $+0.0145$ | $[+0.0018, +0.0272]$ | $0.033$ | $0.0625$ | 5+ / 0− |
| **UTTA − Tent** | **AUROC** | $+0.2370$ | $[-0.1410, +0.6150]$ | $0.160$ | $0.0625$ | 5+ / 0− |
| **UTTA − Tent** | **Harm Rate** | $-0.1800$ | $[-0.3990, +0.0390]$ | $0.085$ | $0.0625$ | 0+ / 5− |

---

## 4. Statistical Significance Language & Multiple Comparisons

- **Holm–Bonferroni Adjustment**: Across the family of primary contrasts (8 primary hypothesis tests):
  - **UTTA vs. Source F1**: Unadjusted $p = 0.007 \rightarrow$ Holm-adjusted $p_{\text{adj}} = 0.051$ (borderline significant).
  - **UTTA vs. Confidence F1**: Unadjusted $p = 0.005 \rightarrow$ Holm-adjusted $p_{\text{adj}} = 0.043$ (statistically detectable, but effect size $\Delta = 0.001$ is clinically negligible).
  - **All AUROC Contrasts**: Not statistically significant after Holm adjustment ($p_{\text{adj}} > 0.05$).
- **Per-Slide Intervals for Other Seeds**:
  - Because $N_{\text{slides}} = 10$, per-slide confidence intervals are wide.
  - Across seeds 123, 2024, 7, and 99, the per-slide AUROC difference UTTA − Source excludes zero **only on seed 2024** (+0.0235). On seeds 123, 7, and 99, the per-slide intervals include zero.
- **Comparison with Confidence-Gated TTA**:
  - Confidence gating achieves mean AUROC $0.936 \pm 0.022$ and F1 $0.831 \pm 0.032$, compared to UTTA-Med ($0.935 \pm 0.022$ AUROC, $0.830 \pm 0.032$ F1).
  - UTTA-Med and confidence gating showed nearly identical AUROC performance ($\Delta = -0.00005$, 95% CI $-0.00078$ to $+0.00069$; two-sided paired $p = 0.869$). This comparison did not demonstrate a statistically significant difference; formal equivalence was not established.
  - Confidence-gated performance slightly exceeds UTTA-Med on F1 by $0.0010$ ($p_{\text{adj}} = 0.043$).
  - MC-Dropout predictive variance is a verified error detector ($r = 0.397 \pm 0.016$), but on binary ResNet-18 classification it is functionally redundant with softmax confidence. The paper frames this transparently as demonstrating no significant AUROC difference, not as superiority over confidence gating.

---

## 5. Coverage and Adaptation Statistics Terminology

Documentation and reporting must strictly adhere to standard, unambiguous definitions:

1. **Adaptation Coverage ($C$)**:
   $$\text{Coverage} = \frac{N_{\text{adapted}}}{N_{\text{total}}}$$
   The fraction of incoming test batches/samples meeting the gating criterion and updating model parameters.
2. **Correction Rate**:
   $$\text{Correction Rate} = \frac{\sum (\hat{y}_{\text{source}} \neq y \land \hat{y}_{\text{tta}} = y)}{\sum (\hat{y}_{\text{source}} \neq y)}$$
   The proportion of baseline errors corrected by test-time adaptation.
3. **Harmful-Adaptation Rate (Harm Rate)**:
   $$\text{Harm Rate} = \frac{\sum (\hat{y}_{\text{source}} = y \land \hat{y}_{\text{tta}} \neq y)}{\sum (\hat{y}_{\text{source}} = y)}$$
   The proportion of previously correct baseline predictions converted into errors by adaptation.
4. **Prediction Flip Rate**:
   $$\text{Flip Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{source}}^{(i)} \neq \hat{y}_{\text{tta}}^{(i)})$$
   The total fraction of predictions modified by adaptation regardless of ground truth.
5. **Adaptation Gain**:
   $$\text{Net Gain} = N_{\text{corrected}} - N_{\text{harmed}}$$
   The net count or percentage change in correct classifications attributable to TTA.
6. **Matched-Coverage Controls**:
   - Random gating and Confidence/Entropy gating evaluated at the exact empirical target coverage of UTTA-Med (~58%).
   - Controls prove that collapse prevention is driven by **which** samples are updated (uncertainty/epistemic filtering), not merely by taking fewer gradient steps.
7. **Performance Improvement vs. Safe/Selective Adaptation**:
   - Always distinguish between **discriminative performance improvement** (marginal AUROC/F1 gains) and **safe/selective adaptation behavior** (bounding harm $< 1\%$, avoiding catastrophic collapse seen in Tent where harm reaches $37.5\% - 39.1\%$).

---

## 6. Implementation Agreement with `src/evaluation/statistics.py`

The canonical implementation in [`src/evaluation/statistics.py`](../src/evaluation/statistics.py) defines:
```python
def bootstrap_ci_by_slide(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    slide_ids: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    ...
```
This implementation accepts true labels, continuous predictions, slide IDs, and a generic metric callable (such as AUROC, F1, or ECE), sampling unique slide clusters with replacement and computing quantile intervals. No documentation may alter or misrepresent this signature.

---

## 7. Canonical Reproduction Workflow

All reproduction references must point exclusively to the production pipeline:
- **Source Training**: `python scripts/train_source.py --config configs/camelyon17.yaml --seed <seed>`
- **Authoritative Thresholds**: Read directly from `configs/tau_by_seed.yaml` (calibrated on Center 1 validation data). No manual `--tau` override.
- **Official Seeds**: `42, 123, 2024, 7, 99`.
- **Adaptation & Evaluation**: `python scripts/adapt.py --checkpoint checkpoints/camelyon17_resnet18_source_s<seed>_BEST.pt --method <source|tent|eata|utta> --seed <seed>`
- **Statistical Summaries**: `python scripts/verify_config.py` and `src/evaluation/statistics.py`.
- Archived scripts (`train_baseline.py`), legacy exploration, and non-canonical overrides are strictly excluded from reproduction instructions.
