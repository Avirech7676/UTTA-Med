# Statistical Protocol — UTTA-Med

## Primary Inference Protocol

- **Across-seed headline reporting:** mean ± SD with paired seed-level comparisons across the five fixed seeds (`df=4`).
- **Cluster-aware inference:** WSI/slide-level bootstrap confidence intervals using the 10 target slides.

### Procedure
For each bootstrap iteration:
1. Sample WSI IDs with replacement from the evaluation set.
2. Include **all** patches belonging to the sampled WSIs.
3. Compute the metric on the resulting patch set.
4. Repeat B times (recommended B ≥ 1000).

Implement as:
```python
# src/evaluation/statistics.py
def bootstrap_ci_by_slide(predictions, slide_ids, metric_fn, n_boot=1000, alpha=0.05, seed=42):
    ...
```

Patch-level bootstrap may be reported only as a secondary sensitivity analysis.

## Multi-Seed Reporting
- Minimum 5 independent seeds.
- Fixed seed list: 42, 123, 2024, 7, 99.
- Report Mean ± SD and 95 % CI for every headline metric.
- Same seeds applied identically to every method.

## Multiple Comparisons
When comparing UTTA-Med against multiple baselines (Tent, EATA, Random, Confidence, Entropy), apply a multiple-comparison correction (Holm or Bonferroni) to the family of primary tests.

## Effect Size
Always report absolute difference, relative difference (where meaningful), and an effect-size measure alongside p-values / CIs.
