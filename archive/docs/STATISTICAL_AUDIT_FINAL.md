# UTTA-Med Stage 14 — Final Statistical Audit

## Primary inferential unit

The primary replicated-experiment unit is the **training/adaptation seed** (n=5). For each contrast, the paired difference is computed within seed and summarized with a two-sided paired t interval with df=4. WSI-level bootstrap is secondary because the target contains only 10 WSIs.

## Primary results

| Contrast | Mean Δ | 95% t CI | Nominal p | Holm-adjusted p | Cohen d |
|---|---:|---:|---:|---:|---:|
| UTTA − Source AUROC | +0.0058 | [-0.0121, +0.0237] | 0.4199 | 0.8397 | +0.40 |\n| UTTA − Tent AUROC | +0.2373 | [-0.1411, +0.6158] | 0.1566 | 0.5396 | +0.78 |\n| UTTA − EATA AUROC | +0.0145 | [+0.0019, +0.0271] | 0.0330 | 0.1979 | +1.43 |\n| UTTA − Random AUROC | +0.2388 | [-0.1158, +0.5934] | 0.1349 | 0.5396 | +0.84 |\n| UTTA − Confidence AUROC | -0.0000 | [-0.0008, +0.0007] | 0.8688 | 0.8688 | -0.08 |\n| UTTA − Source F1 | +0.0231 | [+0.0104, +0.0358] | 0.0072 | 0.0505 | +2.26 |\n| UTTA − Confidence F1 | -0.0010 | [-0.0015, -0.0005] | 0.0054 | 0.0430 | -2.45 |\n| UTTA − Tent harm | -0.1802 | [-0.3994, +0.0390] | 0.0846 | 0.4230 | -1.02 |\n
## Interpretation rules

- UTTA vs Source F1: nominal p=0.0072; the 95% seed-level CI is entirely positive. After Holm correction across the eight prespecified comparisons, p=0.0505, so this should **not** be described as Holm-significant at α=0.05.
- UTTA vs Confidence F1: nominal p=0.0054 and Holm-adjusted p=0.0430; the numerical gap is only −0.0010 F1. Report the magnitude, not only the adjusted p-value.
- UTTA vs Source AUROC: CI includes zero; do not claim a universal AUROC improvement.
- UTTA vs Confidence AUROC: essentially zero mean difference and CI includes zero.
- UTTA vs EATA AUROC: nominal p=0.033, but Holm-adjusted p=0.1979; do not call it significant after correction.
- Wilcoxon two-sided p-values have a finite-sample floor of 0.0625 when all five paired differences have the same sign, so Wilcoxon is supplementary rather than the primary significance test.
- Harm reduction is directionally consistent in all five seeds, but its seed-level t CI includes zero because seeds 7 and 99 produce very large Tent harm values. Report mean±SD and 5/5 direction counts.

## WSI-level inference

Seed 42: 1,000 WSI resamples give UTTA−Source ΔAUROC=+0.0143 [0.0021, 0.0263] and ΔF1=+0.0399 [0.0178, 0.0902]. UTTA−Confidence is −0.00066 AUROC and −0.00108 F1, both with narrow intervals excluding zero in this seed-42 bootstrap.

For seeds 123, 2024, 7, and 99, per-slide paired bootstrap intervals are wider; the documented pattern is that UTTA−Source AUROC excludes zero only for seed 2024. These results reinforce that 10 target WSIs are the appropriate cluster unit and that universal slide-level superiority should not be claimed.

## Statistical close-out

**PASS for analysis validity, with conservative reporting.** The paper should report effect sizes, confidence intervals, nominal p-values where useful, and Holm-adjusted p-values; it should not describe the nominal F1 p=0.007 as multiplicity-adjusted significance.
