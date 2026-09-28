# Gate 10 — WSI-level bootstrap (5 seeds) — SIGNED OFF

Primary inference unit: **hospital-2 whole-slide IDs** `{20–29}` (n = 10).
Resample slides with replacement (B = 1000), keep all patches of sampled slides.

Point estimates match the locked 5-seed table. Use **these CIs** in the paper, never patch-level CIs.

## Per-seed AUROC with WSI 95% CI

| Seed | Source | Tent | EATA | Confidence | UTTA-Med |
|------|--------|------|------|------------|----------|
| 42 | 0.936 [0.843, 0.968] | 0.929 [0.795, 0.959] | 0.939 [0.833, 0.976] | 0.951 [0.867, 0.982] | 0.950 [0.867, 0.981] |
| 123 | 0.935 [0.818, 0.969] | 0.941 [0.853, 0.970] | 0.940 [0.819, 0.980] | 0.942 [0.806, 0.983] | 0.942 [0.806, 0.983] |
| 2024 | 0.932 [0.837, 0.966] | 0.933 [0.807, 0.961] | 0.932 [0.823, 0.975] | 0.956 [0.881, 0.982] | 0.955 [0.879, 0.982] |
| 7 | 0.913 [0.749, 0.961] | **0.343 [0.138, 0.681]** | 0.877 [0.572, 0.960] | 0.902 [0.662, 0.970] | 0.901 [0.663, 0.969] |
| 99 | 0.933 [0.827, 0.964] | **0.344 [0.136, 0.720]** | 0.920 [0.754, 0.968] | 0.927 [0.770, 0.971] | 0.926 [0.769, 0.971] |

## What you may claim

1. Tent **collapse on seeds 7 and 99 is slide-level significant** (Tent CI does not overlap source CI).
2. UTTA / confidence AUROC CIs **overlap source on all 5 seeds**. Do **not** claim a WSI-significant AUROC win.
3. CIs are wide because n_slides = 10. That is the honest unit of inference.
4. Reliability result remains the 5-seed **harm** drop (Tent 0.187 ± 0.179 → gated 0.007 ± 0.003), not AUROC p-values.

## Paper wording (use this)

> Primary 95% CIs resample hospital-2 WSIs (n = 10). Intervals are wide and, for AUROC, overlap between source and gated TTA. Tent collapse on two of five seeds is detectable at the slide level. We therefore report AUROC as mean ± SD across seeds and treat harm-rate reduction as the reliability endpoint.

Do not use `results/confidence_intervals.csv` (seed-level t-interval with Tent AUROC CI above 1.0).
