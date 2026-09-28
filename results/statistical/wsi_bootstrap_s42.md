# WSI-level bootstrap — seed 42 (hospital 2, 10 slides)

Unit of resampling: WSI IDs `{20–29}`,  with replacement; all patches of sampled slides included.
Protocol frozen: `bn_freeze_stats`, lr=`1e-5`, N_MC=20, τ = val 70th percentile of U.

## Point estimate + WSI 95% CI (AUROC)

| Method     | AUROC  | WSI 95% CI     | F1     | Harm  | Coverage |
| ---------- | ------ | -------------- | ------ | ----- | -------- |
| Source     | 0.9355 | [0.843, 0.968] | 0.8350 | —     | —        |
| Tent       | 0.9294 | [0.795, 0.959] | 0.8248 | 0.059 | 1.000    |
| EATA       | 0.9386 | [0.833, 0.976] | 0.8475 | 0.011 | 0.865    |
| Random     | 0.9222 | [0.796, 0.954] | 0.8048 | 0.062 | 0.701    |
| Confidence | 0.9508 | [0.867, 0.982] | 0.8707 | 0.003 | 0.568    |
| Entropy    | 0.9508 | [0.867, 0.982] | 0.8707 | 0.003 | 0.568    |
| UTTA-Med   | 0.9502 | [0.867, 0.981] | 0.8697 | 0.003 | 0.562    |

Point estimates for source / Tent / random / confidence / entropy / UTTA match the locked seed-42 table.
EATA coverage/harm differ slightly from the earlier salvage run — for the 5-seed table keep the locked JSON; use this run for the seed-42 WSI CI figure.

## How to report

- CIs are **wide** because n_slides = 10. That is the honest unit of inference.
- All method CIs **overlap**. Do **not** claim seed-42 AUROC of UTTA/confidence is significant vs source at the WSI level.
- The 5-seed **harm** reduction (Tent 0.187 → gated 0.007) remains the reliability claim; this bootstrap supports **uncertainty of AUROC**, not a p-value win.

Do not replace WSI CIs with patch-level CIs in the paper.
