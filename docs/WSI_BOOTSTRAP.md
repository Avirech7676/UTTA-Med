# Gate 10 — WSI-level bootstrap + reliability diagrams

**No retrain.** Uses frozen 5-seed protocol.

## Why this exists
Patches from the same WSI are correlated (stain, scanner, tissue). Patch-level CIs are too narrow. Primary CIs resample **slides** with replacement, then include every patch of the sampled slides.

Hospital 2 has **10 WSIs**. Expect **wide** 95% CIs. Report them anyway — that is the honest sample size for histopathology TTA.

## How to run
1. New Kaggle notebook, GPU T4, Internet on
2. Add Input: dataset with `camelyon17_resnet18_source_s42_BEST.pt`
3. Use the saved WSI bootstrap artifacts under `results/statistical/wsi_bootstrap/`; the historical notebook is retained in `archive/notebooks/`.
4. Run All (~45–70 min)

Optional extra seeds: add those BEST.pt files, set `SEEDS = [42, 123, 2024, 7, 99]`.

## Send back
- `camelyon17_wsi_bootstrap_s42.json`
- printed table `point | WSI 95% CI (AUROC)`
- `figures/reliability_overlay_s42.png`
- `figures/per_slide_auroc_s42.png`

Keep the `.npz` dumps (`dumps/s42_*.npz`) — they let us recompute stats without GPU.

## How to read the CIs
If UTTA AUROC point = 0.95 and WSI CI is `[0.88, 0.98]`, you may **not** claim a significant AUROC win over source. You **may** still claim lower harm if the harm CI does not overlap Tent's.

That is the scientifically correct next table for the paper.
