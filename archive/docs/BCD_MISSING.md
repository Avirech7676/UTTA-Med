# Remaining B / C / D

## C — already complete (no GPU)

Mandatory Tent + EATA, BN-affine parity, lr selected on hospital 1.
SAR / DLTTA / EATA-C stay cut.

Figure from locked salvage JSON:
`figures/fig_c_tent_stability.png`

Official Tent = `bn_freeze_stats` + `lr=1e-5`. `BN.train()` collapses at every lr.

## B + D — one Kaggle run, no retrain

Import `kaggle_bcd_missing.ipynb`.

| Item | Output |
|---|---|
| B UMAP | `fig_b_umap_split.png`, `fig_b_umap_hospital.png` |
| D1 MC N | `fig_d_mc_n_sensitivity.png` — if corr plateaus by N=20, keep N=20 |
| D2 random@0.562 | JSON field `random_matched_test_coverage` vs locked UTTA 0.9507 / harm 0.003 |

Send back the JSON + the three PNGs.
