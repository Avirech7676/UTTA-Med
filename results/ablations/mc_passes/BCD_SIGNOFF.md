# B / C / D remaining items — seed 42 sign-off

Checkpoint: `camelyon17_resnet18_source_s42_BEST.pt` (epoch 3, val AUROC 0.9793).
No retraining. Protocol frozen.

## B — feature-space shift (supporting evidence)

UMAP of frozen ResNet-18 512-D embeddings, 2,500 patches per split.

- Primary evidence of shift remains the **quantitative drop** (ID AUROC 0.999 → target 0.935).
- UMAP is supporting: hospitals occupy a shared tissue manifold with **hospital-4 isolated mode** and hospital-2 (test) mass shifted along the same curve. Not a clean three-blob separation — do not over-claim.

Figures: `fig_b_umap_split.png`, `fig_b_umap_hospital.png`.

## C — already locked (no change)

Tent + EATA, `bn_freeze_stats`, lr=`1e-5`. SAR/DLTTA remain cut.

## D1 — MC-Dropout N (hospital 1)

Mean prediction AUROC is **identical** at every N (0.9792). Ranking of U vs N=50:

| N  | corr(U, error) | Spearman vs N=50 |
| -- | -------------- | ---------------- |
| 5  | 0.363          | 0.981            |
| 10 | 0.399          | 0.990            |
| 20 | 0.425          | 0.996            |
| 30 | 0.432          | 0.998            |
| 50 | 0.437          | 1.000            |

**N=20 stays frozen.** Gain 5→20 is +0.062 corr; 20→50 is only +0.012. Spearman 0.996 vs the N=50 ranking.

## D2 — random at UTTA test coverage (matched)

| Method            | Coverage | AUROC | F1    | Harm  |
| ----------------- | -------- | ----- | ----- | ----- |
| Source            | —        | 0.935 | 0.835 | —     |
| Random @ 0.70     | 0.701    | 0.922 | 0.805 | 0.062 |
| **Random @ 0.562**| **0.562**| **0.922** | **0.803** | **0.057** |
| UTTA-Med          | 0.562    | 0.951 | 0.870 | 0.003 |
| Confidence        | 0.568    | 0.951 | 0.871 | 0.003 |

Random at the **same test coverage as UTTA** is still worse than source. The win is **which** samples, not how many.

## Paper use

- Figure 2: UMAP (supporting) + performance-drop table (primary).
- Figure / supplement: MC-N curve; state N=20 frozen because ranking saturates.
- Gate-control caption: include Random@0.562 vs UTTA@0.562.

Do not retune N or τ from this run.
