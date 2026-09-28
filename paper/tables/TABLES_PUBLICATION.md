# Phase 15 — Final results package (locked numbers)

Do not retune. Table 8 (ResNet-50) is **empty until Phase 11 GPU finishes**.

## Table 1 — Dataset / protocol

| Role | Hospitals | n patches | n WSI |
|---|---|---:|---:|
| Source train | 0, 3, 4 | 302,436 | 30 |
| In-domain val | 0, 3, 4 | 33,560 | 30 (same slides, held-out patches) |
| OOD val (τ, lr, T) | **1** | 34,904 | 10 |
| Target test | **2** | 85,054 | 10 (slides 20–29) |
| Task | Binary tumor / non-tumor, 96×96 | | |
| Backbone | ResNet-18, 1-logit BCE, dropout 0.5 | | |
| TTA | BN affine only, stats frozen, Adam lr=1e-5, k=1 | | |
| Seeds | {42, 123, 2024, 7, 99} | | |

## Table 2 — Five-seed primary (hospital 2, mean ± SD)

| Method | AUROC | F1 | ECE | Coverage | Harm |
|---|---|---|---|---|---|
| Source-only | 0.930 ± 0.009 | 0.807 ± 0.025 | 0.128 ± 0.026 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.284 ± 0.179 | 1.000 | 0.187 ± 0.179 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.152 ± 0.023 | 0.885 ± 0.034 | 0.025 ± 0.015 |
| Random-gated | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.298 ± 0.166 | 0.700 ± 0.001 | 0.194 ± 0.170 |
| Confidence-gated | **0.936 ± 0.022** | **0.831 ± 0.032** | 0.130 ± 0.030 | 0.581 ± 0.047 | **0.007 ± 0.003** |
| Entropy-gated | 0.936 ± 0.022 | 0.831 ± 0.032 | 0.130 ± 0.030 | 0.581 ± 0.047 | 0.007 ± 0.003 |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.131 ± 0.030 | 0.575 ± 0.047 | **0.007 ± 0.003** |

Entropy ≡ confidence (binary). Caption: F1 and harm are the reproducible endpoints; AUROC is mixed; UTTA ≈ confidence.

## Table 2b — Per-seed AUROC

| Method | 42 | 123 | 2024 | 7 | 99 |
|---|---:|---:|---:|---:|---:|
| Source | 0.9355 | 0.9349 | 0.9320 | 0.9134 | 0.9328 |
| Tent | 0.9294 | 0.9410 | 0.9331 | **0.3429** | **0.3444** |
| EATA | 0.9363 | 0.9403 | 0.9316 | 0.8766 | 0.9203 |
| Random | 0.9223 | 0.9372 | 0.8967 | **0.3543** | **0.3731** |
| Confidence | 0.9508 | 0.9423 | 0.9559 | 0.9020 | 0.9267 |
| UTTA-Med | 0.9507 | 0.9421 | 0.9555 | 0.9016 | 0.9276 |

## Table 3 — Harmful adaptation (per seed)

| Method | 42 | 123 | 2024 | 7 | 99 | mean |
|---|---:|---:|---:|---:|---:|---:|
| Tent | 0.059 | 0.044 | 0.068 | **0.375** | **0.391** | 0.187 |
| EATA | 0.051 | 0.018 | 0.024 | 0.020 | 0.013 | 0.025 |
| Random | 0.062 | 0.048 | 0.104 | **0.373** | **0.385** | 0.194 |
| Confidence | 0.003 | 0.006 | 0.007 | 0.009 | 0.010 | 0.007 |
| UTTA-Med | 0.003 | 0.006 | 0.007 | 0.010 | 0.010 | 0.007 |

F1 Δ UTTA−Source: **+0.035, +0.027, +0.028, +0.016, +0.009** (5/5 positive).

## Table 4 — Calibration (ECE mean ± SD)

See Table 2 ECE column. Seed-42 temperature scaling (fit hospital 1 only): T=1.87, ECE 0.099 → **0.069**, AUROC unchanged. Reliability diagrams: `results/statistical/wsi_bootstrap/figures/reliability_overlay_s*.png`.

## Table 5 — Uncertainty (hospital 1)

| Seed | r(U, error) | Decile-1 err | Decile-10 err |
|---:|---:|---:|---:|
| 42 | 0.416 | 0.09% | 40.9% |
| 123 | 0.398 | 0.00% | 37.5% |
| 2024 | 0.398 | 0.00% | 35.0% |
| 7 | 0.372 | 0.00% | 37.1% |
| 99 | 0.401 | 0.00% | 41.9% |
| mean | **0.397 ± 0.016** | | |

## Table 6 — MC-pass sensitivity (seed 42, hospital 1)

| N | corr(U,err) | Spearman vs N=50 | val AUROC |
|---:|---:|---:|---:|
| 5 | 0.363 | 0.981 | 0.9792 |
| 10 | 0.399 | 0.990 | 0.9792 |
| **20** | **0.425** | **0.996** | 0.9792 |
| 30 | 0.432 | 0.998 | 0.9792 |
| 50 | 0.437 | 1.000 | 0.9792 |

**N=20 frozen.**

## Table 7 — Adaptation-step sensitivity (seed 42, hospital 1)

| k | Tent AUROC | EATA | UTTA | Harm (Tent) |
|---:|---:|---:|---:|---:|
| **1** | **0.977** | **0.977** | **0.977** | **0.020** |
| 2 | 0.972 | 0.970 | 0.956 | 0.039 |
| 5 | 0.375 | 0.361 | 0.304 | 0.517 |
| 10 | 0.335 | 0.314 | 0.280 | 0.518 |

**k=1 frozen.** Gating does not save k≥5.

## Table 8 — ResNet-50 sensitivity

**Not run.** Code exists (`src/models/resnet50.py`). Run Phase 11 GPU notebook. Until then this table is blank in the paper.

## Table 9 — Statistical comparisons (seed-level, n=5)

| Contrast | Mean Δ | 95% t CI | t p | Holm | signs |
|---|---:|---|---:|---:|---|
| UTTA − Source AUROC | +0.006 | [−0.012, +0.024] | 0.42 | n.s. | 3+/2− |
| UTTA − Confidence AUROC | 0.000 | [−0.0008, +0.0007] | 0.87 | n.s. | tie |
| UTTA − Source F1 | **+0.023** | **[+0.010, +0.036]** | **0.007** | 0.051 | **5/5** |
| UTTA − Confidence F1 | −0.001 | [−0.0015, −0.0005] | 0.005 | 0.043 | 0/5* |
| UTTA − Tent harm | −0.180 | [−0.399, +0.039] | 0.085 | n.s. | 5/5 lower |

\*ΔF1 = 0.001 — significant and meaningless.

Seed-42 paired WSI: UTTA−Source AUROC **+0.014 [0.002, 0.026]**. Other seeds: WSI CIs overlap source except Tent collapse.

## Figures (already generated)

| Fig | File |
|---|---|
| Main bars | `figures/fig_main_mean_sd.png` / `fig5_five_seed_main.png` |
| Per-seed AUROC | `figures/fig5_per_seed_auroc.png` |
| Harm per seed | `figures/fig_harm_per_seed.png` / `fig8_harm_5seed.png` |
| U vs error | `figures/fig3_u_error_5seed.png` |
| τ trade-off | `figures/fig4_tau_tradeoff.png` |
| MC-N | `results/figures/fig_d_mc_n_sensitivity.png` |
| k-steps | `figures/fig_adapt_steps_val_s42.png` |
| Reliability | `results/statistical/wsi_bootstrap/figures/reliability_overlay_s*.png` |
| Grad-CAM | `figures/gradcam/source_*.png` + `gradcam_frozen/` |
| UMAP | `results/figures/fig_b_umap_*.png` |
| ResNet-50 | pending Phase 11 |
