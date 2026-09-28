# Gate C — OFFICIAL (seed 42)

Checkpoint: `camelyon17_resnet18_source_s42_BEST.pt` (epoch 3).  
HPs selected on OOD val (hospital 1) only. Test hospital 2 used once.

## Frozen adaptation protocol (all later methods must match)

| Knob | Official value | Why |
|---|---|---|
| Update scope | BN affine γ, β only | Plan v4 parity |
| BN running stats | **Frozen** (`bn_freeze_stats`) | `bn_train` collapses even at 1e-5 |
| Optimizer | Adam | Tent default |
| lr | **1e-5** | Only non-collapsing val lr |
| Stream | Continual, 1 step/batch, shuffle=False | |
| EATA e_margin | **0.4** | Best val AUROC among {0.2, 0.4, 0.55} |

## Val Tent sweep (hospital 1, source AUROC 0.9793)

| BN mode | lr | AUROC after | F1 | Harm |
|---|---:|---:|---:|---:|
| **freeze_stats** | **1e-5** | **0.9771** | **0.9245** | **0.020** |
| freeze_stats | 3e-5 | 0.422 | 0.102 | 0.488 |
| freeze_stats | 5e-5 | 0.367 | 0.009 | 0.516 |
| freeze_stats | 1e-4 | 0.327 | 0.002 | 0.518 |
| bn_train | 1e-5 … 1e-4 | ~0.57 | ~0.62 | ~0.37 |

Original Tent (`BN.train()`, batch stats) collapses. Report that as a failure-mode figure, not as the Tent baseline.

## Official test (hospital 2)

| Method | AUROC | AUPRC | F1 | Sens | Spec | ECE | Brier | Cov | Harm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Source-only | 0.9355 | 0.9446 | 0.8350 | 0.750 | 0.954 | 0.099 | 0.122 | — | — |
| Tent | 0.9294 | 0.9478 | 0.8248 | 0.708 | 0.991 | 0.140 | 0.135 | 1.00 | 0.059 |
| EATA | 0.9363 | 0.9536 | 0.8447 | 0.738 | 0.991 | 0.125 | 0.122 | 0.939 | 0.051 |

Honest reading (RQ2): with a fair Tent, **TTA does not recover the shift**. EATA is a tiny F1 bump; Tent is slightly worse. That is the right baseline for UTTA-Med — not a 0.57 strawman.
