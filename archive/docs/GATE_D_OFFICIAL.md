# Gate D — OFFICIAL seed 42 (do not overclaim)

Checkpoint: `camelyon17_resnet18_source_s42_BEST.pt`, lr=1e-5, BN stats frozen, N_MC=20.  
τ = 70th percentile of source MC-Dropout variance on hospital 1 (`8.88e-6`). Test used once.

## RQ3 — Uncertainty vs error (OOD val)

Pearson corr(U, error) = **0.416**. Error by U-decile: **0.09% → 40.9%**.  
This is a clean `U ↑ ⇒ Error ↑`. MC-Dropout is informative. Do not skip gating.

## Val τ sweep (hospital 1, source AUROC 0.9793)

| τ percentile | Coverage | AUROC | F1 | ECE | Harm |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.20 | 0.9846 | 0.933 | 0.045 | 0.0034 |
| 40 | 0.40 | 0.9851 | 0.934 | 0.044 | 0.0039 |
| 60 | 0.60 | 0.9862 | 0.936 | 0.044 | 0.0035 |
| **70 (selected)** | **0.70** | **0.9867** | **0.937** | **0.043** | **0.0033** |
| 80 | 0.80 | 0.9858 | 0.935 | 0.045 | 0.0033 |

Selection rule (max val AUROC, then min harm) was followed. Frozen.

## Official test table (hospital 2)

| Method | AUROC | AUPRC | F1 | Sens | ECE | Cov | Harm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source-only | 0.9355 | 0.9446 | 0.8350 | 0.750 | 0.0988 | — | — |
| Tent | 0.9294 | 0.9478 | 0.8248 | 0.708 | 0.1396 | 1.00 | 0.0591 |
| EATA (Gate C) | 0.9363 | 0.9536 | 0.8447 | 0.738 | 0.1250 | 0.94 | 0.0514 |
| Random gate | 0.9223 | 0.9416 | 0.8051 | 0.680 | 0.1534 | 0.70 | 0.0617 |
| Confidence gate | **0.9508** | **0.9605** | **0.8708** | **0.788** | **0.0921** | 0.57 | **0.0029** |
| Entropy gate | **0.9508** | **0.9605** | **0.8708** | **0.788** | **0.0921** | 0.57 | **0.0029** |
| UTTA-Med | 0.9507 | 0.9604 | 0.8697 | 0.787 | 0.0930 | 0.56 | 0.0032 |

## How to write this (required honesty)

1. **Gating works.** vs Tent: AUROC +0.021, harm 0.059 → 0.003, ECE 0.140 → 0.093. vs source: AUROC +0.015, F1 +0.035, harm low. Random at 70% coverage is *worse* than source — the win is not “adapt on fewer samples”.
2. **UTTA-Med does not beat confidence/entropy on this seed.** Differences are <0.001 AUROC. RQ6 is a negative/mixed result. That is still a paper.
3. **Binary entropy ≡ confidence.** For one Bernoulli probability, `H(p)` is a strictly monotone function of `max(p,1−p)`. The two rows must be near-identical. State this in Related Work / Ablation. The real controls are **Random vs Confidence vs MC-Dropout**.
4. **Coverage on test (0.56) ≠ val (0.70)** because hospital 2 is more uncertain at the same τ. Report both. Confidence/entropy/UTTA are matched to each other on test (~0.56–0.57). Random was matched to *val* coverage (0.70) — say so; do not pretend it is matched on test.
5. Do **not** claim “first uncertainty-aware TTA” or “MC-Dropout uniquely helps”. Claim: *uncertainty-aware / high-confidence sample selection makes TTA safe under Camelyon17 shift; MC-Dropout variance is a valid signal but is redundant with simple confidence for this binary model.*

## Remaining for the manuscript (priority)

1. Reliability diagrams (source / Tent / confidence / UTTA) — no new training.
2. τ / coverage / harm figure from the val sweep already in this JSON.
3. WSI-level bootstrap CIs — needs per-patch predictions + slide IDs (dump next run).
4. Seeds `{123, 2024, 7, 99}` for the headline table (same protocol).
5. Grad-CAM audit (strongly recommended, not needed to claim the table).
6. Optional cheap check: random gate at test coverage 0.56 (no labels required to set p=0.56).
