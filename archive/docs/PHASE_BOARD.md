# Phase board — Master Plan v4 (frozen)

There is **no ResNet-17**. Official backbone is **ResNet-18**.

| ID | Phase | Status |
|----|--------|--------|
| 0 | Protocol freeze (RQs, leakage, hospitals `{0,3,4}→{1}→{2}`) | **DONE** |
| 1 | Infra, configs, requirements, git | **DONE** |
| 2 | Camelyon17 parquet loader | **DONE** |
| 3–4 | Source ResNet-18 (1-logit, dropout 0.5, Adam) × 5 seeds | **DONE** |
| 5 | Shift evidence (ID vs target) | **DONE** |
| B4 | Tent (`bn_freeze_stats`, lr=1e-5) | **DONE** (collapse seeds 7, 99 — keep) |
| B5 | EATA (same BN, e=0.4) | **DONE** |
| D | MC-Dropout N=20, τ=val 70th pct U | **DONE** |
| B6/B7/B10/B8 | Random / confidence / entropy / UTTA | **DONE** × 5 |
| E | Harm, ECE, Brier, reliability, k-steps | **DONE** |
| 12 | Grad-CAM source + frozen-U | **DONE** |
| WSI | 10-slide bootstrap, all 5 seeds | **DONE** (s42 rebuilt from npz, 24 KB) |
| 13 | Leakage / reproducibility audit | **DONE** |
| 14–15 | Five-seed stats, tables 1–7 | **DONE** |
| 11 | ResNet-50 | **CODE ONLY — do not run for the paper** |
| 16 | IEEE draft `paper/ieee/utta_med.tex` | **DRAFT — remaining work** |
| 17 | Public repo / authors / citation | **PENDING** |

Optional T / SAR / DLTTA: seed-42 supplement only. EATA-C and Camelyon17-C: **invalid, do not cite**.
