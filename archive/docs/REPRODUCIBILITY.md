# Reproducibility audit (Gate 22)

Date: 2026-09-27. Protocol: Master Plan v4, frozen.

## Must-keep artifacts

| Artifact | Role |
|---|---|
| `camelyon17_resnet18_source_s{42,123,2024,7,99}_BEST.pt` | source models |
| `camelyon17_resnet18_full_s{...}.json` | 5-seed point estimates |
| `camelyon17_wsi_bootstrap_s{...}.json` | WSI CIs |
| `configs/camelyon17.yaml` | split truth |
| `src/` + `tests/` | code |

## Locked hyperparameters

| Knob | Value | Chosen on |
|---|---|---|
| Seeds | 42, 123, 2024, 7, 99 | a priori |
| Hospitals | train {0,3,4}, val {1}, test {2} | WILDS |
| Optimizer (source) | Adam lr=1e-4, wd=1e-4 | a priori |
| Early stop | val AUROC, patience 5 | hospital 1 |
| TTA lr | 1e-5 | hospital 1 sweep (seed 42) then frozen |
| BN | affine only, stats frozen for gated TTA | a priori |
| N_MC | 20 | val sensitivity |
| τ | 70th percentile of val U | hospital 1, per seed |
| EATA e_margin | 0.4 | a priori (binary H scale) |
| Inner steps k | 1 | val k-sweep; k≥5 collapses |

## Leakage

- Train ∩ val = ∅, train ∩ test = ∅ (Gate A).
- Target labels unused for train, TTA, τ, T, k, lr.
- No test-driven retune after seeds 7/99.

## What a third party needs

1. HuggingFace `wltjr1007/Camelyon17-WILDS` parquet.
2. This repo + five BEST.pt files.
3. `python -m pytest tests/` (no data).
4. `python scripts/adapt.py --checkpoint ... --method utta --tau <seed-tau>`.

## Known deviations from original papers (state in Setup)

- Tent / EATA / SAR / DLTTA restricted to **BN-affine** (fairness).
- EATA Fisher omitted in the 5-seed headline (EATA-F is optional).
- Binary entropy (sigmoid) instead of softmax entropy.
- Dropout only in the classification head (ResNet-18 MC-Dropout).
