# Phase 14 — Final statistical & scientific audit

**Status: COMPLETE.** Recalculated from locked headline rows + raw `full_s{7,99,2024}.json` + seed-42 BEST source JSON. No retune. Seeds 7 and 99 kept.

Primary inference for Table 1: **paired seed-level Δ** (n=5, df=4, t₀.₉₇₅=2.776).
Primary WSI inference: **slide resample** (hospital 2, n=10 WSIs). Patch-level CIs are forbidden.

## Numerical audit

| Check | Result |
|---|---|
| Recalculate 5-seed means / SDs | Matches Table 1 (AUROC Source 0.930±0.009, Tent 0.698±0.324, Conf 0.936±0.022, UTTA 0.935±0.022) |
| Seed 42 source vs BEST ckpt JSON | **OK** — test AUROC 0.935456, epoch 3, gate B-FULL (not smoke n=1280) |
| Seed 123 | **OK** — locked Gate-D / full-run numbers (no `full_s123.json` in this dump; WSI rerun AUROC matches to 4 decimals) |
| Seed 2024 JSON vs headline | **OK** (mismatches []) |
| Seed 7 JSON vs headline | **OK** |
| Seed 99 JSON vs headline | **OK** |
| Stale excluded | smoke_s42_max20; Tent lr=1e-4 collapse JSON (failure figure only); online-U Grad-CAM AUROC 0.324; Camelyon17-C first-8k all-negative; EATA-C cov=0.001 |
| EATA seed-42 note | Table 1 uses original salvage 0.9363. WSI rerun 0.9386 / harm 0.011. CIs use the rerun. Do not mix. |

Shift AUROC drop **0.070 ± 0.009**. U–error r **0.397 ± 0.016**.

## Statistical audit

| Check | Result |
|---|---|
| Seed-level 95% t CIs | Implemented (`paired_t`, df=4) |
| Holm on 8–11 contrasts | UTTA−Source F1 Holm ≈ 0.051; UTTA−Conf F1 Holm ≈ 0.043 (ΔF1=0.001, negligible); AUROC vs source n.s. |
| Wilcoxon | Floor p=0.0625 at n=5 when all signs agree — not a failed F1 result |
| Effect sizes | Cohen dz: F1 UTTA−Source **2.26**; AUROC UTTA−Source **0.40** |
| Bootstrap | `bootstrap_ci_by_slide` / paired WSI Δ on seed-42 dumps |
| Seed vs WSI | **Distinguished.** Seed-level: F1 CI excludes 0. WSI AUROC CIs overlap source except Tent collapse on 7/99. Seed-42 pooled WSI ΔAUROC UTTA−Source **+0.014 [0.002, 0.026]** |
| Multiple comparisons | Holm reported; do not claim AUROC “significant vs source” |

## Methodological audit (frozen protocol)

| Rule | Held? |
|---|---|
| BN affine-only | Yes |
| BN running stats frozen (`bn_freeze_stats`) | Yes (official Tent) |
| Target labels never affect adaptation | Yes |
| τ from hospital 1 only (70th pct of val U) | Yes |
| Confidence / entropy thresholds from hospital 1 | Yes |
| No target-based model selection | Yes |
| Empty gate → skip batch | Yes |
| k=1 inner steps | Yes (k≥5 collapses all methods) |
| N_MC=20 | Yes (N-sweep plateau) |

## RQ lock (do not reopen)

1. Shift is real.
2. Fair Tent is unstable (2/5 collapse; WSI-significant on 7 and 99).
3. U tracks error.
4. Gating cuts harm ~25×; F1 up **5/5**.
5. Random also collapses → not “fewer samples”.
6. **UTTA-Med ≈ confidence** (RQ6 tie).
7. AUROC vs source is mixed (3+/2−); seed-level CI includes 0.
