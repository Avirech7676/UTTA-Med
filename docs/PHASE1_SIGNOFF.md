# Phase-1 Signoff — Slide-level, Reject Option, Tent Collapse

**Status:** LOCKED  
**Date:** 2026-10-07  
**Compute:** CPU-only recompute from official `results/data/wsi_bootstrap/s{seed}_{method}.npz`  
**Not a new TTA run.** No τ retune. Seeds `{42, 123, 2024, 7, 99}` retained.

**Scripts:** `scripts/upgrade/phase1_all.py` (+ `slide_level_from_npz.py`, `coverage_risk_curves.py`, `tent_collapse_case_study.py`)

**Artifacts:**

| Path | Role |
|---|---|
| `results/statistical/phase1_slide_level.json` | Per-slide + slide-macro |
| `results/statistical/phase1_coverage_risk.json` | Reject-option curves |
| `results/statistical/phase1_tent_collapse.json` | Seeds 7/99 autopsy |
| `results/tables/slide_level_5seed.csv` | Flat export |
| `results/tables/tent_collapse_per_slide.csv` | Collapse export |
| `figures/phase1/fig_slide_macro_auroc.png` | Figure |
| `figures/phase1/fig_coverage_risk.png` | Figure |
| `figures/phase1/fig_tent_slide_damage.png` | Figure |

---

## A. Slide-macro AUROC

Mean over 10 hospital-2 slides, then mean ± sd over 5 seeds.

| Method | Slide-macro AUROC | Per-seed (42 / 123 / 2024 / 7 / 99) |
|---|---:|---|
| Source | 0.916 ± 0.018 | 0.934 / 0.922 / 0.923 / 0.888 / 0.913 |
| Tent | 0.689 ± 0.211 | 0.843 / 0.844 / 0.843 / **0.462** / **0.455** |
| EATA | 0.902 ± 0.038 | 0.937 / 0.911 / 0.921 / 0.838 / 0.903 |
| Random | 0.690 ± 0.193 | 0.842 / 0.835 / 0.817 / **0.473** / **0.484** |
| Confidence | **0.925 ± 0.029** | 0.952 / 0.935 / 0.947 / 0.882 / 0.910 |
| Entropy | **0.925 ± 0.029** | 0.952 / 0.935 / 0.947 / 0.882 / 0.910 |
| UTTA-Med | **0.925 ± 0.029** | 0.951 / 0.935 / 0.947 / 0.881 / 0.910 |

**Signoff claims**

1. Selective gates are stable at the slide unit (0.925 ± 0.029).  
2. Tent collapses on seeds 7 and 99 at slide-macro (~0.46), matching patch-level instability.  
3. Random at matched coverage collapses the same way → coverage alone is not safety.  
4. UTTA-Med ≈ confidence ≈ entropy on this binary task (report as finding).

---

## B. Coverage–risk / reject option (source model)

Rank by \(U_{\mathrm{proxy}} = 1 - |2p - 1|\) (same order as confidence on binary labels). Keep lowest-U fraction \(c\).

| Keep fraction \(c\) | Kept AUROC | Error rate |
|---:|---:|---:|
| 1.0 | 0.930 ± 0.009 | 0.166 ± 0.016 |
| 0.9 | 0.939 ± 0.014 | 0.138 ± 0.017 |
| 0.8 | 0.952 ± 0.017 | 0.114 ± 0.020 |
| 0.7 | 0.964 ± 0.018 | 0.088 ± 0.024 |
| 0.6 | 0.974 ± 0.018 | 0.059 ± 0.029 |
| 0.5 | **0.981 ± 0.016** | **0.036 ± 0.029** |
| 0.3 | 0.987 ± 0.012 | 0.013 ± 0.012 |

**Signoff claims**

1. Deferring the most uncertain 50% of patches reduces error ≈4.6× (16.6% → 3.6%).  
2. Reject option is deployable without re-running TTA.  
3. Paper must state ranking used stored probabilities / \(U_{\mathrm{proxy}}\), not a fresh MC-Dropout pass.

---

## C. Tent collapse localization (seeds 7 and 99)

Worst slides by Tent ΔAUROC vs source (representative):

| Seed | Slide | N | Tumor frac | Source AUROC | Tent AUROC | Tent Δ | Confidence Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 99 | 29 | 12,742 | 0.52 | 0.819 | 0.004 | −0.816 | −0.089 |
| 7 | 29 | 12,742 | 0.52 | 0.738 | 0.007 | −0.731 | −0.114 |
| 99 | 28 | 31,878 | 0.85 | 0.947 | 0.239 | −0.708 | +0.011 |
| 99 | 24 | 7,727 | 0.02 | 0.760 | 0.066 | −0.694 | +0.065 |
| 99 | 26 | 3,815 | 0.13 | 0.744 | 0.055 | −0.688 | +0.031 |

**Signoff claims**

1. Collapse is concentrated on a few slides (especially 29, 28), not uniform noise.  
2. Selective gating largely avoids this failure mode on the same slides.  
3. Seeds 7 and 99 **must** remain in every headline table.

---

## Honest limitations (carry to paper later)

- Binary Camelyon17 patches only; confidence ≡ entropy ≡ UTTA ranking.  
- Reject curves use \(U_{\mathrm{proxy}}\) from stored \(p\), not independent MC variance.  
- n = 10 test slides; WSI bootstrap CIs already in `camelyon17_wsi_bootstrap_s*.json`.  
- Backbone ResNet-18; not a foundation-model TTA study.

---

## Frozen headline (patch-level, unchanged)

From `docs/HEADLINE_5SEED.md` — do not edit:

| Method | AUROC | F1 | Harm | Coverage |
|---|---:|---:|---:|---:|
| Source | 0.930 ± 0.009 | 0.807 ± 0.025 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.187 ± 0.179 | 1.00 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.025 ± 0.015 | 0.89 |
| Random | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.194 ± 0.170 | 0.70 |
| Confidence | **0.936 ± 0.022** | **0.831 ± 0.032** | **0.007 ± 0.003** | 0.58 |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.007 ± 0.003 | 0.58 |

---

**Signoff:** Phase-1 accepted. Next non-paper work = release packaging. Paper integration is deferred until requested.
