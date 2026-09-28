# UTTA-Med — Official 5-seed headline (LOCKED)

Seeds `{42, 123, 2024, 7, 99}`. Protocol frozen: BN stats frozen, TTA lr=1e-5,
N_MC=20, τ = 70th percentile of **that seed's** hospital-1 U, EATA e_margin=0.4.
Test = Camelyon17 hospital 2, once after τ is frozen.

Target labels never used for training, adaptation, or τ.

## Main table (hospital 2) — mean ± std, n = 5

| Method | AUROC | F1 | ECE | Harm | Coverage |
|---|---:|---:|---:|---:|---:|
| Source-only | 0.930 ± 0.009 | 0.807 ± 0.025 | 0.128 ± 0.026 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.284 ± 0.179 | 0.187 ± 0.179 | 1.00 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.152 ± 0.024 | 0.025 ± 0.015 | 0.89 |
| Random-gated | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.298 ± 0.166 | 0.194 ± 0.170 | 0.70 |
| Confidence-gated | **0.936 ± 0.022** | **0.831 ± 0.032** | 0.130 ± 0.030 | **0.007 ± 0.003** | 0.58 |
| Entropy-gated | 0.936 ± 0.022 | 0.831 ± 0.032 | 0.130 ± 0.030 | 0.007 ± 0.003 | 0.58 |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.131 ± 0.031 | 0.007 ± 0.003 | 0.58 |

Entropy ≡ confidence in binary classification (report as one control).

## Per-seed AUROC

| Method | 42 | 123 | 2024 | 7 | 99 |
|---|---:|---:|---:|---:|---:|
| Source | 0.935 | 0.935 | 0.932 | 0.913 | 0.933 |
| Tent | 0.929 | 0.941 | 0.933 | **0.343** | **0.344** |
| EATA | 0.936 | 0.940 | 0.932 | 0.877 | 0.920 |
| Random | 0.922 | 0.937 | 0.897 | **0.354** | **0.373** |
| Confidence | **0.951** | **0.942** | **0.956** | 0.902 | 0.927 |
| UTTA-Med | 0.951 | 0.942 | 0.955 | 0.902 | 0.928 |

## What is actually true (write the paper this way)

1. **RQ1 — shift is real.** Source ID AUROC ≈ 0.999 vs target 0.930 ± 0.009 (drop 0.070 ± 0.009). F1 drop 0.18 ± 0.02. Sensitivity is the failure mode.

2. **RQ2 — ungated TTA is not reliable.** Fair Tent (BN-affine, lr=1e-5) **collapses on 2/5 seeds** (7, 99). Mean AUROC 0.70 is not a “Tent is bad architecture” claim — it is instability. EATA is more stable (never collapsed) but does not beat source on AUROC (0.921 vs 0.930).

3. **RQ3 — MC-Dropout ranks errors.** r(U, error) = 0.397 ± 0.016 on hospital 1. Decile 1 error ≈ 0%; decile 10 ≈ 35–42%. Mandatory gate passed on all 5 seeds.

4. **RQ4 — gating reduces harm.** Tent harm 0.187 ± 0.179 vs gated 0.007 ± 0.003. Random at 70% coverage also collapses on 2/5 seeds → the mechanism is **which** samples, not fewer samples.

5. **RQ5 — F1 is the clean win; AUROC is mixed.** Gated methods improve F1 vs source on **5/5 seeds** (+0.024 mean). AUROC is better on 3/5, slightly worse on the two collapse-prone seeds. Do not claim “UTTA always raises AUROC”.

6. **RQ6 — UTTA-Med ≈ confidence.** Δ AUROC < 0.001 on every seed. MC-Dropout is a valid epistemic signal but **redundant with softmax confidence** for this binary ResNet-18. That is a result, not a failure. The paper’s contribution is the **gated-TTA reliability profile** (coverage + harm + calibration + matched controls), not a new SOTA number.

## Do not claim

- First uncertainty-aware TTA for medical images
- UTTA-Med outperforms confidence
- Tent is a fair published-hyperparameter comparison if you only show the collapsed mean without the 3 non-collapsed seeds
- Clinical validity (Grad-CAM is an audit)

## Remaining (no more source training required)

1. Paper figures from these JSONs (U-bin, τ, main bars, harm)
2. WSI-level bootstrap if per-patch `y, p, slide_id` dumps exist (10 slides on test)
3. Grad-CAM on one saved BEST.pt (optional)
4. Manuscript: Related Work vs EATA/SAR, honest mixed AUROC, strong harm/F1 story
