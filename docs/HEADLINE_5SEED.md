# Locked five-seed headline (hospital 2)

Seeds: {42, 123, 2024, 7, 99}. Protocol frozen. Do not retune.

## Mean ± SD

| Method | AUROC | F1 | Harm | Coverage |
|---|---:|---:|---:|---:|
| Source | 0.930 ± 0.009 | 0.807 ± 0.025 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.187 ± 0.179 | 1.00 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.025 ± 0.015 | 0.89 |
| Random | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.194 ± 0.170 | 0.70 |
| Confidence | **0.936 ± 0.022** | **0.831 ± 0.032** | **0.007 ± 0.003** | 0.58 |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | **0.007 ± 0.003** | 0.58 |

Entropy ≡ confidence.

## Claims

1. Shift is real (AUROC drop 0.070 ± 0.009).
2. Tent collapsed on seeds 7 and 99 — keep them.
3. U tracks error (r ≈ 0.40).
4. Gating cuts harm ~25×; F1 up on 5/5.
5. Random fails at the same coverage.
6. UTTA-Med ≈ confidence.
7. AUROC vs source is mixed.
8. k=1, N=20, τ = val 70th percentile of U.

Camelyon17-C and PathMNIST: supplement / appendix only.
