# UTTA-Med Stage 15 — Final Results Package

## Primary five-seed table

Target: Camelyon17-WILDS center 2, 85,054 patches, 10 WSIs. Values are mean ± SD over seeds {42,123,2024,7,99}.

| Method | AUROC | F1 | ECE | Harm | Coverage |
|---|---:|---:|---:|---:|---:|
| Source-only | 0.930 ± 0.009 | 0.807 ± 0.025 | 0.128 ± 0.026 | N/A | N/A |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.284 ± 0.179 | 18.7% ± 17.9% | 100.0% ± 0.0% |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.152 ± 0.024 | 2.5% ± 1.5% | 88.5% ± 3.4% |
| Random-gated | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.298 ± 0.166 | 19.4% ± 17.0% | 70.0% ± 0.1% |
| Confidence-gated | 0.936 ± 0.022 | 0.831 ± 0.032 | 0.130 ± 0.030 | 0.7% ± 0.3% | 58.1% ± 4.7% |
| Entropy-gated | 0.936 ± 0.022 | 0.831 ± 0.032 | 0.130 ± 0.030 | 0.7% ± 0.3% | 58.1% ± 4.7% |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.131 ± 0.031 | 0.7% ± 0.3% | 57.5% ± 4.7% |

## Per-seed UTTA − Source

| Seed | ΔAUROC | ΔF1 | UTTA harm | Tent harm | UTTA coverage |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.0152 | +0.0347 | 0.0032 | 0.0591 | 0.562 |
| 123 | +0.0072 | +0.0272 | 0.0063 | 0.0439 | 0.583 |
| 2024 | +0.0235 | +0.0283 | 0.0070 | 0.0679 | 0.581 |
| 7 | −0.0118 | +0.0162 | 0.0097 | 0.3751 | 0.510 |
| 99 | −0.0051 | +0.0092 | 0.0103 | 0.3914 | 0.641 |

## Main result statements to use

1. UTTA-Med increased F1 relative to Source in all five seeds; mean paired ΔF1=+0.0231 with 95% seed-level t CI [+0.0104,+0.0358].
2. UTTA-Med increased AUROC in 3/5 seeds; the mean paired ΔAUROC was +0.0058 with CI [−0.0121,+0.0237].
3. Harmful adaptation for UTTA-Med was 0.7% ± 0.3%, versus 18.7% ± 17.9% for Tent and 19.4% ± 17.0% for random gating.
4. Confidence gating and UTTA-Med had nearly identical mean AUROC and F1; UTTA was not superior to confidence gating at matched operating conditions.
5. The target coverage of UTTA was 57.5% ± 4.7%; the 70% validation target was not preserved under the target shift, so both validation coverage and realized target coverage must be reported.

## Required caveat

The final manuscript should not call UTTA-Med universally superior on AUROC. Seeds 7 and 99 show modest AUROC decreases despite F1 gains and substantially lower harmful adaptation than Tent.

## Figure set

Use the figure manifest in `figure_manifest.csv`. Main-text candidates: Figure 5A, Figure 5B, Figure 3, Figure 4, and Figure 6. Place UMAP and representative Grad-CAM panels in supplementary/audit material unless space permits.

## Stage 15 disposition

**Primary results package: COMPLETE.**
**Publication freeze condition:** use this package only after the final repository audit records the exact code commit/environment and confirms that the master results are tied to that commit.
