# Optional methods — seed 42 sign-off

Checkpoint: `camelyon17_resnet18_source_s42_BEST.pt` (epoch 3, val AUROC 0.979).
Protocol: `bn_freeze_stats`, lr=`1e-5`, k=1. **Not** in the 5-seed headline table.

## Valid (use in supplement)

Hospital 2 vs locked seed-42 source (AUROC 0.935, F1 0.835, ECE 0.099):

| Method | AUROC | F1 | ECE | Harm | Cov | Read |
|---|---:|---:|---:|---:|---:|---|
| Source | 0.935 | 0.835 | 0.099 | — | — | locked |
| Tent | 0.929 | 0.825 | 0.140 | 0.059 | 1.00 | locked 5-seed |
| Confidence / UTTA | 0.951 | 0.870 | 0.092 | **0.003** | 0.56 | locked 5-seed |
| **Temperature T=1.87** | 0.935 | 0.835 | **0.069** | — | — | ECE −0.030; AUROC unchanged |
| **SAR** | 0.948 | 0.828 | 0.135 | 0.054 | 0.93 | AUROC up, F1 down, harm ≈ Tent; 116 resets |
| **DLTTA** | **0.952** | **0.876** | 0.095 | 0.024 | 1.00 | seed-42 only; harm 8× UTTA |
| EATA-C | 0.935 | 0.835 | 0.099 | 0.000 | **0.001** | **not a working run** (below) |

Temperature was fit on **hospital 1**. Never on test. AUROC is invariant; calibration improved.

SAR is closer to Tent than to gated UTTA: high coverage, harm 0.054, 116 source resets.

DLTTA looks strong on **this seed**. Do not promote it to Table 1 without the other four seeds.

## Invalid / do not plot

**EATA-C:** coverage **0.0007** (~60 / 85,054 patches). Cosine diversity (`d_margin=0.4`) rejected almost everyone, so the method is source-only with 23 flips. Not a fair EATA-C comparison. Do **not** retune `d_margin` on hospital 2. A val-only sweep would be a new experiment.

**Camelyon17-C:** AUROC **NaN**, F1 **0**. First 8,000 hospital-2 patches are **all non-tumor** (slide-ordered parquet, same bug as Grad-CAM v1). Figure 9 is **not** signed off.

## Caption language

> Optional seed-42 baselines (not in the five-seed table). Temperature scaling fitted on hospital 1 lowers ECE (0.099 → 0.069) with unchanged AUROC. SAR improves AUROC but not F1 or harm relative to Tent. DLTTA matches gated AUROC on this seed with higher harm (0.024 vs 0.003). EATA-C at default diversity margin adapted <0.1% of samples and is omitted.

## Next

- Optional 15 min: shuffled Camelyon17-C only (Figure 9).
- Else: **documents last**.
