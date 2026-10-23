# UTTA-Med

**Uncertainty-Aware Test-Time Adaptive Deep Learning for Reliable Medical Image Classification Under Distribution Shift**

MC-Dropout predictive-variance gate on BN-affine TTA, under real cross-hospital shift on **Camelyon17-WILDS**.

```
Target image → MC-Dropout → U(x) → if U(x) < τ: adapt  else: skip
```

## Headline (hospital 2, mean ± SD, n = 5)

Seeds `{42, 123, 2024, 7, 99}`. Protocol frozen: `bn_freeze_stats`, TTA lr `1e-5`, \(N_{MC}=20\), \(\tau=\) val 70th percentile of \(U\).

| Method | AUROC | F1 | Harm | Coverage |
|---|---:|---:|---:|---:|
| Source-only | 0.930 ± 0.009 | 0.807 ± 0.025 | — | — |
| Tent | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.187 ± 0.179 | 1.00 |
| EATA | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.025 ± 0.015 | 0.89 |
| Random | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.194 ± 0.170 | 0.70 |
| Confidence / Entropy | **0.936 ± 0.022** | **0.831 ± 0.032** | **0.007 ± 0.003** | 0.58 |
| UTTA-Med | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.007 ± 0.003 | 0.58 |

Reproducible property: **selective adaptation, low harm, F1 up on 5/5**. Not universal AUROC gain. **UTTA-Med ≈ confidence.** Tent collapses on seeds 7 and 99 — keep them.

Full stats: [`results/statistical/five_seed_final/STATISTICAL_REPORT.md`](results/statistical/five_seed_final/STATISTICAL_REPORT.md)

## Frozen design

- Hospitals: train `{0,3,4}` → val `{1}` → test `{2}` (`configs/camelyon17.yaml`)
- Target labels never used for training, adaptation, \(\tau\), or temperature
- BN-affine only, running stats frozen
- Empty gate → skip optimizer step
- Primary CIs: WSI-level (10 test slides)

## Repo

```
src/            models, tta, evaluation, calibration
scripts/        train_source, adapt, inspect_local_data
notebooks/      Kaggle runners (no-retrain jobs)
configs/        camelyon17.yaml is the split source of truth
results/        official 5-seed tables
paper/ieee/     Overleaf (utta_med.tex + .bib)
tests/          pytest (no data required for core tests)
```

```bash
pip install -r requirements.txt
pytest tests/ -q
```

## What is done vs remaining

Mandatory 5-seed science, Grad-CAM, WSI bootstrap, MC-N, k-step collapse, matched random coverage, optional T/SAR/DLTTA (seed 42): **done**.

**Optional remaining GPU (not in Table 1):** ResNet-50 seed 42 — `notebooks/kaggle_resnet50_s42.ipynb`. Skip if you are writing the paper.

Permanently skipped: Camelyon16, MIDOG++, ViT/CLIP/LLM, 10 seeds, PathMNIST debug.

Architecture freeze: **1-logit, dropout 0.5, Adam** (`docs/ARCHITECTURE_FREEZE.md`). τ is **not** 0.05 (`configs/tau_by_seed.yaml`).

## Model weights

Source checkpoints for the five official seeds are **not** included in the code release zip (for size and redistribution hygiene).

| Seed | Checkpoint File |
|---:|---|
| 42 | `camelyon17_resnet18_source_s42_BEST.pt` |
| 123 | `camelyon17_resnet18_source_s123_BEST.pt` |
| 2024 | `camelyon17_resnet18_source_s2024_BEST.pt` |
| 7 | `camelyon17_resnet18_source_s7_BEST.pt` |
| 99 | `camelyon17_resnet18_source_s99_BEST.pt` |

**Architecture (locked):** ResNet-18, ImageNet init, 1-logit head, dropout 0.5, BCE-with-logits.  
**Protocol:** `configs/authoritative_resnet18_utta.yaml`, `configs/tau_by_seed.yaml`.

Model checkpoints are not included in this source release. The source model can be reproduced using the provided training configuration and scripts (`scripts/train_source.py`). For verification, official checkpoints correspond to the following SHA-256 checksums (documented in `reproducibility/ARTIFACT_MANIFEST.csv`):

| Seed | Checkpoint File | SHA-256 Checksum |
|---:|---|---|
| 42 | `camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| 123 | `camelyon17_resnet18_source_s123_BEST.pt` | `cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224` |
| 2024 | `camelyon17_resnet18_source_s2024_BEST.pt` | `d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261` |
| 7 | `camelyon17_resnet18_source_s7_BEST.pt` | `9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b` |
| 99 | `camelyon17_resnet18_source_s99_BEST.pt` | `0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540` |

When present, checkpoints should be placed under:
```text
checkpoints/camelyon17_resnet18_source_s{seed}_BEST.pt
```

Headline metrics do **not** require reloading weights; they are frozen in `results/metrics/full_s*.json`, `docs/HEADLINE_5SEED.md`, and `docs/PHASE1_SIGNOFF.md`. Weights are only needed to re-run TTA adaptation or Grad-CAM.

## Do not claim

- First uncertainty-aware medical TTA
- UTTA-Med beats confidence
- Drop seeds 7/99
- Patch-level CIs as primary
- Clinical validation of Grad-CAM

## Author & License

- **Author & Rights:** Avinash Reddy Cheerapareddy (<avinashreddych7@gmail.com>)
- **License:** MIT License (see [`LICENSE`](LICENSE))
