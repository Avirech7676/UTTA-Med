# UTTA-Med reproducibility report

Do **not** invent a GitHub URL. This tree has no public remote until you publish one.

## Official architecture (verified against seed-42 BEST.pt)

- ResNet-18, **1 logit** (`classifier.3.weight` shape `[1, 512]`), BCE, **dropout 0.5**
- Source: **Adam** `1e-4` / wd `1e-4`, early-stop hospital 1
- TTA: **Adam** `1e-5`, `bn_freeze_stats`, k=1, N_MC=20
- τ = 70th percentile of hospital-1 U (`configs/tau_by_seed.yaml`) — **not 0.05**
- Seeds `{42, 123, 2024, 7, 99}`

## Checkpoint inventory (this package)

All five source checkpoints are present in `checkpoints/`:

| Seed | BEST.pt in *this* tree | Checkpoint Path | JSON Metric Record |
|---:|---|---|---|
| 42 | **yes** | `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | `results/metrics/camelyon17_resnet18_full_s42.json` |
| 123 | **yes** | `checkpoints/camelyon17_resnet18_source_s123_BEST.pt` | `results/metrics/camelyon17_resnet18_full_s123.json` |
| 2024 | **yes** | `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt` | `results/metrics/camelyon17_resnet18_full_s2024.json` |
| 7 | **yes** | `checkpoints/camelyon17_resnet18_source_s7_BEST.pt` | `results/metrics/camelyon17_resnet18_full_s7.json` |
| 99 | **yes** | `checkpoints/camelyon17_resnet18_source_s99_BEST.pt` | `results/metrics/camelyon17_resnet18_full_s99.json` |

All five checkpoints are cryptographically locked in `reproducibility/ARTIFACT_MANIFEST.csv`. Large external prediction artifacts remain external and are documented in the manifest.

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Expected: all tests pass; `test_wsi_no_overlap` / `test_hospital_separation` skip if WILDS PNG data is absent. YAML hospital isolation always runs.

## What is **not** claimed

- A public `github.com/anonymous/...` URL
- That all 35 raw prediction dumps are bundled inside a code-only zip (documented in ARTIFACT_MANIFEST.csv)
- Patch-level CIs as primary inference
- UTTA-Med beats confidence
