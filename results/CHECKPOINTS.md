# Source checkpoints

Official files are named `camelyon17_resnet18_source_s{SEED}_BEST.pt`.

All five source checkpoints are present in `checkpoints/`:
- Seed 42: `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`
- Seed 123: `checkpoints/camelyon17_resnet18_source_s123_BEST.pt`
- Seed 2024: `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt`
- Seed 7: `checkpoints/camelyon17_resnet18_source_s7_BEST.pt`
- Seed 99: `checkpoints/camelyon17_resnet18_source_s99_BEST.pt`

| Seed | In this package? | Checkpoint File | Status |
|---:|---|---|---|
| 42 | **yes** | `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | Verified (SHA256 locked) |
| 123 | **yes** | `checkpoints/camelyon17_resnet18_source_s123_BEST.pt` | Verified (SHA256 locked) |
| 2024 | **yes** | `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt` | Verified (SHA256 locked) |
| 7 | **yes** | `checkpoints/camelyon17_resnet18_source_s7_BEST.pt` | Verified (SHA256 locked) |
| 99 | **yes** | `checkpoints/camelyon17_resnet18_source_s99_BEST.pt` | Verified (SHA256 locked) |

Each file is a dict with keys `model`, `epoch`, `val_auroc`, `seed`.
`model['classifier.3.weight']` shape **[1, 512]** (1-logit).

Large external prediction artifacts remain external and are documented in `reproducibility/ARTIFACT_MANIFEST.csv`.
