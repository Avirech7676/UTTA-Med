# Optional experiments (NOT Table 1)

Primary science is Camelyon17 ResNet-18 five-seed under `results/metrics/full_s*.json`.

Put every optional GPU track JSON **here**, under the matching subfolder.

| Track | Subfolder | Canonical JSON name | Table 1? |
|-------|-----------|---------------------|----------|
| X1 ResNet-50 s42 source | `resnet50/` | `camelyon17_resnet50_source_s42.json` | **No** |
| X1 ResNet-50 TTA | `resnet50/` | **Do not store** (NaNs — rejected) | **No** |
| X2 PCam source + TTA | `pcam/` | `pcam_tta_repair_s42.json` | **No** (supplement) |
| X3 PathMNIST + C | `pathmnist/` | `pathmnist_c_tta_s42.json` | **No** (supplement) |
| X4 MIDOG LODO | `midog/` | `midog_patches_lodo_s42.json` | **No** (failed proxy) |
| Camelyon17-C | `camelyon17_c/` | `camelyon17_c_shuffled_s42.json` | **No** (supplement) |
| NCT-CRC P1 | `nctcrc/` | `nctcrc_p1_s42.json` | **No** (single seed) |

## On your PC

```
<project-root>\results\optional\
  resnet50\
  pcam\
  pathmnist\
  midog\
  camelyon17_c\
  nctcrc\
  OPTIONAL_SIGNOFF.md   ← status table
  README.md             ← this file
```

Copy from Kaggle `/kaggle/working/*.json` into the matching subfolder.

**Never** mix these numbers into `results/metrics/full_s*.json` or Table 1.
