# Merge this pack into `C:\Users\avina\OneDrive\Desktop\UTTA-Med`

Extract **over** the existing folder (overwrite). Do **not** replace `checkpoints\*.pt`.

```powershell
cd C:\Users\avina\OneDrive\Desktop
Expand-Archive -Path UTTA-Med-pending-update.zip -DestinationPath . -Force
```

If Windows creates `UTTA-Med\UTTA-Med\...`, move the inner files up one level.

## What this pack adds

| Path | Why |
|---|---|
| `results/metrics/camelyon17_wsi_bootstrap_s42.json` | Full ~24 KB WSI file (was 1.1 KB stub) |
| `results/statistical/wsi_bootstrap/camelyon17_wsi_bootstrap_s42.json` | Same, canonical location |
| `results/metrics/camelyon17_resnet18_full_s123.json` | Rich Gate-D JSON (was 2.2 KB stub) |
| `results/metrics/camelyon17_resnet18_full_s7.json` | Rich JSON if you still had a stub |
| `results/predictions/s42_*.npz` | Official 85,054-patch dumps |
| `results/statistical/wsi_bootstrap/figures/reliability_*s42*.png` | Seed-42 reliability diagrams |
| `figures/gradcam_frozen/frozen_*.png` | Frozen-source-U Grad-CAM |
| `figures/gradcam/source_*.png` | Source TP/TN/FP/FN/high-U |
| `figures/gradcam_post_tta/compare_*.png` | Post-TTA CAM (UTTA column from invalid online-U run — see docs) |
| `scripts/rebuild_s42_wsi_from_npz.py` | How s42 WSI JSON was rebuilt |
| `requirements.txt` | Adds `umap-learn` |

Keep your five `checkpoints/*_BEST.pt` files. This zip does **not** include them.
