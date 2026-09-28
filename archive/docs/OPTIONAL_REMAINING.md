# Remaining optional experiments

These were listed as skip. They are now **implemented** and run on **one**
frozen seed-42 ResNet-18. Do **not** retrain the 5-seed headline.

## What is in the Kaggle job (required remaining GPU)

Uses `camelyon17_resnet18_source_s42_BEST.pt`.

| Block | Time on T4 | Output |
|---|---|---|
| Temperature scaling (fit hospital 1, apply hospital 2) | ~8 min | ECE/Brier before vs after T |
| SAR | ~4 min | AUROC/F1/ECE/harm/coverage |
| EATA-C | ~4 min | same |
| DLTTA | ~3 min | same (coverage=1) |
| Camelyon17-C (5 corruptions × severity 1,3, 8k patches) | ~15 min | source AUROC under C |

Total ~35–50 min.

## Differentiation (write this in the paper)

| Method | Sample filter | Extra |
|---|---|---|
| Tent | none | entropy min |
| EATA | entropy < e_margin | — |
| **EATA-C** | entropy + cosine diversity | no SAM, no Fisher |
| EATA-F | EATA-C | Fisher vs source |
| **SAR** | entropy | SAM + EMA reset |
| DLTTA | none | dynamic LR |
| UTTA-Med | MC-Dropout U < τ | — |

Never call SAR “EATA-C”. Never call UTTA “just EATA”.

## Stretch (do not block the paper)

1. **ResNet-50 seed 42** — 20-epoch source train, then UTTA once. Capacity check B2/B9.
2. **PathMNIST-C** — `pip install medmnist`; different 9-class task. Supporting only.
3. **EATA-F** — estimate Fisher on a source-train subset, then adapt.

## Permanently out of scope

Camelyon16, MIDOG++, ViT, CLIP, LLM, segmentation, 10 seeds, PathMNIST as the debug pipeline.

## Protocol still frozen

- BN-affine only, `bn_freeze_stats` for UTTA (SAR/EATA use their paper BN train mode unless noted)
- lr = 1e-5, k = 1
- Target labels never used for T, τ, or adaptation
- Negative results remain valid
