# Checkpoint Inventory & Status

All 5 official source checkpoints are present locally in this directory:
- camelyon17_resnet18_source_s42_BEST.pt (43.72 MB)
- camelyon17_resnet18_source_s123_BEST.pt (43.72 MB)
- camelyon17_resnet18_source_s2024_BEST.pt (43.72 MB)
- camelyon17_resnet18_source_s7_BEST.pt (43.72 MB)
- camelyon17_resnet18_source_s99_BEST.pt (43.72 MB)

All checkpoints are 1-logit BCE models with dropout=0.5 and Adam optimizer.
Sha256 checksums are documented in reproducibility/ARTIFACT_MANIFEST.csv.
There is no ResNet-17. Official backbone is ResNet-18 on Camelyon17-WILDS.