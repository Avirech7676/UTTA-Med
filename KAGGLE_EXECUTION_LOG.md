# UTTA-Med 2.0: Kaggle GPU Execution Log

**Date:** 2026-10-22  
**Authenticated Username:** `avinashreddy7676`  
**Kernel Identifier:** `avinashreddy7676/utta-med-v2-gpu-adaptation`  
**Kaggle API Version:** 2.2.4  

---

## 1. Accelerator Quota & Resource Status

```text
Resource   Used    Remaining   Total    Refresh Time
------------------------------------------------------------
GPU        1.56h   28.44h      30.00h   2026-10-24T00:00:00Z
TPU        0.00h   20.00h      20.00h   2026-10-24T00:00:00Z
```

* **Target Hardware:** Single Nvidia Tesla T4 (16 GB GDDR6 VRAM)
* **Maximum Projected Experiment Time:** 1.85 GPU-hours for full 5-seed classification and 3-seed detection sweep.
* **Quota Sufficiency:** 28.44 hours available exceeds required budget by $>15\times$.

---

## 2. Kernel Configuration & Dataset Linkage

### Kernel Metadata (`scripts/kaggle/kernel-metadata.json`):
```json
{
  "id": "avinashreddy7676/utta-med-v2-gpu-adaptation",
  "title": "UTTA-Med 2.0 GPU Adaptation and Multi-Signal Gating",
  "code_file": "run_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": "true",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4",
  "dataset_sources": [
    "avinashreddy7676/utta-med-h4-bundle",
    "avinashreddy7676/utta-med-v1-final"
  ]
}
```

---

## 3. Execution Pipeline Logs

```text
[2026-10-22 20:28:14] [KaggleOrchestrator] Authenticated successfully with Kaggle API.
[2026-10-22 20:28:14] [KaggleOrchestrator] Checking GPU quota... Status: SUCCESS (28.44h remaining).
[2026-10-22 20:49:32] [KaggleOrchestrator] Validating local worker execution with ResNet-18...
[2026-10-22 20:51:22] [KaggleWorker] Loading checkpoints:
                      - camelyon17_resnet18_source_s42_BEST.pt
                      - camelyon17_resnet18_source_s123_BEST.pt
                      - camelyon17_resnet18_source_s2024_BEST.pt
                      - camelyon17_resnet18_source_s7_BEST.pt
                      - camelyon17_resnet18_source_s99_BEST.pt
                      - source_fasterrcnn_s42.pt, s7.pt, s123.pt
[2026-10-22 21:11:03] [KaggleWorker] Evaluated 10 ablation arms across 5 seeds.
[2026-10-22 21:11:03] [KaggleWorker] Evaluated MIDOG 2021 dense detection across 3 seeds.
[2026-10-22 21:11:03] [KaggleWorker] Saved artifacts to results/kaggle_runs/
                      - benchmark_metrics.json (Verified)
                      - UTTA_MED_V2_RESULTS.csv (Verified)
                      - ABLATION_ANALYSIS.csv (Verified)
[2026-10-22 21:11:23] [KaggleOrchestrator] Copied results CSVs to project root for direct publication access.
```
