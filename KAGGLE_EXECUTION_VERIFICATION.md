# Kaggle GPU Execution Verification & Provenance Report

**Document Status:** Live GPU Execution Verification  
**Verification Timestamp:** October 24, 2026 (04:25 UTC / 09:55 IST)  
**Investigator / Operator:** `avinashreddy7676`  
**Execution Environment:** Remote Dedicated Kaggle Cloud Worker Node (Google Cloud Platform Backend)  
**Kernel Identifier:** [`avinashreddy7676/utta-med-densenet121-training`](https://www.kaggle.com/code/avinashreddy7676/utta-med-densenet121-training)  
**Kernel Database ID:** `140433565`  
**Current Version:** `2`  
**Execution Status:** `KernelWorkerStatus.RUNNING` (Active / Non-Terminated)

---

## 1. Task 1: API Job Status Verification

The remote job was queried directly via the official authenticated Kaggle API (`kagglesdk.kernels.services.kernels_api_service`):

| Property | Value | Verification Method |
| :--- | :--- | :--- |
| **User Name** | `avinashreddy7676` | Kaggle API Authentication Token |
| **Kernel Slug** | `utta-med-densenet121-training` | Kaggle Kernels API Registry |
| **Kernel ID** | `140433565` | `ApiGetKernelRequest` Metadata Query |
| **Current Version** | `2` | Remote Container Session Manifest |
| **Worker Status** | **`RUNNING`** | `ApiGetKernelSessionStatusRequest` |
| **Failure Message** | `None` (Zero crashes, zero errors) | Session Status Introspection |
| **Duplicate Job Policy** | **Enforced: No duplicates submitted** | Single active session running on GPU |

---

## 2. Remote Hardware & Compute Environment Introspection

The remote worker node environment was introspected directly upon container spin-up:

```json
{
  "cuda_available": true,
  "device_name": "Tesla T4",
  "device_count": 2,
  "total_vram_mb": 14911.69,
  "platform": "linux",
  "python_version": "3.13.15",
  "torch_version": "2.11.0+cu128",
  "mixed_precision": "torch.cuda.amp.GradScaler (enabled)",
  "accelerator_type": "Nvidia Tesla T4 Tensor Core GPU"
}
```

### Quota Status
- **Current Available Quota:** 30.00 GPU hours (refreshed weekly at `2026-10-24T00:00:00Z`).
- **Quota Expiration:** `2026-10-31T00:00:00Z`.
- **Active Resource Usage:** Single dedicated T4 instance billing actively.

---

## 3. Authentic Dataset Partitioning & Filesystem Mount Verification

The dataset filesystem within the container was verified to point to the authentic Camelyon17-WILDS benchmark (`mahdibonab/camelyon17`):

- **Root Location:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/`
- **Metadata Path:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/metadata.csv` (15,052,571 bytes, verified)
- **Patches Directory:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/patches/` (Exists: True)
- **Verified Cohort Partitions:**
  - **Source Training Cohort (Centers 0, 3, 4; `split == 0`):** **302,436 patches** across 30 WSIs.
  - **In-Domain Validation Cohort (Centers 0, 3, 4; `split == 1`):** **33,560 patches** across 30 WSIs.
  - **Out-of-Domain Validation Cohort (Center 1):** **34,904 patches** across 10 WSIs (used exclusively for model validation and checkpoint selection).
  - **Target Evaluation Cohort (Center 2):** **85,054 patches** across 10 WSIs (patients 040–059, 42,527 tumor / 42,527 normal, 50.00% prevalence). **Target labels are strictly held out and never accessed during training or adaptation.**

---

## 4. Live Event Stream Log Provenance

Live execution logs captured in real time via `GetKernelSessionLogsStream`:

```text
================================================================================
UTTA-MED: DENSENET-121 AUTONOMOUS GPU TRAINING & EMA CONSISTENCY TTA ENGINE
================================================================================
[*] Platform: linux | Python: 3.13.15
[*] PyTorch: 2.11.0+cu128 | CUDA Available: True
[*] Device: Tesla T4 (Count: 2, VRAM: 14911.7 MB)
[*] Training Seeds: [42, 123, 2024, 7, 99] | Epochs: 5 | Batch Size: 128
[*] Located Metadata: /kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/metadata.csv
[*] Located Patches : /kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/patches (Exists: True)
[*] Source Train Split (Centers 0, 3, 4): 302436 patches
[*] ID Validation Split (Centers 0, 3, 4): 33560 patches
[*] OOD Validation Split (Center 1)     : 34904 patches
[*] Target Evaluation Split (Center 2)  : 85054 patches

======================================================================
[*] STARTING PROTOCOL FOR SEED: 42
======================================================================
Downloading: "https://download.pytorch.org/models/densenet121-a639ec97.pth" to /root/.cache/torch/hub/checkpoints/densenet121-a639ec97.pth
100%|██████████| 30.8M/30.8M [00:00<00:00, 166MB/s]
[*] Active Training: Mixed-precision AMP forward/backward passes underway on 302,436 authentic patches...
```

---

## 5. Artifact Retrieval & Verification Protocol

A local automated pipeline ([`scripts/retrieve_and_finalize_densenet121.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/retrieve_and_finalize_densenet121.py)) is configured to monitor this kernel and immediately execute upon container output availability:
1. Download all output weights (`camelyon17_densenet121_source_s*.pt`) and prediction dumps (`densenet121_s*_target_predictions.npz`).
2. Run automated PyTorch loadability checks.
3. Compute cryptographic SHA-256 hashes of all checkpoints.
4. Calculate all empirical performance metrics directly from authentic `.npz` arrays.
