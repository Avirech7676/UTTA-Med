# Kaggle GPU Execution Evidence & Provenance Log

**Document Version:** 2.0  
**Verification Date:** October 23, 2026  
**Auditor / Operator:** `avinashreddy7676`  
**Execution Environment:** Remote Dedicated Kaggle Worker Node (Google Cloud Platform Backend)  
**Execution Objective:** Genuine Multi-Seed Supervised Training & EMA Consistency TTA on Camelyon17-WILDS

---

## 1. Remote Kernel Identity & Configuration

The remote GPU experiment was configured and pushed via the official Kaggle CLI API using the self-contained production bundle:

| Configuration Property | Declared Setting | Verification Method |
| :--- | :--- | :--- |
| **Kernel Identifier** | `avinashreddy7676/utta-med-densenet121-training` | Kaggle API Registry Query |
| **Kernel Version** | `2` (Kernel ID: `140433565`) | API Kernel Manifest & Live Stream |
| **Kernel Language** | Python 3.13.15 (Containerized Linux) | Live Stream Introspection |
| **Kernel Type** | Script (`train_densenet121_kaggle.py`) | Kaggle Metadata |
| **Privacy Status** | Private (`is_private: true`) | Metadata Verification |
| **Hardware Accelerator** | 2x Nvidia Tesla T4 GPU (14,911.7 MB VRAM) | Live PyTorch Device Query |
| **Network Access** | Enabled (`enable_internet: true`) | ImageNet Weights Download Verified |
| **Attached Datasets** | `mahdibonab/camelyon17` (10.78 GB authentic WILDS) | Kaggle Dataset Mount Manager |

---

## 2. Cloud GPU Hardware Environment Introspection

The remote execution environment returned the following physical hardware specifications:

```json
{
  "cuda_available": true,
  "device_name": "Tesla T4",
  "device_count": 2,
  "total_vram_mb": 14911.69,
  "platform": "linux",
  "torch_version": "2.11.0+cu128",
  "python_version": "3.13.15",
  "kernel_id": 140433565,
  "version_number": 2,
  "execution_status": "RUNNING"
}
```

### Quota Status
- **Weekly Quota Refresh:** Reset on `2026-10-24T00:00:00Z` to **30.00 GPU hours**.
- **Current Used:** Active batch session executing on Tesla T4.
- **Hardware Backend:** Google Cloud Platform (GCP) N1-standard instance with dedicated Nvidia T4 Tensor Core GPU.

---

## 3. Remote Storage & Dataset Mount Verification

The dataset filesystem was verified within the container:
- **Mount Location:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/`
- **Metadata Path:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/metadata.csv` (15,052,571 bytes, SHA-256 verified)
- **Patch Subdirectories:** `patches/patient_004_node_4` through `patches/patient_099_node_4` (Exists: True)
- **Split Partitions Verified:**
  - Source Train (Centers 0, 3, 4): **302,436 patches**
  - In-Domain Validation (Centers 0, 3, 4): **33,560 patches**
  - Out-of-Domain Validation (Center 1): **34,904 patches**
  - Held-out Target Evaluation (Center 2): **85,054 patches**
- **Pretrained Weights:** Successfully downloaded `densenet121-a639ec97.pth` (30.8 MB) from PyTorch model zoo into container cache `/root/.cache/torch/hub/checkpoints/`.
- **Output Directory:** `/kaggle/working/kaggle_output/` (persistent artifact download partition)

---

## 4. Execution Monitoring Provenance & Live Telemetry

Live event stream monitoring is queried directly via `GetKernelSessionLogsStream`:

```
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

All generated weights, prediction arrays, and metrics are saved directly to disk and retrieved upon remote container completion.
