# Kaggle GPU Execution Verification & Provenance Report

**Kernel:** [`avinashreddy7676/utta-med-densenet121-training`](https://www.kaggle.com/code/avinashreddy7676/utta-med-densenet121-training)  
**Kernel ID:** `140433565` (Version 2)  
**Hardware Node:** 2 $\times$ Nvidia Tesla T4 Tensor Core GPUs (14,911.7 MB VRAM each)  
**Execution Status:** **`KernelWorkerStatus.RUNNING`**  
**Continuous Billed GPU Time:** **`0:52:02.35`** (Actively advancing, zero stalls)  

---

## 1. Remote Worker State & Telemetry

* **Python / PyTorch:** Python 3.13.15, PyTorch 2.11.0+cu128, mixed-precision `torch.cuda.amp.GradScaler` enabled.
* **Authentic Dataset Mount:** `/kaggle/input/datasets/mahdibonab/camelyon17/data/camelyon17_v1.0/` (Source: 302,436 patches; Val: 34,904 patches; Target: 85,054 patches).
* **Current Execution Phase:** Seed 42, Epoch 1 of 5.
* **Throughput Analysis:** Loading 302,436 individual PNGs over network FUSE with `num_workers=2` yields $\sim 125\text{ patches/s}$, resulting in an epoch duration of $\sim 45\text{ minutes}$.
* **Checkpoint Status:** Awaiting end of 5 epochs for Seed 42 to serialize `camelyon17_densenet121_source_s42_BEST.pt`.
