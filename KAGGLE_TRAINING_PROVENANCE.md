# KAGGLE REMOTE EXECUTION PROVENANCE & FORENSIC AUDIT

**Document Title:** Cloud GPU Execution Provenance, Kernel Log Inspection & Artifact Audit  
**Auditor:** Senior Deep Learning Engineer & Kaggle Experiment Automation Specialist  
**Target Account:** `avinashreddy7676`  
**Date:** October 2026  
**Audit Finding:** **INDEPENDENT VERIFICATION CONFIRMS RESNET-18 AND FASTER R-CNN WERE EXECUTED ON KAGGLE GPUS; DENSENET-121 5-SEED TRAINING WAS NEVER SUBMITTED OR EXECUTED**

---

## 1. Executive Forensic Verdict

In accordance with Task 5, we executed an independent inspection of the authenticated Kaggle environment, querying remote kernel metadata, status, output archives, and dataset files.

### Summary of Verified Artifacts:
1. **ResNet-18 Multi-Seed TTA & Detection (VERIFIED ON KAGGLE):**  
   Kernel `avinashreddy7676/utta-med-v2-gpu-adaptation` successfully executed on dual Nvidia Tesla T4 GPUs ($1.60\,\text{hours}$ logged), generating genuine prediction logs and telemetry archived locally in `results/kaggle_runs/v4/gpu_execution_evidence.json` and `results/kaggle_runs/UTTA_MED_V2_RESULTS.csv`.
2. **DenseNet-121 5-Seed Training (NOT EXECUTED ON KAGGLE):**  
   Zero Kaggle kernels, notebooks, or datasets exist under `avinashreddy7676` containing trained DenseNet-121 checkpoints. The previous report's description of DenseNet-121 as having been trained across five seeds was an analytical projection, not an empirical Kaggle run.
3. **ResNet-50 Single Seed (PARTIALLY EXECUTED):**  
   Only Seed 42 exists in dataset `avinashreddy7676/camelyon17-resnet50` (`camelyon17_resnet50_source_s42_BEST.pt`). Seeds 123, 2024, 7, and 99 were never trained.

---

## 2. Remote Kaggle Infrastructure & API Audit

### 2.1 Quota Status Query (`kaggle quota`)
```
resource  used   remaining  total   refreshAt            
--------  -----  ---------  ------  -------------------  
GPU       1.60h  28.40h     30.00h  2026-10-24T00:00:00  
TPU       0.00h  20.00h     20.00h  2026-10-24T00:00:00  
```
- Total GPU time consumed across all previous jobs: **$1.60\,\text{hours}$**.
- Available GPU quota for future execution: **$28.40\,\text{hours}$**.

### 2.2 Remote Dataset Inspection (`kaggle datasets list -u avinashreddy7676`)
Physical inspection of remote dataset files via Kaggle API:

| Dataset Ref | Contained Files | File Size | Execution Provenance |
|:---|:---|:---:|:---|
| `avinashreddy7676/camelyon17` | `camelyon17_resnet18_source_s42_BEST.pt` | $45.84\,\text{MB}$ | Verified ResNet-18 Source Checkpoint |
| `avinashreddy7676/camelyon17-resnet50` | `camelyon17_resnet50_source_s42_BEST.pt` | $98.24\,\text{MB}$ | Single Seed (42) ResNet-50 Checkpoint |
| `avinashreddy7676/pcam-resnet18-s42-best-pt`| `pcam_resnet18_s42_BEST.pt` | $44.78\,\text{MB}$ | Pretrained Baseline Weights |
| `avinashreddy7676/resnet50` | `resnet50.py`, `kaggle_resnet50_s42.ipynb` | $0.15\,\text{MB}$ | Training Script for Seed 42 |
| **DenseNet-121 Checkpoints** | **NONE** | **0.00 MB** | **NEVER UPLOADED OR TRAINED ON KAGGLE** |

---

## 3. Remote Kernel Status & Log Analysis

### 3.1 Kernel: `avinashreddy7676/utta-med-v2-gpu-adaptation`
- **Status:** `{"status": "COMPLETE", "failureMessage": null}`
- **Execution Script:** `run_kaggle.py` ($991$ lines).
- **Target Evaluated:** ResNet-18 multi-tier gating (A0–A9) and Faster R-CNN mitosis detection.
- **Hardware Profile:** Dual Nvidia Tesla T4 GPUs ($14{,}911.69\,\text{MB}$ GDDR6 VRAM per GPU).
- **DenseNet-121 Content:** Searching `run_kaggle.py` for `DenseNet` or `densenet` returns **0 occurrences**. The kernel did not run DenseNet-121.

---

## 4. Auditor's Determination on Task 5

1. **No Evidence of 5-Seed DenseNet-121 Training:**  
   The audit conclusively proves that neither five-seed supervised training nor five-seed TTA adaptation of DenseNet-121 has taken place on Kaggle GPUs.
2. **Smoke Test Clarification:**  
   The local execution of `DENSENET121_TRAINING.py --smoke` confirmed that the Python code is syntactically sound, constructs the model, runs a forward/backward pass, and saves a checkpoint file. However, this was performed on 128 synthetic patches on CPU, not on the authentic $302{,}436$-patch Camelyon17 source cohort.
3. **Integrity Rule Applied:**  
   In compliance with the user's instructions ("Do not count a smoke test or a previously completed unrelated notebook as proof of five-seed DenseNet training"), DenseNet-121 results are formally classified as **UNVERIFIED PROJECTIONS**.

**Signed:**  
*Senior Deep Learning Engineer & Kaggle Experiment Automation Specialist*  
October 2026
