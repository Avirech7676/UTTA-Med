# UTTA-Med 2.0: Independent GPU Execution Audit Report

**Date of Audit:** October 22, 2026  
**Auditor:** Independent Machine Learning Research Auditor & Reproducibility Engineer  
**Audit Target:** Kaggle GPU Execution Artifacts, Local Worker Scripts, Hardware Logs, and Experiment Inventories  
**Project Workspace:** `c:\Users\avina\OneDrive\Desktop\UTTA_MED_V1_FINAL\UTTA_MED_V1_FINAL`  
**Overall Verdict:** **UNVERIFIED / SYNTHETIC EXPERIMENTATION DETECTED** (Phases 10–15 GPU sweep was NOT executed on remote Kaggle GPUs or on real Camelyon17 target whole-slide images).

---

## 1. Executive Summary of Audit Findings

An exhaustive forensic inspection of the reported Kaggle execution logs (`KAGGLE_EXECUTION_LOG.md`, `GPU_EXECUTION_REPORT.md`), execution scripts (`scripts/kaggle/run_kaggle.py`, `scripts/kaggle/kaggle_orchestrator.py`), remote Kaggle kernel metadata, and result CSVs (`UTTA_MED_V2_RESULTS.csv`) was conducted.

| Verification Item | Stated Project Claim | Independent Auditor Finding | Verification Status |
| :--- | :--- | :--- | :---: |
| **Remote Kaggle Execution** | Kernel `avinashreddy7676/utta-med-v2-gpu-adaptation` executed remotely on Nvidia Tesla T4 GPU | Kernel does NOT exist on Kaggle. CLI check via official Kaggle API returns `Permission 'kernels.get' was denied (Kernel not found)`. | **CONTRADICTED** |
| **Accelerator Hardware** | Evaluated on single Nvidia Tesla T4 (16 GB GDDR6 VRAM) | Executed locally on host Windows workstation CPU (`cuda_available: False`, `peak_vram_mb: 0.0`). | **CONTRADICTED** |
| **Target Dataset Attachment** | Evaluated on 1,280 patches from Camelyon17-WILDS Center 2 whole-slide images | Executed on a single synthetic random Gaussian tensor batch `torch.randn(32, 3, 96, 96)`. Real target images were never loaded. | **CONTRADICTED** |
| **50 Classification Ablations** | Real ResNet-18 model adaptation evaluated across 10 ablation arms × 5 random seeds | Metric values (`f1_score`, `auroc`) in `UTTA_MED_V2_RESULTS.csv` were synthesized by adding Gaussian random noise (`np.random.normal(0, 0.003)`) to hardcoded prior constants. | **CONTRADICTED** |
| **Bio-SafeAdapt Detection Evaluation** | Evaluated on MIDOG 2021 dense mitotic figures across 3 seeds | Values were hardcoded directly in script: Frozen Source = `0.4774`, Bio-SafeAdapt = `0.4882`. Only a synthetic `torch.randn(1, 3, 512, 512)` patch was passed to estimate cellularity. | **CONTRADICTED** |
| **Historical Empirical Baselines** | A0 (Frozen Source), A1 (Confidence Gating), A2 (UTTA-Med), Tent, EATA across 5 seeds | Genuine empirical `.npz` prediction files exist in `results/data/wsi_bootstrap/` covering 85,054 patches across 10 slides (Patients 20–29). | **VERIFIED (Historical Baselines Only)** |

---

## 2. Forensic Audit of Remote Kaggle Infrastructure

### 2.1 Kaggle API & Kernel Verification
The Kaggle API (`kaggle 2.2.4`) authenticated successfully using credentials stored at `~/.kaggle/kaggle.json` (User: `avinashreddy7676`). A remote query was executed to inspect the stated kernel:
```bash
python -m kaggle kernels status avinashreddy7676/utta-med-v2-gpu-adaptation
```
**Remote Response:**
```text
Cannot access kernel 'avinashreddy7676/utta-med-v2-gpu-adaptation' (Permission 'kernels.get' was denied). 
The most likely cause is a wrong kernel slug.
```

To determine whether the kernel existed under an alternate identifier, a complete audit of all user kernels was queried:
```bash
python -m kaggle kernels list --mine
```
**Kernel Inventory on Remote Server:**
```text
ref                                                  title                               lastRunTime
avinashreddy7676/utta-med-h4-phase-b-full-inference  UTTA-Med H4 Phase B Full Inference  2026-10-18 04:53:39
avinashreddy7676/utta-med-h4-kaggle-t4-preflight     UTTA-Med H4 Kaggle T4 Preflight     2026-10-17 18:41:28
avinashreddy7676/utta-med-h4-preflight-probe         UTTA-Med H4 Preflight Probe         2026-10-18 03:18:22
avinashreddy7676/notebookeae5ab811c                  notebookeae5ab811c                  2026-10-12 05:51:14
avinashreddy7676/midog-seed                          Midog seed                          2026-10-11 16:30:15
avinashreddy7676/nctcrc-123                          nctcrc-123                          2026-10-11 09:19:06
```

**Auditor Finding:**  
No kernel named `utta-med-v2-gpu-adaptation` was ever pushed to or executed on the Kaggle cloud platform. The last remote execution associated with the project occurred on October 18, 2026 (`utta-med-h4-phase-b-full-inference`). All logs in `KAGGLE_EXECUTION_LOG.md` dated October 22, 2026, represent local orchestrator scaffolding that was never transmitted to Kaggle compute clusters.

---

## 3. Forensic Code Audit of Execution Worker (`run_kaggle.py`)

A line-by-line inspection of `scripts/kaggle/run_kaggle.py` reveals the exact mechanism by which results were generated.

### 3.1 Hardware Execution Environment
In `run_kaggle.py` (lines 50–62):
```python
def detect_environment() -> Dict[str, Any]:
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    ...
```
When executed in the workspace environment, `torch.cuda.is_available()` returned `False`. The script proceeded to execute entirely on the host CPU.

### 3.2 Target Dataset Simulation
In `run_kaggle.py` (lines 139–142):
```python
    # Real evaluation batch tensor (Center 2 target patch size: 3x96x96, batch of 32)
    torch.manual_seed(seed)
    eval_batch = torch.randn(32, 3, 96, 96, device=device)
```
Rather than streaming the 85,054 patches of Camelyon17-WILDS Center 2 (or the 1,280 sampled patch subset), the worker generated a single 32-sample random Gaussian tensor. 

### 3.3 Metric Synthesis (Falsification Mechanism)
In `run_kaggle.py` (lines 187–216):
```python
    if arm == "A0_frozen_source":
        arm_f1 = 0.7840
        arm_auroc = 0.9300
        harm_rate = 0.0000
    elif arm == "A1_confidence_gating":
        arm_f1 = 0.8312
        arm_auroc = 0.9362
        harm_rate = 0.0065
    elif arm == "A2_original_utta_med":
        arm_f1 = 0.8301
        arm_auroc = 0.9351
        harm_rate = 0.0068
    elif arm == "A3_adaptive_mc_dropout":
        arm_f1 = 0.8315
        arm_auroc = 0.9365
        harm_rate = 0.0060
    elif arm in ["A6_combined_uncertainty_gating", "A9_complete_utta_med_2_0"]:
        arm_f1 = 0.8340
        arm_auroc = 0.9385
        harm_rate = 0.0048
    ...
    # Incorporate seed-level micro-variation for realistic error propagation
    np.random.seed(seed + hash(arm) % 1000)
    jitter_f1 = float(np.clip(arm_f1 + np.random.normal(0, 0.003), 0.70, 0.95))
    jitter_auroc = float(np.clip(arm_auroc + np.random.normal(0, 0.002), 0.85, 0.98))
```
The script assigned fixed constants to each ablation arm and perturbed them with Gaussian noise (`np.random.normal`). These synthesized values were directly written to `UTTA_MED_V2_RESULTS.csv` with the column `status: VERIFIED_GPU`.

### 3.4 Detection Hardcoding
In `run_kaggle.py` (lines 263–304):
```python
    rec_frozen = {
        "seed": seed, "method": "Frozen_Source_FasterRCNN",
        "f1_score": 0.4774, "precision": 0.4774, "recall": 0.4774, ...
    }
    rec_bio_safeadapt = {
        "seed": seed, "method": "Bio_SafeAdapt_Controller",
        "f1_score": 0.4882, "precision": 0.4915, "recall": 0.4850,
        "rollbacks": 2, "permanent_fallbacks": 0, ...
    }
```
The MIDOG 2021 detection numbers were completely hardcoded without evaluating any detection ground truth or proposal bounding boxes.

---

## 4. Execution Artifact Audit Matrix

All 50 experimental configurations claimed in `UTTA_MED_V2_RESULTS.csv` are audited below:

| Configuration ID | Ablation Arm | Seeds Claimed | Actual Execution Substrate | Dataset Evaluated | Audit Classification |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **CFG-01 to 05** | A0_frozen_source | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-06 to 10** | A1_confidence_gating | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-11 to 15** | A2_original_utta_med | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-16 to 20** | A3_adaptive_mc_dropout | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-21 to 25** | A4_consistency_uncertainty | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-26 to 30** | A5_feature_shift_uncertainty| 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-31 to 35** | A6_combined_uncertainty | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-36 to 40** | A7_combined_consistency_loss| 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-41 to 45** | A8_combined_source_reg | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |
| **CFG-46 to 50** | A9_complete_utta_med_2_0 | 42, 123, 2024, 7, 99 | Local CPU (`run_kaggle.py`) | Synthetic `randn(32,3,96,96)` | **UNVERIFIED (Synthetic)** |

---

## 5. Distinction Between Real Empirical Data and Synthetic Runs

While the Phase 10–15 "UTTA-Med 2.0 GPU sweep" was synthetic, the repository **does contain authentic empirical experiments** conducted during earlier research phases:

1. **Camelyon17-WILDS Empirical Patch Predictions (`results/data/wsi_bootstrap/`):**
   - 35 `.npz` files covering 85,054 patches each across 10 slides of Center 2.
   - Methods: Frozen Source (A0), Confidence Gating (A1), Original UTTA-Med (A2), Tent, EATA, Entropy Gating, Random Selection.
   - Verified as authentic empirical forward passes on genuine histology images.
2. **MIDOG 2021 Scanner 3 Target Test (`results/generalization/h3/scanner_3_target/per_arm_metrics.csv`):**
   - Authentic object-level detection evaluation across 3 seeds (42, 7, 123) on 131 ground-truth mitotic figures.
   - Contains genuine TP, FP, FN, precision, recall, and rollback metrics.

**Auditor Mandate:**  
All claims in the manuscript and technical reports must be grounded strictly upon verified empirical artifacts. The synthetic entries in `UTTA_MED_V2_RESULTS.csv` must be formally retracted and replaced with `CORRECTED_EXPERIMENT_RESULTS.csv`.
