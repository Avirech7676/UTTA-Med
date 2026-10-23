# Stage C2.5: Assessment Integrity Audit

**Auditor:** Independent Scientific Auditor & Medical Imaging Evaluation Specialist  
**Status:** Complete  
**Date:** October 2026  
**Dataset:** Camelyon17-WILDS (Hospital 1 / Center 1)  
**Scope:** Stage C2 Calibration and Evaluation Protocol Verification  

---

## 1. Executive Summary

This forensic audit investigates the integrity of the data partitions, sampling mechanisms, and evaluation denominators employed in Stage C2. 

The audit establishes three key findings:
1. **The 320-Patch Denominator is a Sparse Subsample, Not a Full Evaluation:** The 23,987 patches across the five assessment WSIs were not comprehensively evaluated; exactly 64 patches per slide (320 patches total, a 1.33% sampling fraction) were extracted via uniform stream striding as an intentional compute cap for CPU execution.
2. **Center-0 Slide 6 Provenance Misattribution:** Slide 6 is not an out-of-distribution validation control; it is derived from Center 0, which is one of the source training hospitals in Camelyon17-WILDS.
3. **Partition Disjointness is Preserved:** Calibration WSIs (`[10, 12, 14, 16, 18]`) and assessment WSIs (`[11, 13, 15, 17, 19]`) are strictly disjoint with zero slide-level or patch-level overlap.

---

## 2. Task A — Assessment Denominator Forensic

### 2.1 Why Stage C2 Defines 23,987 Patches but Evaluated 320
The Hospital 1 validation partition (`val` split, Center 1) comprises 34,904 patches distributed across 10 whole-slide images:
- Calibration partition (5 WSIs): 10,917 patches.
- Assessment partition (5 WSIs): 23,987 patches.

In `scripts/v2/calibrate_stage_c2.py`, line 87 defines:
```python
def extract_slide_patches(subset, slide_id, max_patches=64):
    ...
```
And line 171 executes:
```python
patches_per_slide = 64
```

#### Nature of the Limit:
- **Root Cause:** **Intentional compute sampling design / smoke-test cap**.
- **Engineering Justification:** On host CPU architecture, one batch (32 patches) requires approximately $8.9\text{ seconds}$ for forward inference, loss evaluation, backward propagation, optimizer stepping, and rolling-window collapse monitoring. Evaluating all 23,987 assessment patches across five regimens would require:
  $$\frac{23,987}{32} \approx 750\text{ batches} \times 8.9\text{ s} \times 5\text{ regimens} \approx 33,375\text{ seconds} \approx 9.27\text{ CPU hours}.$$
  To ensure deterministic local CPU execution without running asynchronous multi-hour batch jobs, the script sampled 64 patches per WSI.
- **Scientific Audit Finding:** While computationally understandable, **it is scientifically unacceptable to report or imply that 320 patches constitutes a complete five-WSI evaluation**. It must be explicitly reported as a **sparse, stride-sampled evaluation ($1.33\%$ sampling ratio)**.

### 2.2 Exact Evaluated Patch Breakdown & Class Balance

Each slide was sampled using uniform spatial stream striding (`step = n / max_patches`):

| WSI ID | Split Role | True WSI Patch Count ($N_{\mathrm{true}}$) | Evaluated Patch Count ($N_{\mathrm{eval}}$) | Sampling Ratio | Tumor Patches ($y=1$) | Normal Patches ($y=0$) | Evaluated Tumor Prevalence |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Slide 11** | Assessment | 5,618 | 64 | 1.14% | 38 | 26 | 59.38% |
| **Slide 13** | Assessment | 4,494 | 64 | 1.42% | 53 | 11 | 82.81% |
| **Slide 15** | Assessment | 3,842 | 64 | 1.67% | 0 | 64 | 0.00% |
| **Slide 17** | Assessment | 5,093 | 64 | 1.26% | 15 | 49 | 23.44% |
| **Slide 19** | Assessment | 4,940 | 64 | 1.30% | 4 | 60 | 6.25% |
| **Total Assessment** | Held-Out | **23,987** | **320** | **1.33%** | **110** | **210** | **34.38%** |
| **Slide 6** | Control | 7,680 | 64 | 0.83% | 0 | 64 | 0.00% |

---

## 3. Task D — Calibration Integrity & Center-0 Slide 6 Provenance

### 3.1 Partition Disjointness
- **Calibration WSIs:** `[10, 12, 14, 16, 18]` (Center 1)
- **Assessment WSIs:** `[11, 13, 15, 17, 19]` (Center 1)
- **Hospital 2 Target WSIs:** `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]` (Center 2)

Audit confirms that zero WSI overlap exists between calibration and assessment. Thresholds were selected exclusively from calibration slide distributions. Target Hospital 2 labels and patches were strictly unaccessed.

### 3.2 Center-0 Slide 6 Provenance Audit
In Stage C2 documentation, Slide 6 was referred to as an "independent negative validation control." Forensic indexing of Camelyon17-WILDS metadata reveals:
```text
Split train  : centers=[0, 3, 4], slides=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30-49]
Split id_val : centers=[0, 3, 4], slides=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 30-49]
Split val    : centers=[1],       slides=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Split test   : centers=[2],       slides=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
```

#### Scientific Correction:
1. **Center 0 is a Source Training Hospital:** Center 0 is one of the three training institutions (Centers 0, 3, 4).
2. **Slide 6 is In-Distribution Tissue:** Slide 6 is present in the `train` split and `id_val` split. It is **not** an out-of-distribution institution.
3. **True Out-of-Distribution Negative Specimen:** The only legitimate out-of-distribution 0%-tumor specimen in the development data is **Slide 15** from Hospital 1 (`val` split).
4. **Correction Enforced:** Slide 6 must be designated an **in-distribution training-center negative control**, not an independent clinical validation control.
