# Stage E0: Metric Reconciliation and Forensic Source Audit

**Auditor:** Independent Scientific Auditor, Senior PyTorch Engineer, and Medical Imaging Reproducibility Researcher  
**Date:** October 2026  
**Status:** Audit Completed & Signed Off  
**Scope:** Hospital 1 Frozen-Source Metric Traceability & Forensic Discrepancy Investigation  

---

## 1. Executive Summary & Forensic Determination

During the Stage D2 evaluation, a documented discrepancy was identified between:
- The True Frozen Source Brier score reported in **Stage C2.5**: **`0.082867`**
- The Frozen Source Brier score presented in the **Stage D2 summary table** (`STAGE_D2_NATURAL_POLICY_RESULTS.md`): **`0.0573`**

### Forensic Finding
1. **The Machine-Readable Code and Data are 100% Bit-Exact Identical:**
   Inspection of the canonical execution artifact `results/v2/stage_d2/stage_d2_benchmark_results.json` reveals:
   ```json
   "Frozen Source": {
     "metrics": {
       "f1": 0.8632478632478633,
       "brier": 0.08286656948071035,
       "ece": 0.0786241677836774,
       "auroc": 0.9735930735930736,
       "auprc": 0.9542381802683046
     }
   }
   ```
   The underlying PyTorch execution in Stage D2 produced **`brier = 0.08286657`**—identical to the **`0.082867`** audited and locked in Stage C2.5.
2. **Root Cause of the Discrepancy:**
   In `scripts/v2/evaluate_stage_d2.py`, the terminal logger output:
   ```python
   print(f" -> AUROC: {m['auroc']:.4f} | AUPRC: {m['auprc']:.4f} | F1: {m['f1']:.4f} | ECE: {m['ece']:.4f} | Patch Cov: {t['patch_coverage']:.1%} ...")
   ```
   The standard-output logger printed AUROC, AUPRC, F1, and ECE, but omitted Brier score from the console output line. During real-time documentation authoring while the benchmark executed, the markdown summary table was populated with `0.0573`—a value transcribed in error from historical coverage-risk summary tables (`results/tables/coverage_risk_summary.csv` and `results/five_seed_final_stats.json` where `0.0573` appears as an error rate and delta metric).
3. **Conclusion:**
   There was **NO changed sample**, **NO changed checkpoint**, **NO changed predictions**, **NO changed labels**, and **NO calculation bug**. The discrepancy was entirely a **documentation transcription error** in the human-facing markdown table. The underlying machine-readable outputs are authoritative and consistent.

---

## 2. Forensic Traceability Matrix

| Verification Dimension | Stage C2 (Initial) | Stage C2.5 (Audited Baseline) | Stage D2 (Machine-Readable JSON) | Stage D2 (Markdown Summary Table) | Stage E0 (Canonical Recomputation) | Match Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Source Checkpoint Path** | `camelyon17_resnet18_source_s42_BEST.pt` | `camelyon17_resnet18_source_s42_BEST.pt` | `camelyon17_resnet18_source_s42_BEST.pt` | `camelyon17_resnet18_source_s42_BEST.pt` | `camelyon17_resnet18_source_s42_BEST.pt` | **IDENTICAL** |
| **Checkpoint SHA-256 Hash** | `f0b85ec1a897...` | `f0b85ec1a897...` | `f0b85ec1a897...` | `f0b85ec1a897...` | `f0b85ec1a897...` | **IDENTICAL** |
| **Evaluated WSIs** | `[11, 13, 15, 17, 19]` | `[11, 13, 15, 17, 19]` | `[11, 13, 15, 17, 19]` | `[11, 13, 15, 17, 19]` | `[11, 13, 15, 17, 19]` | **IDENTICAL** |
| **Patch Denominator** | 320 patches | 320 patches | 320 patches | 320 patches | 320 patches | **IDENTICAL** |
| **Sampling Logic** | Exploratory stride | Preregistered uniform stride | Preregistered uniform stride | Preregistered uniform stride | Preregistered uniform stride | **IDENTICAL** |
| **AUROC** | 0.974026 | 0.973593 | 0.973593 | 0.9736 | 0.973593 | **IDENTICAL** |
| **AUPRC** | 0.954830 | 0.954238 | 0.954238 | 0.9542 | 0.954238 | **IDENTICAL** |
| **F1 Score** | 0.863248 | 0.863248 | 0.863248 | 0.8632 | 0.863248 | **IDENTICAL** |
| **ECE** | 0.079454 | 0.078624 | 0.078624 | 0.0786 | 0.078624 | **IDENTICAL** |
| **Brier Score** | 0.082422 | **0.082867** | **0.082867** | 0.0573 *(Typo)* | **0.082867** | **VERIFIED** |

---

## 3. Detailed Audit Components

### A. Source Checkpoint Integrity
- **File:** `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`
- **Algorithm:** SHA-256
- **Hash:** `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`
- **State Dict Structure:** Verified matching `models.resnet18` backbone with 2-layer classifier (`nn.Linear(512, 512)`, `nn.ReLU()`, `nn.Dropout(0.5)`, `nn.Linear(512, 1)`).

### B. Patch Sampling Logic and Ordering
- **Partition:** Hospital 1 (Center 1) validation set, restricted to Assessment WSIs `[11, 13, 15, 17, 19]`.
- **Ordering:** Sorted ascending by `patch_id` per WSI.
- **Stride:** Uniform spatial stream stride:
  $$\text{step} = \max(1, \lfloor N_{\text{slide}} / 64 \rfloor)$$
  Indices sampled: $[0, \text{step}, 2\cdot\text{step}, \dots, 63\cdot\text{step}]$.
- **Consistency:** Exact 64 patches per WSI (total 320 patches). Across Stage C2.5, Stage D2, and Stage E0, the exact same patch identifiers and ordering were evaluated.

### C. Data Transformations
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```
Applied identically without random augmentations during frozen-source inference.

### D. Mathematical Implementation of Brier Score
All pipelines compute Brier score as the mean squared error between model-predicted probability $p_i \in [0, 1]$ and true binary label $y_i \in \{0, 1\}$:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
Both `np.mean((p - y) ** 2)` and `sklearn.metrics.brier_score_loss(y, p)` yield exactly **`0.08286656948071035`** on the 320 assessment patches.

---

## 4. Canonical Frozen-Source Per-WSI Performance

Primary independent statistical units are the 5 individual assessment slides:

| WSI ID | Hospital-1 Role | Total Patches ($N$) | Tumor Patches ($y=1$) | Normal Patches ($y=0$) | AUROC | AUPRC | F1 Score | ECE | Brier Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **WSI 11** | Assessment | 64 | 36 | 28 | 0.9939 | 0.9958 | 0.9315 | 0.0588 | 0.053558 |
| **WSI 13** | Assessment | 64 | 48 | 16 | 0.9760 | 0.9951 | 0.9320 | 0.0885 | 0.076782 |
| **WSI 15** | Assessment | 64 | 0 | 64 | 0.5000* | 0.0000* | 0.0000* | 0.0416 | 0.028393 |
| **WSI 17** | Assessment | 64 | 15 | 49 | 0.9959 | 0.9866 | 0.8108 | 0.1515 | 0.111586 |
| **WSI 19** | Assessment | 64 | 16 | 48 | 0.9167 | 0.3292 | 0.4211 | 0.1720 | 0.144014 |
| **Pooled** | Assessment | **320** | **115** | **205** | **0.9736** | **0.9542** | **0.8632** | **0.0786** | **0.082867** |

*\*Note on WSI 15:* WSI 15 contains zero tumor patches in this sample ($0/64$). Discrimination metrics (AUROC/AUPRC) are mathematically undefined for single-class sets (conventionally set to $0.5$ and $0.0$), but calibration metrics (ECE $0.0416$, Brier $0.028393$) are well-defined.

---

## 5. Formal Correction Record

Historical Stage D2 documents are preserved. This record officially amends Section 3 of `docs/v2/STAGE_D2_NATURAL_POLICY_RESULTS.md`:

> **Correction Note (October 2026):**  
> In Table 1 of `STAGE_D2_NATURAL_POLICY_RESULTS.md`, the Brier score column for Frozen Source was erroneously transcribed as `0.0573`. The authoritative, verified Brier score for Frozen Source is **`0.082867`**, which matches the canonical execution artifact `stage_d2_benchmark_results.json` and the Stage C2.5 baseline audit.
