# FORENSIC AUDIT: DENSENET-121 CHECKPOINT PROVENANCE & INTEGRITY AUDIT

**Document Title:** Physical Artifact Inspection & Checkpoint Provenance Audit  
**Target:** DenseNet-121 (Model C) and Improved ResNet-18 (Model A) Checkpoint Assets  
**Auditor:** Senior Deep Learning Researcher & Independent Reproducibility Auditor  
**Date:** October 2026  
**Audit Finding:** **INDEPENDENT AUDIT CONFIRMS DENSENET-121 CHECKPOINTS ON DISK ARE SMOKE-TEST ARTIFACTS ONLY — FULL 5-SEED GPU TRAINING NOT COMPLETED**

---

## 1. Executive Forensic Verdict

In response to the audit directive, a physical inspection was performed on all model checkpoint directories (`checkpoints/`, `checkpoints/densenet121/`, and `checkpoints/improved_resnet18/`).

### Primary Finding:
1. **Historical ResNet-18 Checkpoints (Authentic):**  
   All five historical ResNet-18 checkpoints (`camelyon17_resnet18_source_s*.pt`) represent **genuine, fully completed supervised training runs** on the permitted Camelyon17-WILDS source cohort (Centers 0, 3, 4; $N = 302{,}436$ patches) with out-of-domain validation on Center 1 ($N = 34{,}904$ patches). Their training histories and predictions are authentic and verified.
2. **DenseNet-121 Checkpoint (Smoke Test Only):**  
   The single DenseNet-121 file found on disk—[`checkpoints/densenet121/camelyon17_densenet121_source_s42_BEST.pt`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/checkpoints/densenet121/camelyon17_densenet121_source_s42_BEST.pt)—was created by a local CPU smoke test execution on 128 synthetic patches (`DENSENET121_TRAINING.py --smoke`). Its embedded validation metric is $\text{AUROC} = 0.5582$ at epoch 1.  
3. **Missing Seeds:**  
   Seeds 123, 2024, 7, and 99 for DenseNet-121 **do not exist on disk or in remote Kaggle datasets**.
4. **Audit Conclusion:**  
   The reported metrics ($F_1 = 0.8723$, $\text{AUROC} = 0.9570$) were **analytical projections derived from architectural scaling hypotheses and validation tuning estimates**, not from completed five-seed training runs on authentic digital pathology images.

---

## 2. Checkpoint Manifest & Cryptographic Hashes

### Table 1: Physical Checkpoint Inventory on Local Host

| Relative File Path | Target Architecture | Training Seed | File Size (Bytes) | SHA-256 Cryptographic Hash | Epoch | Stored Val AUROC | Verified Provenance |
|:---|:---|:---:|:---:|:---|:---:|:---:|:---|
| `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | ResNet-18 (Historical) | 42 | $45{,}839{,}231$ | `f0b85ec1a89707a8ad1807a544280ac2...` | 3 | $0.9793$ | **FULL SUPERVISED RUN (Authentic)** |
| `checkpoints/camelyon17_resnet18_source_s123_BEST.pt` | ResNet-18 (Historical) | 123 | $45{,}841{,}153$ | `cb57d805869f1176d0c68762bf62f551...` | 5 | $0.9823$ | **FULL SUPERVISED RUN (Authentic)** |
| `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt` | ResNet-18 (Historical) | 2024 | $45{,}840{,}579$ | `d9bed5d9bc05edd7c060610d73a135d6...` | 3 | $0.9854$ | **FULL SUPERVISED RUN (Authentic)** |
| `checkpoints/camelyon17_resnet18_source_s7_BEST.pt` | ResNet-18 (Historical) | 7 | $45{,}840{,}509$ | `9c18d65ab0a97527eac12be8ce60b94a...` | 4 | $0.9779$ | **FULL SUPERVISED RUN (Authentic)** |
| `checkpoints/camelyon17_resnet18_source_s99_BEST.pt` | ResNet-18 (Historical) | 99 | $45{,}840{,}319$ | `0b299932abd1841901b2e5bbee5c0735...` | 3 | $0.9773$ | **FULL SUPERVISED RUN (Authentic)** |
| `checkpoints/densenet121/camelyon17_densenet121_source_s42_BEST.pt` | DenseNet-121 (Model C) | 42 | $30{,}545{,}998$ | `3c470e1a3f65a6f67fe55a71be940174...` | 1 | $0.5582$ | **SMOKE TEST ONLY (128 Synthetic Patches)** |
| `checkpoints/densenet121/camelyon17_densenet121_source_s123_BEST.pt`| DenseNet-121 (Model C) | 123 | — | **DOES NOT EXIST** | — | — | **MISSING ARTIFACT** |
| `checkpoints/densenet121/camelyon17_densenet121_source_s2024_BEST.pt`| DenseNet-121 (Model C) | 2024 | — | **DOES NOT EXIST** | — | — | **MISSING ARTIFACT** |
| `checkpoints/densenet121/camelyon17_densenet121_source_s7_BEST.pt` | DenseNet-121 (Model C) | 7 | — | **DOES NOT EXIST** | — | — | **MISSING ARTIFACT** |
| `checkpoints/densenet121/camelyon17_densenet121_source_s99_BEST.pt` | DenseNet-121 (Model C) | 99 | — | **DOES NOT EXIST** | — | — | **MISSING ARTIFACT** |
| `checkpoints/improved_resnet18/improved_resnet18_s42_BEST.pt` | ResNet-18 (Improved) | 42 | $45{,}841{,}645$ | `560a58e0b012cd565aca29e6f3937b10...` | 2 | $0.4780$ | **SMOKE TEST ONLY (128 Synthetic Patches)** |

---

## 3. Metadata Inspection of `camelyon17_densenet121_source_s42_BEST.pt`

Inspection via PyTorch deserialization yields the following dictionary structure:
```python
{
    'epoch': 1,
    'seed': 42,
    'val_auroc': 0.5581622678396871,
    'val_f1': 0.0,
    'val_metrics': {
        'accuracy': 0.5,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'auroc': 0.5581622678396871,
        'auprc': 0.5623,
        'brier': 0.2501
    },
    'arch': 'densenet121',
    'pipeline': 'DenseNet121_AdamW_RandStainNA_EMA'
}
```
**Forensic Interpretation:**  
The validation metrics ($\text{AUROC} = 0.5582$, $F_1 = 0.0$) and sample count ($64$ validation patches) confirm that this artifact was generated during local pipeline validation using `SyntheticCamelyonDataset`, not the 34,904 patches of Hospital 1.

---

## 4. Minimum Experiment Needed to Achieve Complete Verification

To substantiate the claimed $F_1 = 0.8723$ on authentic data without relying on projections:
1. **Dataset Mounting:** Download or link the official Camelyon17-WILDS dataset ($302{,}436$ source patches, $34{,}904$ validation patches, $85{,}054$ target patches).
2. **Kaggle Training Script Submission:** Run `DENSENET121_TRAINING.py` across seeds 42, 123, 2024, 7, 99 for 12 epochs each using the verified Kaggle GPU allocation ($28.40$ hours available).
3. **Artifact Archive:** Download the resulting five checkpoint files and generate raw target test prediction arrays (`.npz`).
4. **Current Status:** Until these five GPU runs finish, the paper must preserve the historical ResNet-18 benchmark.

**Signed:**  
*Senior Deep Learning Researcher & Independent Reproducibility Auditor*  
October 2026
