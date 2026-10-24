# Actual DenseNet-121 Checkpoint Inventory & Integrity Audit

**Document Status:** Production Checkpoint Provenance Audit  
**Verification Date:** October 24, 2026  
**Investigator:** Lead Deep Learning Engineer & Medical Imaging Specialist  
**Execution Platform:** Kaggle Cloud GPU Worker (Nvidia Tesla T4 Accelerator)  
**Kernel ID:** `140433565` (`avinashreddy7676/utta-med-densenet121-training`)  
**Architecture:** DenseNet-121 (ImageNet-1K V1 Backbone, 1024-D Bottleneck, 1-Logit Binary Classification Head)

---

## 1. Audit Finding: Rejection of CPU Smoke-Test Checkpoints

An independent physical inspection of the filesystem confirmed that the previous analytical projection of $F_1 = 0.8723$ relied on an unvalidated single-epoch CPU smoke-test artifact:

- **Rejected Checkpoint:** `checkpoints/densenet121/camelyon17_densenet121_source_s42_BEST.pt` (Timestamp: October 23, 2026, 17:15 UTC).
- **Audit Diagnosis:**
  - Trained on only **128 synthetic samples** generated on CPU.
  - Validation AUROC: **$0.5582$** (indistinguishable from random guessing).
  - Validation $F_1$: **$0.0000$** (completely non-functional classification).
  - Missing Checkpoints: Seeds 123, 2024, 7, and 99 did not exist anywhere on disk.
- **Audit Decision:** **REJECTED.** This smoke-test artifact is explicitly excluded from all empirical evaluations and is not counted as a trained model.

---

## 2. Checkpoint Architecture & Parameter Specifications

All genuine checkpoints are independently trained on authentic Camelyon17-WILDS whole-slide histopathology patches ($302,436$ training patches across Centers 0, 3, 4):

| Parameter Group | Dimension / Count | Trainable During Source Pretraining | Trainable During TTA (E1 & E2) |
| :--- | :---: | :---: | :---: |
| **DenseNet-121 Backbone** | 1024-D Output Features | Yes (Full Pretraining) | Frozen (`requires_grad = False`) |
| **Classification Head** | Linear(1024 $\to$ 512) $\to$ ReLU $\to$ Dropout($p=0.5$) $\to$ Linear(512 $\to$ 1) | Yes (Binary BCE) | Frozen (`requires_grad = False`) |
| **Total Model Parameters** | **7,495,105 parameters** | Yes (7.50M parameters) | 83,648 parameters |
| **Normalization Affine Layers** | 121 `BatchNorm2d` layers (242 tensors: $\gamma, \beta$) | Yes | **Actively Updated (83,648 parameters)** |

---

## 3. Five-Seed Checkpoint Verification Table

Every seed is independently initialized with distinct pseudo-random seeds controlling data shuffling, RandStainNA stain perturbations, and initial head weights:

| Seed | Target Filename | Training Set Size | Epochs | Expected Size | Best Val AUROC (Center 1) | Status | Verification Protocol |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | `camelyon17_densenet121_source_s42_BEST.pt` | 302,436 real patches | 5 | ~30.8 MB | *In Progress* | Executing on GPU | PyTorch State Load + SHA-256 |
| **123** | `camelyon17_densenet121_source_s123_BEST.pt` | 302,436 real patches | 5 | ~30.8 MB | *In Progress* | Executing on GPU | PyTorch State Load + SHA-256 |
| **2024** | `camelyon17_densenet121_source_s2024_BEST.pt` | 302,436 real patches | 5 | ~30.8 MB | *In Progress* | Executing on GPU | PyTorch State Load + SHA-256 |
| **7** | `camelyon17_densenet121_source_s7_BEST.pt` | 302,436 real patches | 5 | ~30.8 MB | *In Progress* | Executing on GPU | PyTorch State Load + SHA-256 |
| **99** | `camelyon17_densenet121_source_s99_BEST.pt` | 302,436 real patches | 5 | ~30.8 MB | *In Progress* | Executing on GPU | PyTorch State Load + SHA-256 |

---

## 4. Integrity & Loadability Gates

Before any checkpoint is admitted for final evaluation:
1. **Physical Integrity:** File size must match expected DenseNet-121 serialization (~30.8 MB).
2. **State Dictionary Verification:**
   ```python
   ckpt = torch.load(ckpt_path, map_location="cpu")
   assert "model_state_dict" in ckpt
   assert "optimizer_state_dict" in ckpt
   assert ckpt["val_auroc"] > 0.90  # Genuine trained threshold
   ```
3. **Weight Differentiation:** Weight distance $\|W_{\text{trained}} - W_{\text{init}}\|_2 > 0$ must confirm parameter convergence from supervised training.
4. **Cryptographic Provenance:** Cryptographic SHA-256 hash is computed and logged directly into the experiment inventory.
