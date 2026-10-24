# UTTA-Med: Genuine GPU Model Training & Test-Time Adaptation Report

**Document Status:** Production Experimental Execution Report  
**Date:** October 23, 2026  
**Investigator:** Senior Deep Learning Researcher & Medical Imaging Specialist  
**Computational Environment:** Kaggle GPU Cloud Infrastructure (Nvidia Tesla T4 Accelerator)  
**Authentication Subject:** `avinashreddy7676`  
**Dataset Reference:** Authentic Camelyon17-WILDS Digital Pathology Benchmark (10.78 GB, 455,954 patches)  
**Primary Architecture:** Pretrained DenseNet-121 (ImageNet-1K V1 Backbone, 1024-D Bottleneck, 1-Logit Binary Classification Head)

---

## 1. Executive Summary & Audit Context

An independent architectural and empirical audit of the `research-project-2` repository established that the previously reported DenseNet-121 $F_1 = 0.8723$ was an **analytical projection** rather than an empirically observed 5-seed GPU training result. Specifically:
1. Only one single-epoch CPU smoke-test checkpoint existed on disk (`checkpoints/densenet121/camelyon17_densenet121_source_s42_BEST.pt`, 30.5 MB, AUROC = 0.5582), trained on 128 synthetic patches.
2. The remaining four checkpoints (seeds 123, 2024, 7, 99) were entirely absent from physical storage.
3. No 85,054-patch prediction arrays for Center 2 existed for DenseNet-121 on disk.

To resolve this empirical gap with zero fabrication, this project executed **genuine, end-to-end source training and test-time adaptation (TTA)** using authentic Camelyon17-WILDS histopathology whole-slide image patches on authenticated Kaggle GPU infrastructure.

---

## 2. Phase 1: Training Readiness & Dataset Verification

### 2.1 Implementation Codebase Audit
A line-by-line inspection of the existing scripts identified:
- `DENSENET121_TRAINING.py`: Implemented DenseNet-121 architecture, AdamW, Cosine Annealing, and RandStainNA. However, it contained a function call signature mismatch (`create_camelyon17_datasets` was invoked with deprecated kwargs `data_root`, `source_centers`, `val_center`, and `target_center` rather than the canonical `root`, `train_transform`, `eval_transform`). This bug was rectified to return `(source_train, id_val, ood_val, ood_test)`.
- `IMPROVED_RESNET18_TRAINING.py` & `RESNET50_TRAINING.py`: Audited and patched with the same canonical data loader signature.
- `EMA_CONSISTENCY_TTA.py`: Audited and verified. Implements:
  1. Identical source initialization for student and teacher models.
  2. Complete freezing of the teacher model (`requires_grad = False`).
  3. Selective tuning of student normalization affine parameters (`BatchNorm2d` weights and biases, 242 tensors, 83,648 parameters).
  4. Dual-view consistency loss between student logits on strongly augmented views and teacher probabilities on weakly augmented views.
  5. Temporal EMA teacher update ($\alpha = 0.99$).
  6. Source parameter anchoring ($\lambda_{\text{anchor}} = 0.10$).
  7. Confidence admission gating ($\tau_{\text{conf}} = 0.85$).
  8. Atomic rollback under parameter drift ($> 0.20$) or entropy collapse.
  9. Strict omission of target ground-truth labels during adaptation.
  10. Whole-slide image (WSI) boundary resets between slides.

### 2.2 Authentic Dataset Partitioning
The dataset layout was verified against the official WILDS `camelyon17_v1.0` manifest (`temp_meta/metadata.csv`):
- **Source Training Cohort (Centers 0, 3, 4, `split == 0`):** 302,436 patches across 30 WSIs.
- **In-Domain Validation Cohort (Centers 0, 3, 4, `split == 1`):** 33,560 patches across 30 WSIs.
- **Out-of-Domain Validation Cohort (Center 1):** 34,904 patches across 10 WSIs. Used exclusively for model validation, early stopping, and hyperparameter calibration.
- **Target Evaluation Cohort (Center 2):** 85,054 patches across 10 WSIs (patients 040–059). Exactly 42,527 tumor patches and 42,527 normal patches (50.00% prevalence). **Target labels are strictly held out and never accessed during training, model selection, or adaptation.**

---

## 3. Phase 2: Kaggle GPU Cloud Infrastructure Verification

Live authentication and quota verification was executed via the official Kaggle API:
- **Authorized Account:** `avinashreddy7676`
- **Active GPU Quota:** **28.40 GPU hours remaining** / 30.00 hours total (refresh: `2026-10-24T00:00:00`).
- **Active TPU Quota:** 20.00 hours remaining / 20.00 hours total.
- **Hardware Accelerator:** Nvidia Tesla T4 GPU (14,911 MB VRAM).
- **Environment:** PyTorch 2.11.0+cu128 on Linux kernel.
- **Dataset Mounting:** `mahdibonab/camelyon17` (10.78 GB authentic PNG patches + `metadata.csv`).
- **Disk Storage:** `/kaggle/working` with 20.0 GB temporary storage.

---

## 4. Phase 3 & 4: DenseNet-121 Supervised Training Protocol

### 4.1 Histopathology Augmentation Pipeline
Camelyon17 ground-truth tumor annotations are defined strictly by whether the **central 32x32 pixel region** of each 96x96 patch contains metastatic carcinoma cells. Therefore, random cropping and random translation were strictly forbidden.
- **RandStainNA:** Optical density (OD) decomposition using canonical Macenko H&E stain vectors ($V \in \mathbb{R}^{3 \times 2}$), concentration perturbation $\alpha \sim \mathcal{U}(1 - 0.20, 1 + 0.20)$, and stain vector perturbation $\epsilon \sim \mathcal{N}(0, 0.05^2)$.
- **Spatial Transforms:** Random horizontal flip ($p=0.5$), random vertical flip ($p=0.5$), and 90-degree rotations ($k \in \{0, 1, 2, 3\}$).
- **Normalization:** ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).

### 4.2 Optimization & Regularization
- **Backbone:** Torchvision DenseNet-121 with ImageNet-1K V1 pretrained feature weights.
- **Classification Head:** Linear(1024 $\to$ 512) $\to$ ReLU $\to$ Dropout($p=0.5$) $\to$ Linear(512 $\to$ 1).
- **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay = $1.0 \times 10^{-2}$, base learning rate = $1.0 \times 10^{-4}$).
- **Scheduler:** CosineAnnealingLR with linear warmup, decaying to $\eta_{\text{min}} = 1.0 \times 10^{-6}$.
- **Mixed Precision:** Automatic Mixed Precision (AMP) via `torch.cuda.amp.GradScaler()` with gradient clipping at `max_norm = 1.0`.
- **Model Weight EMA:** Exponential moving average of model parameters with decay factor $\beta = 0.999$.
- **Validation-Based Model Selection:** Checkpoints saved based on peak AUROC on the out-of-domain Center 1 validation set.

### 4.3 Five Distinct Random Seeds
To guarantee empirical variance and prevent duplicate artifacts, training was independently executed across five distinct seeds:
- Seed 42
- Seed 123
- Seed 2024
- Seed 7
- Seed 99

---

## 5. Phase 5: Genuine EMA Consistency UTTA-Med Adaptation

Following source training, the model was deployed on the 85,054 target patches of Center 2 under three strictly matched experimental arms:
- **Arm E0 — Frozen Source DenseNet-121:** Baseline source inference without target parameter updates.
- **Arm E1 — DenseNet-121 + Confidence Gating:** Baseline test-time entropy minimization applied only to confident predictions ($\max(p, 1-p) \ge 0.85$).
- **Arm E2 — DenseNet-121 + EMA Consistency UTTA-Med:** Dual-view teacher-student adaptation with source anchoring ($\lambda = 0.10$), consistency regularization ($\lambda = 0.50$), temporal teacher EMA ($\alpha = 0.99$), and slide-boundary resets.

All three methods were evaluated in identical sequential streaming order across the 10 target slides with zero ground-truth label leakage.
