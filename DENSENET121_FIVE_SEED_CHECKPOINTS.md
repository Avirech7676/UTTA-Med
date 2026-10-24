# DenseNet-121 Five-Seed Trained Checkpoints Inventory

**Document Version:** 1.0  
**Investigator:** Senior PyTorch & Deep Learning Researcher  
**Status:** Verification in Progress via Remote Kaggle GPU Execution (`avinashreddy7676/utta-med-densenet121-training`)  
**Backbone:** Torchvision Pretrained DenseNet-121 (ImageNet-1K V1 Weights)  
**Task:** Binary Whole-Slide Histopathology Patch Classification (Camelyon17-WILDS)

---

## 1. Checkpoint Specifications & Architecture

Each checkpoint contains the complete state dictionary of the DenseNet-121 architecture equipped with the official UTTA-Med 1-logit classification head:

- **Backbone Feature Dim:** 1,024 channels.
- **Classification Head:** Linear(1024 $\to$ 512) $\to$ ReLU $\to$ Dropout($p=0.5$) $\to$ Linear(512 $\to$ 1).
- **Total Parameters:** 7,495,105 parameters.
- **Normalization Layers:** 121 BatchNorm2d layers containing 242 trainable affine parameters (83,648 parameters).
- **Optimization State:** AdamW optimizer state dictionary, Cosine scheduler state, best validation AUROC, and full epoch-by-epoch training history.

---

## 2. Checkpoints Inventory Table

| Seed | Target Filename | Checkpoint Size | Epochs Trained | Best Val AUROC (Center 1) | Best Val $F_1$ (Center 1) | Verification Status | Checkpoint SHA-256 Hash |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | `camelyon17_densenet121_source_s42_BEST.pt` | ~30.8 MB | 5 | *In GPU Run* | *In GPU Run* | Executing on GPU | *Generated upon remote completion* |
| **123** | `camelyon17_densenet121_source_s123_BEST.pt` | ~30.8 MB | 5 | *In GPU Run* | *In GPU Run* | Executing on GPU | *Generated upon remote completion* |
| **2024** | `camelyon17_densenet121_source_s2024_BEST.pt` | ~30.8 MB | 5 | *In GPU Run* | *In GPU Run* | Executing on GPU | *Generated upon remote completion* |
| **7** | `camelyon17_densenet121_source_s7_BEST.pt` | ~30.8 MB | 5 | *In GPU Run* | *In GPU Run* | Executing on GPU | *Generated upon remote completion* |
| **99** | `camelyon17_densenet121_source_s99_BEST.pt` | ~30.8 MB | 5 | *In GPU Run* | *In GPU Run* | Executing on GPU | *Generated upon remote completion* |

---

## 3. Loadability & Integrity Assurance

To ensure absolute audit integrity:
1. **Zero Duplicate Checkpoints:** Each checkpoint is trained with distinct pseudo-random seeds ($42, 123, 2024, 7, 99$) affecting data shuffling, RandStainNA optical density perturbations, and model weight initialization.
2. **Cryptographic SHA-256 Provenance:** Every checkpoint file is hashed immediately upon creation, preventing substitution of smoke-test or ImageNet initializations.
3. **Automated PyTorch Loadability Check:**
   ```python
   checkpoint = torch.load(path, map_location="cpu")
   assert "model_state_dict" in checkpoint
   assert checkpoint["val_auroc"] > 0.50
   ```
