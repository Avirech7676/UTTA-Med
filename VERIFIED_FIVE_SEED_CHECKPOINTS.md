# Verified Five-Seed Checkpoint Preservation & Architecture Report

**Project:** UTTA-Med Project 2  
**Document ID:** `VERIFIED_FIVE_SEED_CHECKPOINTS.md`  
**Model Architecture:** DenseNet-121 (`torchvision.models.densenet121`) with Binary Linear Head (`nn.Linear(1024, 1)`) and Feature Dropout ($p=0.50$)  
**Authentic Training Split:** Camelyon17-WILDS Centers 0, 3, 4 ($302,436$ patches)  
**Permitted Validation Split:** Camelyon17-WILDS Center 1 ($34,904$ out-of-domain patches)  
**Strictly Isolated Test Split:** Camelyon17-WILDS Center 2 ($85,054$ target evaluation patches)  
**Execution Node:** 2 $\times$ Nvidia Tesla T4 Tensor Core GPUs (`avinashreddy7676/utta-med-densenet121-training`, Kernel ID: `140433565`)  

---

## 1. Formal Rejection of Synthetic CPU Checkpoint

In accordance with Phase 3 & Task 3 requirements:
* **The 128-sample CPU test checkpoint (`models/camelyon17_densenet121_source_s42.pt`, size 28.1 MB) is formally rejected as non-canonical and invalid.**
* It was generated during local pipeline dry-runs on an infinitesimal subset ($0.042\%$ of the training corpus).
* It produced degraded representations ($F_1 \approx 0.4376$) and must never be cited as an authentic trained baseline.
* The historical projected F1 of $0.8723$ is recognized as an ungrounded projection and is strictly excluded from all verified results.

---

## 2. Checkpoint Preservation Protocol & Durable Storage Schema

To ensure that checkpoints survive unexpected Kaggle container terminations or timeouts, every training session implements the following durable serialization schema:

### Per-Epoch Checkpoint Serialization
After every completed epoch $e \in \{1, \dots, 5\}$, the model serializes:
`checkpoints/densenet121/camelyon17_densenet121_source_s{seed}_epoch{e}.pt`

### Comprehensive State Dictionary Contents
Each saved checkpoint dictionary contains:
```python
checkpoint_dict = {
    # 1. Architectural & Experiment Metadata
    "seed": seed,
    "arch": "densenet121",
    "epoch": epoch,
    "timestamp": datetime.datetime.now().isoformat(),
    "dataset_config": {
        "source_centers": [0, 3, 4],
        "n_source_patches": 302436,
        "val_center": 1,
        "n_val_patches": 34904,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
    },
    
    # 2. Weights & Model Parameters
    "model_state_dict": model.state_dict(),
    "ema_state_dict": ema_model.model.state_dict(),
    "ema_decay": 0.999,
    
    # 3. Optimization & Schedule States
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    
    # 4. Full Random Number Generator States (for Exact Resumption)
    "rng_states": {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all(),
    },
    
    # 5. Validation Performance Metrics (Center 1)
    "val_metrics": {
        "auroc": val_m["auroc"],
        "f1": val_m["f1"],
        "recall": val_m["recall"],
        "precision": val_m["precision"],
        "accuracy": val_m["accuracy"],
        "brier": val_m["brier"],
    },
    "history": history,
}
```

### Best Model Tracking
If `val_m["auroc"]` exceeds the previous best validation AUROC, the file is duplicated to:
`checkpoints/densenet121/camelyon17_densenet121_source_s{seed}_BEST.pt`

---

## 3. Five-Seed Checkpoint Verification Registry

| Seed | Target Checkpoint Path | Backbone | Training Corpus | Best Epoch | Validation AUROC | Validation F1 | Checkpoint SHA-256 | Load Gate | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | `camelyon17_densenet121_source_s42_BEST.pt` | DenseNet-121 | Camelyon17 Centers 0, 3, 4 | In Run | Awaiting Finish | Awaiting Finish | Awaiting Finish | Pending Download | **IN_GPU_EXECUTION** |
| **123** | `camelyon17_densenet121_source_s123_BEST.pt` | DenseNet-121 | Camelyon17 Centers 0, 3, 4 | Queued | Pending Run | Pending Run | Pending Run | Queued | **QUEUED_MODULAR** |
| **2024** | `camelyon17_densenet121_source_s2024_BEST.pt` | DenseNet-121 | Camelyon17 Centers 0, 3, 4 | Queued | Pending Run | Pending Run | Pending Run | Queued | **QUEUED_MODULAR** |
| **7** | `camelyon17_densenet121_source_s7_BEST.pt` | DenseNet-121 | Camelyon17 Centers 0, 3, 4 | Queued | Pending Run | Pending Run | Pending Run | Queued | **QUEUED_MODULAR** |
| **99** | `camelyon17_densenet121_source_s99_BEST.pt` | DenseNet-121 | Camelyon17 Centers 0, 3, 4 | Queued | Pending Run | Pending Run | Pending Run | Queued | **QUEUED_MODULAR** |

---

## 4. Multi-Seed Execution Strategy Under the 6-Hour Kaggle Constraint

* **Hard Container Limit:** Kaggle terminates notebook sessions after **$6.0\text{ hours}$ ($21,600\text{ seconds}$)**.
* **Per-Seed Compute Requirement:**
  - 5 Epochs Training ($302,436$ patches $\times 5$): $\approx 4.0\text{ hours}$.
  - Validation & Center 2 Target Adaptation: $\approx 0.35\text{ hours}$.
  - Total time per seed: $\approx 4.35\text{ hours}$.
* **Timeout Conflict:** Running all 5 seeds sequentially in one session requires $5 \times 4.35 = \mathbf{21.75\text{ hours}}$, which is physically impossible within a single 6-hour Kaggle container.
* **Modular Solution:**
  1. **Seed 42:** Running in Kernel ID `140433565`. It will finish in $\approx 4.35\text{ hours}$ ($< 6.0\text{ hours}$) and produce its durable checkpoint.
  2. **Seeds 123, 2024, 7, 99:** Deployed as independent, modular single-seed Kaggle sessions. Each runs for $\approx 4.35\text{ hours}$, safely inside the 6.0-hour timeout window.
  3. **Zero Retraining:** Completed seeds are permanently preserved; only uncompleted seeds are scheduled.

---

## 5. Strict Target-Test Label Isolation

To protect scientific validity:
* **Zero Center 2 Access During Training:** Ground-truth labels from Center 2 ($N = 85,054$) are **never accessed, loaded, or inspected** during model training, learning-rate scheduling, early stopping, or checkpoint selection.
* **Selection Criterion:** Checkpoints are selected solely based on Out-of-Domain validation performance on Center 1 ($N = 34,904$).
* **Adaptation Isolation:** During test-time adaptation (E1 and E2), the model receives only unlabeled target patches. Ground truth labels are evaluated exclusively post-hoc by evaluation scripts to calculate final confusion matrices and ROC curves.
