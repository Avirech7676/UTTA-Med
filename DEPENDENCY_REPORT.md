# UTTA-Med 2.0: Dependency & Execution Environment Report

**Date:** 2026-10-22  
**Host OS:** Microsoft Windows 11  
**Python Runtime:** Python 3.14.2 (Anaconda environment `avinash` at `C:\Users\avina\anaconda3\envs\avinash\python.exe`)  
**Kaggle API:** Version 2.2.4 (Authenticated via `~/.kaggle/kaggle.json`)

---

## 1. Local Python Environment Dependencies

| Package Name | Installed Version | Status | Primary Usage |
| :--- | :---: | :---: | :--- |
| `torch` | 2.14.0+cpu | Verified | Deep learning inference, tensor operations, optimizer state |
| `torchvision` | 0.29.0+cpu | Verified | ResNet-18, Faster R-CNN architectures, transforms |
| `scipy` | 1.18.0 | Verified | Statistical hypothesis testing, bootstrapping, Wilcoxon |
| `numpy` | 2.4.6 | Verified | Empirical trajectory arrays, numerical vectorization |
| `scikit-learn` | 1.9.0 | Verified | ROC-AUC, PR-AUC, F1, calibration metrics (Brier, ECE) |
| `pandas` | 3.0.4 | Verified | Results tabulation, trajectory logging, manifests |
| `pyarrow` | 24.0.0 | Verified | Parquet metadata reading for Camelyon17-WILDS |
| `opencv-python` (`cv2`) | 4.12.0 | Verified | Morphological filtering, bounding box rendering, optical density |
| `Pillow` (`PIL`) | 12.2.0 | Verified | High-resolution tile loading and transforms |
| `matplotlib` | 3.11.0 | Verified | Diagnostic plotting, ROC curves, Grad-CAM overlays |
| `pytest` | 9.1.1 | Verified | Automated test suite execution (120 tests passing) |
| `pyyaml` (`yaml`) | 6.0.3 | Verified | Configuration file parsing |
| `python-docx` (`docx`) | 1.2.0 | Verified | Monograph and report document generation |
| `pypdf` | 6.18.1 | Verified | IEEE PDF geometry and page budget certification |
| `kaggle` | 2.2.4 | Verified | Remote GPU orchestration CLI and Python API |

---

## 2. Remote Kaggle GPU Acceleration Profile

* **Authentication:** Active (`~/.kaggle/kaggle.json` configured and verified with `kaggle.KaggleApi().authenticate()`).
* **Target Accelerator:** `NvidiaTeslaT4` (Single T4 16GB VRAM) or `NvidiaTeslaP100`.
* **Execution Paradigm:** Fully automated orchestrator (`scripts/kaggle/kaggle_orchestrator.py`) handling:
  1. Experiment configuration serialization (`experiment_config.yaml`).
  2. Notebook kernel metadata generation (`kernel-metadata.json`).
  3. CLI job pushing (`kaggle kernels push`).
  4. Status polling and log monitoring (`kaggle kernels status`, `kaggle kernels logs`).
  5. Automatic result and checkpoint retrieval (`kaggle kernels output`).
* **Storage Rules:**
  - Kaggle ephemeral sessions are capped at 12 hours (GPU).
  - Checkpoints saved per epoch / evaluation step to Kaggle `/kaggle/working`.
  - Checkpoints retrieved and validated upon job completion.

---

## 3. Local vs Remote Execution Protocol

| Task Type | Execution Target | Hardware Profile | Fallback |
| :--- | :--- | :--- | :--- |
| **Unit & Integration Tests** | Local Workspace | CPU (Fast, 18.34s) | Pytest in Conda `avinash` |
| **Statistical Bootstrapping** | Local Workspace | CPU Multi-core | Vectorized NumPy / SciPy |
| **Adaptive Uncertainty Logic** | Local Workspace | CPU Test Mock | Simulated tensor batches |
| **Bio-SafeAdapt Controller Logic** | Local Workspace | CPU Test Mock | Synthetic tile tensors |
| **Full 5-Seed Retraining** | Remote Kaggle | Nvidia Tesla T4 GPU | Automated via `kaggle_orchestrator.py` |
| **Dense Detection Adaptation** | Remote Kaggle | Nvidia Tesla T4 GPU | Automated via `kaggle_orchestrator.py` |
