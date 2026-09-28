# Legacy vs official notebooks

Use the official runners. Do not re-run the legacy ones for paper numbers.

| File | Status | Why |
|---|---|---|
| `kaggle_seed123_full.ipynb` (and 2024/7/99) | **official** | 5-seed headline |
| `kaggle_gate_d_utta.ipynb` | **official** seed-42 Gate D | frozen-source U |
| `kaggle_wsi_remaining_seeds.ipynb` | **official** WSI CIs | |
| `kaggle_frozen_utta_gradcam.ipynb` | **official** Figure 7 UTTA column | AUROC ~0.95 |
| `kaggle_gradcam_s42.ipynb` | **legacy** | 3k slide-order cap, TP=0 |
| `kaggle_adapt_steps_gradcam.ipynb` | **legacy UTTA CAM** | online U, UTTA AUROC 0.32; **k-step table is valid** |
| `01_colab_source_train.ipynb` | **legacy host** | training recipe still valid; prefer Kaggle seeds |
| `kaggle_utta_med_gate_bc.ipynb` | **legacy** | overwrote best ckpt every epoch |
