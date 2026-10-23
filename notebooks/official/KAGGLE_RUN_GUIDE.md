# Kaggle GPU Fresh Training Guide: MedMNIST-C, MIDOG, and NCT-CRC-HE

This guide provides instructions to run fresh training and Test-Time Adaptation (TTA) experiments on Kaggle using a free **GPU T4** accelerator.

All necessary scripts and dependencies are bundled in [`notebooks/official/UTTA_Med_Kaggle_GPU_Track.zip`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/notebooks/official/UTTA_Med_Kaggle_GPU_Track.zip).

---

## 1. Environment & Session Settings

For every Kaggle session:
1. **Accelerator**: Select **GPU T4 x1** (Settings panel on right).
2. **Internet**: Toggle **Internet ON** (Required for downloading pretrained ResNet-18 weights and datasets).
3. **Language**: Python 3.

---

## 2. Benchmark Tasks Overview

| Task | Script | Description | Extra Input Required | Runtime (T4) | Output Artifacts |
|---|---|---|---|---|---|
| **MedMNIST-C** | `kaggle_medmnistc_tta.py` | 9-class PathMNIST fresh training + Tent, Confidence, and UTTA under 5 corruptions | None (auto-installs `medmnist` and downloads) | ~15–20 min | `pathmnist_c_tta_s42.json`<br>`pathmnist_r18_s42.pt` |
| **MIDOG 2021** | `kaggle_midog_patches.py` | Mitosis vs impostor classification proxy, Leave-One-Scanner-Out (LODO) TTA | None (auto-imports directly from Zenodo 4643381) or local Kaggle input | ~10–15 min | `midog_patches_lodo_s42.json`<br>`midog_resnet18_s42_BEST.pt` |
| **NCT-CRC-HE** | `kaggle_nctcrc_p1.py` | 9-class colorectal histology fresh training on 100K + τ calibration on 10% + TTA on 7K validation | None (auto-downloads from HF / Zenodo) or add NCT-CRC dataset | ~1.5–2.5 hours | `nctcrc_p1_s42.json`<br>`nctcrc_resnet18_s42_BEST.pt` |
| **MIDOG Audit** | `kaggle_midog_source_localization_audit_h3_1_r4.py` | Fail-closed cryptographic localization audit for Faster R-CNN detectors | MIDOG WSI TIFFs + `source_fasterrcnn_s*.pt` | ~15 min | `utta_med_h3_1_r4_localization_audit.zip` |

---

## 3. Step-by-Step Execution Instructions

### A. Task 1: MedMNIST-C (PathMNIST + Corruptions)
1. In a Kaggle notebook session (GPU T4, Internet ON), upload [`notebooks/official/kaggle_medmnistc_tta.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/notebooks/official/kaggle_medmnistc_tta.py) and [`notebooks/official/tta_kit.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/notebooks/official/tta_kit.py) into `/kaggle/working/` (or upload `UTTA_Med_Kaggle_GPU_Track.zip` and run `!unzip UTTA_Med_Kaggle_GPU_Track.zip`).
2. Run the command:
   ```bash
   python kaggle_medmnistc_tta.py
   ```
3. What happens:
   - Downloads PathMNIST dataset.
   - Trains ResNet-18 from scratch for 6 epochs on 89,996 patches.
   - Validates on validation split, computing source test accuracy and macro F1.
   - Computes epistemic uncertainty threshold $\tau$ (70th percentile) and confidence threshold via Monte Carlo dropout ($N=20$).
   - Evaluates source under synthetic MedMNIST-C corruptions (Gaussian noise, defocus blur, brightness, contrast, JPEG).
   - Evaluates Tent, Confidence Gating, and UTTA adaptation on corrupted test streams.
   - Saves `pathmnist_c_tta_s42.json` and `pathmnist_r18_s42.pt` to `/kaggle/working/`.

---

### B. Task 2: MIDOG 2021 (Scanner Shift Proxy — Direct Zenodo Import)
1. In Kaggle (GPU T4, Internet ON), run the command:
   ```bash
   python kaggle_midog_patches.py
   ```
   *(Optional: If you have already attached a Kaggle dataset containing `MIDOG.json` + images, it will use the local files automatically; otherwise, it downloads directly from Zenodo record [4643381](https://zenodo.org/records/4643381)).*
2. What happens:
   - Auto-discovers local files or downloads `MIDOG.json` and required `.tiff` images directly from Zenodo 4643381.
   - Parses scanner metadata (Hamamatsu XR, Hamamatsu S360, Aperio CS2, Leica GT450).
   - Holds out the test scanner for out-of-domain evaluation.
   - Crops 64x64 patches centered on verified 50x50 mitotic/non-mitotic bounding boxes.
   - Trains ResNet-18 binary classifier on training scanner patches for 7 epochs.
   - Calibrates $\tau$ on in-domain held-out validation patches.
   - Evaluates Source, Tent, Random Gating, Confidence Gating, and UTTA on the unseen scanner.
   - Computes AUROC, F1, ECE, Brier score, and Harm Rate vs Correction Rate.
   - Saves `midog_patches_lodo_s42.json` and `midog_resnet18_s42_BEST.pt` to `/kaggle/working/`.

---

### C. Task 3: NCT-CRC-HE (9-Class Colorectal Histology)
1. Open [`notebooks/official/kaggle_nctcrc_p1.ipynb`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/notebooks/official/kaggle_nctcrc_p1.ipynb) on Kaggle or execute [`notebooks/official/kaggle_nctcrc_p1.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/notebooks/official/kaggle_nctcrc_p1.py):
   ```bash
   python kaggle_nctcrc_p1.py
   ```
2. What happens:
   - Loads NCT-CRC-HE-100K and CRC-VAL-HE-7K (from `/kaggle/input` if attached, or streaming via HuggingFace `1aurent/NCT-CRC-HE` / Zenodo).
   - Partitions a 10% stratified validation split for early stopping and $\tau$ calibration.
   - Trains ResNet-18 (9 classes) for up to 8 epochs with patience=3.
   - Evaluates source model on out-of-domain CRC-VAL-HE-7K.
   - Performs Monte Carlo Dropout ($N=20$) on validation set to calculate $\tau = \text{percentile}_{70}(U)$.
   - Computes rank correlations: Spearman $\rho(U, 1 - \text{conf})$, Spearman $\rho(U, H)$, and Spearman $\rho(1 - \text{conf}, H)$.
   - Runs Tent, Random Gating, Confidence Gating, Entropy Gating, and UTTA on 7K target stream.
   - Saves `nctcrc_p1_s42.json` and `nctcrc_resnet18_s42_BEST.pt` to `/kaggle/working/`.

---

### D. Task 4: MIDOG Source Localization Audit (Faster R-CNN H3.1-R4b)
1. Run the command:
   `ash
   python kaggle_midog_source_localization_audit_h3_1_r4b.py
   `
2. What happens:
   - Auto-checks and installs missing dependencies (	orchvision, pillow, etc.).
   - Authenticates cryptographic SHA-256 provenance of the new certified Faster R-CNN source detector checkpoints:
     - Seed 42: f9c34437e37cb13e38a6179c247ea8069ce74869fbd180201590b6137347003
     - Seed 7: 3f1ed8e0e3997eac411cd41bf663a3eafe36fc6f37ec44ec03ba2b880a4fa58e
     - Seed 123: 44584d70480a7913eeff93e9ac6a8e4217c5ea90298d3de3623356817afd9348
   - Confirms zero contamination from held-out Scanner 3 (Cases 101–150).
   - Evaluates centroid Euclidean bipartite matching at 7.5 µm (30 px) against official MIDOG annotations.
   - Packages utta_med_h3_1_r4b_new_checkpoints.zip in /kaggle/working/.

---

### E. Task 5: MIDOG Target Adaptation (Stage H3.2-R3a)
1. Once Task 4 completes, run:
   `ash
   python kaggle_midog_target_adaptation_h3_2_r3a.py
   `
2. Evaluates the 5 adaptation arms across all 3 seeds on held-out Scanner 3 cases.

---

## 4. Collecting Results Back into Repository

Once training finishes on Kaggle:
1. Download the output JSON files from Kaggle's `/kaggle/working/` output panel.
2. Place them into the local repository at:
   - `results/optional/pathmnist/pathmnist_c_tta_s42.json`
   - `results/optional/midog/midog_patches_lodo_s42.json`
   - `results/optional/nctcrc/nctcrc_p1_s42.json`
3. Optional model weights can be stored in `checkpoints/` or local archival storage.
4. Run the integrity audit to ensure all primary and supplementary numbers remain locked:
   ```bash
   python scripts/verify_manuscript_tables.py
   ```
