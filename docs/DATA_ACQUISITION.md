# Dataset Acquisition & Configuration Guide

This document outlines instructions for downloading, verifying, and configuring the computational pathology datasets evaluated in the **SafeAdapt-TTA** research study.

---

## 1. Camelyon17-WILDS (Multicenter Histopathology Classification)

**Task:** Lymph node breast cancer metastasis patch classification under multicenter shift.  
**License:** CC0 1.0 Universal Public Domain Dedication.  
**Source:** CAMELYON17 Challenge / WILDS Benchmark (Koh et al., ICML 2021).

### Option A: Automatic Download via WILDS Python Package

```python
import wilds

# Downloads official Camelyon17 dataset (version 1.0)
dataset = wilds.get_dataset(dataset="camelyon17", download=True, root_dir="./data")
```

### Option B: Direct Parquet Download (Recommended for Fast Local Testing)

Download the pre-extracted patch Parquet files:
- `train-*.parquet` (Hospitals 0, 3, 4; $n = 302{,}436$ patches)
- `validation-*.parquet` (Hospital 1 held-out development; $n = 34{,}904$ patches)
- `test-*.parquet` (Hospital 2 held-out target; $n = 85{,}054$ patches)

Place the downloaded files into:
```
data/camelyon17_v1.0/
├── metadata.csv
├── train-*.parquet
├── validation-*.parquet
└── test-*.parquet
```

### Environment Variable Setup
Set `CAMELYON17_ROOT` in your environment:
```bash
# Linux / macOS
export CAMELYON17_ROOT="/path/to/data/camelyon17_v1.0"

# Windows PowerShell
$env:CAMELYON17_ROOT = "C:\path\to\data\camelyon17_v1.0"
```

Verify data integrity:
```bash
python scripts/inspect_local_data.py
```

---

## 2. MIDOG 2021 Challenge (Dense Mitotic Object Detection)

**Task:** Cross-scanner mitotic figure detection under severe optical shift.  
**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0).  
**Source:** MICCAI 2021 Mitosis Domain Generalization Challenge (Aubreville et al.).

- **Annotations:** Standardized COCO-format ground-truth coordinates are already pre-packaged in the repository:
  - `data/MIDOG.json`
  - `data/midog_scanner3_coco_metadata.json`
- **WSI Image Files:** Scanned breast cancer tissue images (Scanners 1, 2, 3) can be acquired from the official challenge repository at [Grand Challenge MIDOG 2021](https://midog2021.grand-challenge.org/).

---

## 3. NCT-CRC-HE-100K (Multi-Class Colorectal Histology)

**Task:** 9-class colorectal tissue phenotyping (DEB, LYM, MUC, MUS, NORM, STR, TUM).  
**License:** CC-BY 4.0.  
**Source:** Kather et al., PLOS Medicine 2019 / Zenodo (`10.5281/zenodo.1214456`).

Download the 100,000 non-overlapping H&E tissue patches from Zenodo:
```bash
# Example curl download
curl -L -o NCT-CRC-HE-100K.zip https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
unzip NCT-CRC-HE-100K.zip -d data/nct_crc/
```

Execute the evidential Subjective Logic benchmark:
```bash
python scripts/run_evidential_nct_benchmark.py
```
