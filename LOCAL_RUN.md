# Running UTTA-Med Locally

This guide explains how to verify, evaluate, and reproduce UTTA-Med experiments locally on Windows or Linux.

---

## 1. Environment Setup

### Prerequisites
- Python 3.10 or 3.11 (Python 3.11 recommended)
- Optional: NVIDIA GPU with CUDA for accelerated adaptation and inference

### Setup Virtual Environment
```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Dataset Configuration

UTTA-Med uses the official **Camelyon17-WILDS** benchmark formatted as HuggingFace Parquet files:
- `train-*.parquet`
- `validation-*.parquet` (split: hospital 1 = OOD validation; hospitals 0, 3, 4 = ID validation)
- `test-*.parquet` (hospital 2 = OOD target test)

Set the `CAMELYON17_ROOT` environment variable pointing to your data directory:

```powershell
# Windows PowerShell
$env:CAMELYON17_ROOT = "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"
```

```bash
# Linux / Bash
export CAMELYON17_ROOT="/path/to/Camelyon17-Data"
```

Verify your dataset splits and check for zero leakage:
```powershell
python scripts\inspect_local_data.py
```

---

## 3. Checkpoints

All five official source model checkpoints are included in `checkpoints/`:
- `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`
- `checkpoints/camelyon17_resnet18_source_s123_BEST.pt`
- `checkpoints/camelyon17_resnet18_source_s2024_BEST.pt`
- `checkpoints/camelyon17_resnet18_source_s7_BEST.pt`
- `checkpoints/camelyon17_resnet18_source_s99_BEST.pt`

Their cryptographic SHA256 checksums are frozen in `reproducibility/ARTIFACT_MANIFEST.csv`.

---

## 4. Verification Suite (Offline — No GPU Required)

Run the automated verification suite to validate architecture freeze, protocol rules, and statistical consistency:

```powershell
# 1. Unit tests
pytest tests/ -q

# 2. Configuration isolation check
python scripts\verify_config.py

# 3. Protocol & zero-leakage check
python scripts\verify_protocol.py

# 4. Gate guardrails check
python scripts\verify_step2_guardrails.py

# 5. Audit 5-seed metrics and checkpoint hashes
python scripts\audit_5seeds.py

# 6. Verify cross-table and metric consistency
python scripts\verify_consistency.py
```

---

## 5. Execution Protocol

### Source Training (Optional — Checkpoints already provided)
Official entry point:
```powershell
python scripts\train_source.py --config configs\camelyon17.yaml --seed 42
```
*Note: Source training uses ResNet-18, 1-logit BCE loss, MC-Dropout $p=0.5$, Adam optimizer ($lr=1e-4$), early stopping on Center 1 (OOD val).*

### Test-Time Adaptation
Run TTA methods on the target domain (Center 2):
```powershell
# UTTA-Med (uses per-seed tau from configs/tau_by_seed.yaml)
python scripts\adapt.py --checkpoint checkpoints\camelyon17_resnet18_source_s42_BEST.pt --method utta

# Tent (Adam, lr=1e-5, bn_freeze_stats)
python scripts\adapt.py --checkpoint checkpoints\camelyon17_resnet18_source_s42_BEST.pt --method tent

# EATA
python scripts\adapt.py --checkpoint checkpoints\camelyon17_resnet18_source_s42_BEST.pt --method eata
```

### Key Protocol Rules
1. **Uncertainty Threshold ($\tau$):** Chosen strictly as the 70th percentile of predictive variance $U$ on Hospital 1 validation data (`configs/tau_by_seed.yaml`). **Never** tuned on test Hospital 2.
2. **Batch Normalization:** `bn_freeze_stats` — batch normalization affine parameters $(\gamma, \beta)$ are updated; running mean and variance are frozen to prevent empirical drift.
3. **No Retraining Required:** All headline tables (`results/tables/table2_five_seed_primary.csv`) and statistical reports (`results/statistical/five_seed_final/STATISTICAL_REPORT.md`) are pre-computed from the frozen experimental runs.
