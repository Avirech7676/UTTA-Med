# Installation and Environment Setup

This guide provides instructions for setting up the environment required to run, evaluate, and reproduce the **SafeAdapt-TTA** framework and benchmarks across Windows, Linux, and macOS.

---

## 1. System Requirements

### Hardware
- **CPU:** Minimum 4 physical cores (8+ recommended for batch data loading).
- **RAM:** Minimum 16 GB system memory (32 GB recommended for whole-slide multi-patch processing).
- **GPU (Optional for Evaluation, Recommended for Full Training):**
  - NVIDIA GPU with $\ge 8$ GB VRAM (e.g., RTX 3080/4080, A100, V100, T4).
  - CUDA 11.8 or 12.1 supported.
- **Disk Space:** ~5 GB for code, checkpoints, and dependencies; ~15 GB if storing local Camelyon17 Parquet patches and MIDOG WSIs.

### Software
- **Operating System:** Ubuntu 20.04/22.04 LTS, Debian 11/12, macOS 12+, or Windows 10/11 (64-bit).
- **Python:** Python 3.10, 3.11, or 3.12 (Python 3.11 recommended).

---

## 2. Environment Installation

### Option A: Conda / Mamba (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/[ANONYMIZED]/SafeAdapt-TTA.git
cd SafeAdapt-TTA

# 2. Create environment from environment.yml
conda env create -f environment.yml

# 3. Activate environment
conda activate utta-med

# 4. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')"
```

### Option B: Python Virtual Environment (venv + pip)

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# For GPU PyTorch (Linux CUDA 11.8 / 12.1 example):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Verifying Installation

To ensure that all dependencies, numeric routines, and test modules are functioning properly without downloading external datasets:

```bash
# Run the automated test suite (145 tests)
pytest tests/ -q

# Run the automated 57-point numerical table consistency audit
python scripts/verify_manuscript_tables.py
```

Both commands should complete with **100% passing results**.

---

## 4. Next Steps

- Consult [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for instructions to regenerate every table and figure in the paper.
- Consult [docs/DATA_ACQUISITION.md](docs/DATA_ACQUISITION.md) for downloading and configuring external computational pathology benchmarks.
