# Final Project Reproducibility & Open Science Report

**Project:** UTTA-Med Project 2  
**Reproducibility Standard:** 14-Point Computational Pathology Verification  

---

## 1. Verified Reproducibility Package

1. **Source Code Integrity:** All core algorithms (`src/models/`, `scripts/`) version-controlled under git branch `research-project-2`.
2. **Dataset Manifests:** Canonical manifests for Camelyon17 Center 2 ($85,054$ patches) and MIDOG Scanner 3 (118, 131, 117 mitoses) verified.
3. **Deterministic Seeds:** Fixed random seeds (42, 123, 2024, 7, 99) controlling PyTorch, NumPy, and CUDA RNG.
4. **Automated Reproduction Script:** [`scripts/build_reproducibility_package.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA_MED_V1_FINAL/UTTA_MED_V1_FINAL/scripts/build_reproducibility_package.py) automates end-to-end evaluation.
