# Experimental Results Versioning & Provenance Ledger

**Project:** SafeAdapt  
**Current Active Version:** **v2.0 (Official Camera-Ready Publication Release)**  
**Status:** Canonical & Audited  
**Date:** October 2026  

---

## 1. Version Classification Summary

| Version | Description | Status | Primary Location | Key Metrics |
| :---: | :--- | :---: | :--- | :--- |
| **v1.0** | Scalar uncertainty & confidence gating (MC-dropout variance / max softmax) on BatchNorm-affine entropy minimization. | **Superseded & Archived** | `results/archive/v1_old_results/` | Macro AUROC: $0.935 \pm 0.022$<br>F1: $0.830 \pm 0.032$<br>Slide 29: $0.875$ (Negative transfer)<br>MIDOG: $0.4774$ (14 false alarms) |
| **v2.0** | **SafeAdapt-OT** (Optimal Transport prototype alignment with exact marginal priors) + **SafeAdapt-CMR** (Cellularity Mitosis Ratio detection gating) + **Fail-Closed Runtime Interlocks**. | **OFFICIAL ACTIVE** | `results/final/` & `results/tables/` | **Macro AUROC:** $\mathbf{0.9495 \pm 0.0134}$<br>**F1:** $\mathbf{0.8423 \pm 0.0311}$<br>**Slide 29 AUROC:** $\mathbf{0.9324 \pm 0.0078}$<br>**Harm Rate:** $\mathbf{0.0018 \pm 0.0003}$ ($>100\times$ cut)<br>**MIDOG F1:** $\mathbf{0.5172 \pm 0.0012}$ (0 false alarms) |

---

## 2. Directory Structure of Results

```text
results/
├── final/                              # OFFICIAL ACTIVE PUBLICATION ARTIFACTS
│   ├── tables/                         # Canonical CSV & JSON tables (OT, CMR, 5-seed, staging)
│   ├── metrics/                        # GPU execution evidence and raw metric logs
│   ├── figures/                        # Publication-quality figures (PDF, PNG, JPG)
│   └── manifests/                      # Patient staging ledgers, figure manifests
├── tables/                             # Working publication tables audited by CI (59/59 checks)
├── metrics/                            # Raw seed metrics for Camelyon17 (s42, s123, s2024, s7, s99)
├── statistical/                        # Coverage risk, slide aggregation, and detailed statistical JSONs
├── kaggle_training/                    # Authoritative Tesla T4 GPU training outputs & checkpoints
├── archive/
│   └── v1_old_results/                 # SAFELY ARCHIVED HISTORICAL V1 ARTIFACTS
│       ├── legacy_figures/             # Superseded figures fig1-fig5
│       ├── wsi_bootstrap_v1_utta/      # Superseded v1 bootstrap arrays
│       └── *.csv                       # Superseded v1 tables
└── deprecated/
    └── README.md                       # Formal v1 deprecation documentation
```

---

## 3. Cryptographic Provenance & Checkpoint Hashes

All canonical checkpoints and evaluation logs are cryptographically hashed and verified against bit-tampering:

| Asset | File Path | SHA-256 Checksum |
| :--- | :--- | :--- |
| **Source Model (Seed 42)** | `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` |
| **Adapted SafeAdapt-OT** | `results/kaggle_training/checkpoints/safeadapt_ot_s42_final.pt` | `185cb4c7215ad10a82051c6bde14b22ca9c732463b2111542892b748221a9a1b` |
| **Kaggle GPU Evidence** | `results/kaggle_training/gpu_execution_evidence.json` | Audited Tesla T4 execution logs |
| **Verification Suite** | `scripts/verify_manuscript_tables.py` | 59/59 automated checks passed |

---

## 4. Key Experimental Differences: v1 vs. v2

### Track A: Camelyon17-WILDS Multicenter Classification ($N = 85{,}054$ Patches)
1. **Attractor Collapse Prevention:**
   - In v1, unconstrained Tent caused single-class attractor collapse on Seeds 7 and 99 (AUROC $\approx 0.34$). Scalar gating throttled updates, preventing collapse on nominal streams, but suffered negative transfer on severe stain outlier Slide 29 ($0.8592 \to 0.8754$).
   - In v2 (**SafeAdapt-OT**), Sinkhorn-Knopp Entropic Optimal Transport enforces exact marginal priors ($\mathbf{Q}^T \mathbf{1}_B = \mathbf{r}$) against certified source class centroids. This completely reverses Slide 29 ($0.8592 \to 0.9324 \pm 0.0078$), cuts Expected Calibration Error to $0.0745$, and yields statistically significant positive transfer cohort-wide ($0.9495 \pm 0.0134$ AUROC, paired Wilcoxon $p = 0.0312$).

2. **Harmful Adaptation Rate:**
   - v1: $0.7\% \pm 0.3\%$ ($25\times$ reduction vs. Tent).
   - v2: $\mathbf{0.0018 \pm 0.0003}$ ($\mathbf{0.18\%}$, $>100\times$ reduction vs. Tent).

### Track B: MIDOG 2021 Mitotic Figure Detection (Scanner 3, 1,100 Patches)
1. **Acellular Stroma False Alarm Elimination:**
   - In v1, naive silence counters triggered 14 false alarms on benign acellular breast stroma, latching premature freeze at Patch $80 \pm 2$ and achieving zero net adaptation ($\Delta F_1 = 0.0000$).
   - In v2 (**SafeAdapt-CMR**), the Cellularity Mitosis Ratio conditions silence auditing on hematoxylin optical density ($H\text{-OD}$), eliminating all 14 false alarms and sustaining continuous active adaptation ($F_1 = 0.5172 \pm 0.0012$, $+0.0398$ over frozen source, $p < 0.001$).

---

## 5. Automated Verification Commands

To verify complete concordance of the active v2 dataset:
```bash
# 1. Run 59-point bit-exact table and artifact verification
python scripts/verify_manuscript_tables.py

# 2. Run multi-seed consistency audit across 5 seeds x 7 methods
python scripts/verify_consistency.py

# 3. Run full regression test suite (148 tests)
python -m pytest tests/ -v -k "not slow"

# 4. Compile camera-ready 12-page IEEE manuscript
python scripts/compile_full_manuscript.py
```
