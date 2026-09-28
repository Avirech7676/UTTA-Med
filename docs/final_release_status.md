# UTTA-Med: Final Release Status & Research Freeze

## 1. Study Status: FROZEN & COMPLETE

All mandatory research components, scientific protocols, and statistical pipelines for **UTTA-Med** are frozen and fully reproduced across 5 random seeds (`{42, 123, 2024, 7, 99}`).

- **Model Backbone**: ResNet-18 (ImageNet-1K pretrained weights).
- **Classification Head**: 1-logit Binary Cross Entropy (BCE) with Sigmoid activation.
- **Regularization**: MC-Dropout (`p = 0.5`, `N_MC = 20` passes).
- **Source Training**: Hospitals `{0, 3, 4}` with Adam optimizer (`lr = 1e-4`, `weight_decay = 1e-4`), early stopping on Hospital 1 (OOD val) with patience = 5.
- **Authoritative Gate**: Predictive variance threshold $\tau$ calibrated at the 70th percentile of Hospital 1 predictive variance (`configs/tau_by_seed.yaml`).
- **Target Test Adaptation**: Unlabeled stream on Hospital 2 (`configs/camelyon17.yaml`), batch-norm affine updates only with running statistics frozen (`bn_freeze_stats`), continual adaptation (`lr = 1e-5`, steps = 1).

---

## 2. Definitive Results Summary (Hospital 2, 5 Seeds, Mean ± SD)

| Method | AUROC | F1 Score | Harm Rate | Coverage |
| :--- | :---: | :---: | :---: | :---: |
| **Source-Only (Unadapted)** | 0.930 ± 0.009 | 0.807 ± 0.025 | — | — |
| **Tent (Standard TTA)** | 0.698 ± 0.324 | 0.506 ± 0.411 | 0.187 ± 0.179 | 1.000 |
| **EATA (Entropy Filter)** | 0.921 ± 0.026 | 0.809 ± 0.031 | 0.025 ± 0.015 | 0.890 |
| **Random Gate (Matched)** | 0.697 ± 0.304 | 0.487 ± 0.385 | 0.194 ± 0.170 | 0.700 |
| **Confidence / Entropy** | **0.936 ± 0.022** | **0.831 ± 0.032** | **0.007 ± 0.003** | 0.580 |
| **UTTA-Med (Ours)** | 0.935 ± 0.022 | 0.830 ± 0.032 | 0.007 ± 0.003 | 0.580 |

### Key Scientific Findings
1. **Selective Adaptation**: UTTA-Med achieves near-zero harm rate ($< 0.01$) across all 5 seeds while improving F1 score on 5/5 seeds.
2. **Failure of Standard TTA**: Unconstrained Tent suffers catastrophic collapse on seeds 7 and 99 under clinical distribution shifts.
3. **Statistical Integrity**: Primary confidence intervals and hypothesis tests are evaluated at the slide (WSI) level ($N=10$ test slides) rather than treating individual patches as i.i.d. observations.

---

## 3. Repository Organization

```text
UTTA-Med/
├── README.md               # Overview, quickstart, headline metrics, MIT license
├── LICENSE                 # Official MIT License (Avinash Reddy Cheerapareddy)
├── requirements.txt        # Core reproduction dependencies
├── environment.yml         # Conda environment specification
├── configs/
│   ├── camelyon17.yaml     # Dataset center split definitions (Source: 0,3,4 | Val: 1 | Test: 2)
│   ├── tau_by_seed.yaml    # Authoritative frozen tau thresholds per seed
│   └── FINAL/              # Canonical model, TTA, and evaluation configs
├── src/                    # Tested modular source package (tta, models, data, training, evaluation)
├── scripts/                # Authoritative pipeline scripts (train_source.py, adapt.py, etc.)
├── checkpoints/            # Source checkpoints & inventory documentation
├── results/
│   ├── metrics/            # Canonical 5-seed full evaluation JSONs (s42, s123, s2024, s7, s99)
│   ├── statistical/        # WSI-level bootstrap distributions and paired statistics
│   ├── tables/             # Publication tables (LaTeX/CSV)
│   ├── figures/            # Architecture diagrams & tradeoff plots
│   └── ablations/          # MC-Dropout passes (N=5..50) and adaptation steps ablations
├── reproducibility/        # Checksum manifests, hardware specs, and git commit freeze
├── docs/                   # Authoritative protocol and audit documentation
├── notebooks/official/     # The 7 official Kaggle/Colab execution notebooks
├── paper/ieee/             # IEEE manuscript LaTeX sources, figures, and bibliography
└── archive/                # Complete historical research artifacts, planning notes, and legacy notebooks
```

---

## 4. External Data & Large Artifact Locations
- **Raw Parquet Shards**: Camelyon17-WILDS 21 shards hosted on HuggingFace Datasets.
- **Multi-Seed Checkpoint Weights & Predictions**: Kaggle Dataset `uttam-checkpoints` (SHA256 verified against `reproducibility/ARTIFACT_MANIFEST.csv`).
