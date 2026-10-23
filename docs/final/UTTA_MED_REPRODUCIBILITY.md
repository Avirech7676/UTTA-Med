# UTTA-Med: Complete Reproducibility Package and Verification Manifest

**Audit Standard:** Full Scientific Reproducibility and Computational Provenance  
**Date:** October 2026  
**Auditor:** Independent Scientific Auditor & Reproducibility Specialist  
**Test Suite Status:** 120 Passed, 2 Skipped, 0 Failed (100% Core Passing Rate)

---

## 1. System and Hardware Environment Specifications

All results, benchmarks, and regression suites reported in the UTTA-Med v2 consolidation were verified under the following standardized software and hardware environment:

- **Operating System:** Windows 11 Enterprise / Ubuntu 22.04 LTS Compatible
- **Python Version:** Python 3.10+ / 3.11.x / 3.14.x (verified on conda env `avinash` with Python 3.14.2)
- **PyTorch Core:** PyTorch 2.x (`torch >= 2.0.0`, verified on PyTorch 2.14.0+cpu)
- **Computer Vision:** `torchvision >= 0.15.0`
- **Data & Numerical Science:** `numpy >= 1.24.0`, `pandas >= 2.0.0`, `scipy >= 1.10.0`, `scikit-learn >= 1.2.0`
- **Benchmarking Benchmark:** WILDS Benchmark (`wilds >= 2.0.0`)
- **Virtual Environment:** Dedicated isolated environment (e.g. `conda activate avinash` or `.venv`)

---

## 2. Machine-Readable Artifact Manifest and Checksums

To verify bit-exact provenance, the SHA-256 cryptographic hashes of all canonical data, configuration, checkpoint, and result artifacts are tabulated below:

| Relative File Path | SHA-256 Checksum | Description |
|:---|:---|:---|
| `checkpoints/camelyon17_resnet18_source_s42_BEST.pt` | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` | Canonical frozen source model checkpoint (ResNet-18, Seed 42) |
| `configs/v2/safeadapt_e2_frozen.yaml` | `3042225de8be5bd013a5725898a787802714004fb2b3ee95d61ba14e18df1418` | Frozen runtime configuration executed for Stage E2 evaluation |
| `results/tables/headline_5seed.json` | `f3893c5d6e246e7fcae8c56fa68c5b969e46a788876c1f8a8cf38c114ce9bf14` | v1 multi-seed raw evaluation dictionary |
| `results/tables/table_main_results_final.csv` | `320df9cd65e84be39a4b3bbb5f4408a47d5c932bbee037dab9f25006a66f2243` | v1 multi-seed main results table |
| `results/tables/table_paired_statistics_final.csv` | `b7510afe90e26e6e21c0f9ce87fcb1f0415ee358a60671f5ba5b37eb1d11f2f6` | v1 paired statistical analysis with Holm-Bonferroni correction |
| `results/v2/stage_d2/stage_d2_benchmark_results.json` | `143ca46038968393b160ed8a7bf5c013e0c845d30c9e95b3367be33fecb02b7c` | Stage D2 uncertainty gating natural & matched budget benchmarks |
| `results/v2/stage_e0/stage_e0_reconciliation.json` | `0da8f056d65d5012655c7605b79ec44708c67655c9a756030df5ceb99df6984c` | Stage E0 bit-exact reconciliation of Brier metric and patch IDs |
| `results/v2/stage_e1/stage_e1_evaluation_results.json` | `ab060f1ec9e9d725e667272e801488739d54a153f6ddaa75dfaebae611a726fc` | Stage E1 Hospital-1 4-arm streaming evaluation |
| `results/v2/stage_e2/stage_e2_target_results.json` | `f5c90277991999ecc6ae5a63fa3532d74fb5a14b417207d4071efd1533ac1ff4` | Stage E2 Hospital-2 4-arm target evaluation |

---

## 3. Step-by-Step Reproduction Guide

### Step 1: Environment Setup
Clone the repository and instantiate the Python virtual environment:

```bash
git clone https://github.com/Avirech7676/UTTA-Med.git
cd UTTA-Med
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy scikit-learn scipy wilds pyyaml pytest
```

### Step 2: Automated Verification of Test Suite
Run the comprehensive regression and safeguard test suite to verify numerical handlers, rollback mechanics, and gating logic:

```bash
pytest tests/ -v
```

Expected result:
```
====================== 120 passed, 2 skipped, 7 warnings in 11.2s =======================
```

### Step 3: Verification of Checksums and Artifact Integrity
Run the cryptographic integrity auditor and table verifier (verifies 54/54 numerical checks against stored JSON/CSV artifacts):

```bash
python scripts/verify_manuscript_tables.py
```

This verifies that the canonical checkpoint SHA-256 matches `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`, verifies artifact hashes, and checks all values in Tables 1 through 5.

### Step 4: Regeneration of Authoritative Tables (Tables 1 through 8)
To regenerate all 8 authoritative tables directly from machine-readable JSON/CSV artifacts without manual transcription:

```bash
python scripts/generate_all_final_tables.py
```

This outputs bit-exact markdown tables for:
1. Historical v1 multi-seed performance.
2. Historical v1 paired statistical significance.
3. Stage D2 gating policy comparisons.
4. Stage E1 Hospital-1 4-arm evaluation.
5. Stage E2 Hospital-2 4-arm target evaluation.
6. Per-WSI target performance on Hospital 2 (WSIs 20–29).
7. Safeguard intervention counts and parameter displacements.
8. Runtime latencies and computational speedup benchmarks.

---

## 4. Frozen Evaluation Command (Reference Protocol)

The frozen Stage E2 target evaluation was executed via the following command:

```bash
python scripts/v2/evaluate_stage_e2.py --config configs/v2/safeadapt_e2_frozen.yaml
```

Under this configuration:
- Model: ResNet-18 (`checkpoints/camelyon17_resnet18_source_s42_BEST.pt`)
- Target Domain: Camelyon17-WILDS Center 2 (WSIs 20–29)
- Sampling Budget: 128 patches per slide ($N = 1,280$ patches)
- Batch Size: 32 (4 batches per slide)
- Optimizer: SGD ($\eta = 10^{-4}, \mu = 0.9$)
- Parameter Displacement Bound: $L_\infty \le 0.05$
- Confidence Gating Threshold: $\tau = 0.90$
- Collapse Monitor Window: $W = 50$, Minimum Observations: $\min = 20$
- Prediction Timing: `pre_adaptation`
- Reset Boundary: `per_wsi_boundary`

---

## 5. Automated Regression Test Suite Mapping

The unit test suite validates every critical algorithmic invariant of the system:

| Test File | Key Test Cases | Architectural Verification |
|:---|:---|:---|
| `tests/test_safeadapt_v2.py` | `test_atomic_rollback`, `test_parameter_bound_clamping`, `test_non_finite_rejection` | Proves deterministic recovery from numerical faults and enforces $L_\infty \le 0.05$. |
| `tests/test_safeadapt_c1_monitor.py` | `test_monitor_warning_escalation`, `test_wsi_boundary_reset`, `test_min_samples_threshold` | Proves collapse monitor behavior, thresholding, and cross-slide state reset. |
| `tests/test_gating_v2.py` | `test_confidence_gating`, `test_mc_dropout_variance`, `test_hybrid_gating` | Proves bit-exact selection semantics and Spearman rank correlation consistency. |
| `tests/test_no_leakage.py` | `test_zero_target_labels_in_tta`, `test_immutable_source_model` | Proves no label leakage enters adaptation or monitoring routines. |
| `tests/test_coverage_risk.py` | `test_sweeps_coverage_and_adaptation` | Proves monotonic risk curves under selective adaptation sweeps. |
