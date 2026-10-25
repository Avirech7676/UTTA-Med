# Reproducibility Guide: SafeAdapt-TTA

**Audit Status:** 57 / 57 Automated Checks Passed (100% Bit-Exact Match)  
**Test Suite:** 145 Passed, 2 Skipped, 0 Failed (`pytest tests/`)  
**Random Seeds Audited:** 5 Seeds (42, 7, 123, 2024, 99)

This document provides exact instructions to reproduce all numerical results, statistical tables, and publication figures from the SafeAdapt manuscript.

---

## 1. Quick Verification (Offline Pre-Flight Audit)

No dataset downloads or GPU hardware are required to run the primary integrity auditor. It validates all published metrics against cryptographic hashes and frozen machine-readable outputs:

```bash
python scripts/verify_manuscript_tables.py
```

**Expected Console Output:**
```
================================================================================
STAGE F.1: AUTOMATED NUMERICAL CONSISTENCY VERIFICATION AUDIT
================================================================================
[headline_5seed] results/tables/headline_5seed.json -> SHA256: 9f0f0753835c...
[table_main] results/tables/table_main_results_final.csv -> SHA256: 320df9cd65...
[table_paired] results/tables/table_paired_statistics_final.csv -> SHA256: b7510afe...
[stage_d2] results/v2/stage_d2/stage_d2_benchmark_results.json -> SHA256: 143ca460...
[stage_e0] results/v2/stage_e0/stage_e0_reconciliation.json -> SHA256: 0da8f056...
[stage_e1] results/v2/stage_e1/stage_e1_evaluation_results.json -> SHA256: ab060f1e...
[stage_e2] results/v2/stage_e2/stage_e2_target_results.json -> SHA256: f5c90277...
[config_e2] configs/v2/safeadapt_e2_frozen.yaml -> SHA256: 3042225d...
[checkpoint] checkpoints/camelyon17_resnet18_source_s42_BEST.pt -> SHA256: f0b85ec1...

--- Verifying Table 1 (v1 Main Results) ---
--- Verifying Table 2 (v1 Paired Statistics) ---
--- Verifying Table 3 (Stage D2 Gating) ---
--- Verifying Table 4 (Stage E1 Hospital 1) ---
--- Verifying Table 5 (Stage E2 Hospital 2) ---
--- Verifying Theorem 1 Head Lipschitz Regularity ---

Total verification checks performed: 57
Passed: 57 / 57
Failed: 0 / 57
```

---

## 2. Table-by-Table Reproduction Commands

| Manuscript Table | Script Execution Command | Primary Output File |
|:---|:---|:---|
| **Table 1: 5-Seed Primary Benchmark** | `python scripts/five_seed_final_stats.py` | `results/tables/table_main_results_final.csv` |
| **Table 2: Paired Statistics & Harm** | `python scripts/generate_all_final_tables.py` | `results/tables/table_paired_statistics_final.csv` |
| **Table 3: Stage D2 Gating Semantics** | `python scripts/v2/evaluate_stage_d2.py` | `results/v2/stage_d2/stage_d2_benchmark_results.json` |
| **Table 4: Hospital 1 Calibration** | `python scripts/v2/evaluate_stage_e1.py` | `results/v2/stage_e1/stage_e1_evaluation_results.json` |
| **Table 5: Hospital 2 Target Arms** | `python scripts/v2/evaluate_stage_e2.py` | `results/v2/stage_e2/stage_e2_target_results.json` |
| **Table 6: MIDOG Scanner 3 (CMR)** | `python scripts/run_day6_cellularity_detection.py` | `results/tables/cmr_full_1100_summary.csv` |
| **Table 7: NCT-CRC Evidential Logic** | `python scripts/run_evidential_nct_benchmark.py` | `results/tables/nctcrc_evidential_results.csv` |
| **Table 8: Slide 29 OT-TTA Reversal** | `python scripts/run_ot_tta_camelyon17.py` | `results/tables/ot_tta_camelyon17_results.csv` |
| **pTNM Whole-Slide Staging** | `python scripts/eval_wsi_ptnm_staging.py` | `results/tables/wsi_staging_results.csv` |

---

## 3. Figure-by-Figure Reproduction Commands

| Manuscript Figure | Generation Command | Output Location |
|:---|:---|:---|
| **Figure 2: Coverage-Risk Tradeoff** | `python scripts/coverage_risk_curves.py` | `figures/fig_coverage_risk_tradeoff.png` |
| **Figure 3: Slide 29 Collapse Forensics**| `python scripts/run_tent_collapse_analysis.py` | `figures/fig_d6_slide29_seed7_hist.png` |
| **Figure 4: Multi-Seed Macro AUROC** | `python scripts/five_seed_final_stats.py` | `figures/fig_main_bars.png` |
| **Figure 5: Parameter Drift Trajectories**| `python scripts/v2/evaluate_stage_e2.py` | `figures/fig_d6_tent_collapse_slides.png` |

---

## 4. Cryptographic Provenance Checksums

All official source model weights are checksummed. Verify local checkpoints against frozen hashes:

```bash
python -c "
import hashlib, glob
for p in sorted(glob.glob('checkpoints/*.pt')):
    with open(p, 'rb') as f:
        print(f'{p:55s} {hashlib.sha256(f.read()).hexdigest()}')
"
```

Expected primary source checkpoint hash:
- `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`: `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000`
