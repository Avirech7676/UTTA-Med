# Stage E2: Mandatory Runtime Configuration Audit

**Auditor:** Senior PyTorch Research Engineer, Medical Imaging ML Scientist, and Independent Scientific Auditor  
**Date:** October 2026  
**Status:** Pre-Execution Invariants Verified & Locked  
**Target Domain:** Camelyon17-WILDS Hospital 2 (Center 2, Test WSIs `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29]`)  
**Runner:** `scripts/v2/evaluate_stage_e2.py`  
**Configuration File:** `configs/v2/safeadapt_e2_frozen.yaml`

---

## 1. Executive Summary

Prior to unsealing the Hospital-2 target evaluation data, all twelve mandatory runtime invariants and hard fail-closed code assertions were formally audited against `scripts/v2/evaluate_stage_e2.py` and `configs/v2/safeadapt_e2_frozen.yaml`.

This pre-execution audit verifies that the evaluation runner strictly adheres to the reconciled detector configuration and prespecified target sampling protocol, preventing any silent reversion to unmonitored defaults.

---

## 2. Twelve Mandatory Pre-Execution Invariant Checklist

| # | Runtime Invariant | Target Value / Specification | Verification Mechanism | Status |
| :---: | :--- | :--- | :--- | :---: |
| **01** | **Configuration Source** | `configs/v2/safeadapt_e2_frozen.yaml` | Explicit path resolution and YAML parsing; verified unedited. | **CONFIRMED** |
| **02** | **Effective Config Persistence** | Stored in JSON output artifact | `run_stage_e2_evaluation()` embeds full parsed YAML dictionary into `results/v2/stage_e2/stage_e2_target_results.json`. | **CONFIRMED** |
| **03** | **Monitor Window Size ($W$)** | `50` patches | Enforced by hard code assertion: `assert cfg['monitor']['window_size'] == 50`. | **CONFIRMED** |
| **04** | **Monitor Minimum Samples ($\min$)** | `20` patches | Enforced by hard code assertion: `assert cfg['monitor']['min_samples_for_alarm'] == 20`. | **CONFIRMED** |
| **05** | **Window Feasibility Relation** | $\min \le W$ ($20 \le 50$) | Enforced by assertion: `assert cfg['monitor']['min_samples_for_alarm'] <= cfg['monitor']['window_size']`. | **CONFIRMED** |
| **06** | **Sampling Budget ($N_{\text{patch}}$)** | `128` patches per WSI | Enforced by assertion: `assert cfg['execution']['patches_per_slide'] == 128`. Uniform spatial stride applied. | **CONFIRMED** |
| **07** | **Adaptation Batch Size ($B$)** | `32` patches per batch | Enforced by assertion: `assert cfg['adaptation']['batch_size'] == 32` (4 batches per WSI). | **CONFIRMED** |
| **08** | **Confidence Gating Threshold** | $\tau_{\text{conf}} = 0.90$ | Enforced by assertion: `assert cfg['gating']['tau_conf'] == 0.90`. Zero-selected bypass active. | **CONFIRMED** |
| **09** | **Prediction Timing** | `pre_adaptation` | Enforced by assertion: `assert cfg['execution']['prediction_timing'] == 'pre_adaptation'`. $p_{\text{eval}} = \sigma(f_{\theta_t}(x_t))$ before step. | **CONFIRMED** |
| **10** | **Boundary Reset Invariant** | `per_wsi_boundary` | Enforced by assertion: `assert cfg['execution']['reset_policy'] == 'per_wsi_boundary'`. Model ($\theta_0$), optimizer, and monitor reset every slide. | **CONFIRMED** |
| **11** | **Source Checkpoint SHA-256** | `f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000` | Real-time byte hash computation matching `checkpoints/camelyon17_resnet18_source_s42_BEST.pt`. | **CONFIRMED** |
| **12** | **Fail-Closed Execution Gate** | Immediate exception on mismatch | All assertions execute prior to opening Center-2 test data files; any discrepancy aborts evaluation immediately. | **CONFIRMED** |

---

## 3. Monitoring Opportunity Analysis on Target Pipeline

With 128 patches per slide and batch size 32:
- **Batch 1 ($t=1$, 32 patches):** Total accumulated patches in slide buffer = 32. Because $32 \ge 20$ ($\min$), the monitor is **eligible** and evaluates the multi-signal alarm policy. (Eligible Decision 1).
- **Batch 2 ($t=2$, 64 patches):** Total accumulated patches = 64. Sliding window uses the most recent $W=50$ patches. Eligible Decision 2.
- **Batch 3 ($t=3$, 96 patches):** Sliding window uses the most recent $W=50$ patches. Eligible Decision 3.
- **Batch 4 ($t=4$, 128 patches):** Sliding window uses the most recent $W=50$ patches. Eligible Decision 4.

Across 10 whole-slide images, the evaluation path provides **40 eligible monitoring decision points** (4 per slide), allowing consecutive escalation counters ($c \ge 2$ for rollback, $c \ge 4$ for fallback) to operate with complete mathematical validity within each slide.
