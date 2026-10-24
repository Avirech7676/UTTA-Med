# Stage C1: Collapse-Aware Controller Implementation Report

**Document Version:** 1.0-Implementation  
**Date:** October 10, 2026  
**Roles:** Senior PyTorch Engineer, Medical Imaging Research Scientist, Independent Reproducibility Auditor  
**Status:** COMPLETE (All CPU Deterministic Acceptance Tests Passing)  
**Strict Boundaries:** Zero modification of frozen v1 checkpoints, tables, raw metrics, or manuscript results. Zero GPU training, Kaggle runs, or Hospital-2 evaluation.

---

## 1. Executive Summary

Stage C1 extends the existing Stage B `SafeAdaptController` with label-free, slide-aware prediction-collapse monitoring. Prediction collapse—the catastrophic phenomenon wherein test-time adaptation models drive all predictions into a single degenerate class or reduce logit variance to near zero (as forensically audited in Phase 1 for ResNet-50 and Tent seeds 7/99)—is monitored online using rolling-window measurements without requiring any target ground-truth labels.

Crucially, Stage C1 enforces essential digital pathology guardrails: **a Whole-Slide Image (WSI) is never classified as collapsed solely because its predicted-positive fraction is near zero**, because benign lymph nodes legitimately contain 0% tumor patches. Multi-signal persistence is required before triggering atomic rollback or permanent source fallback.

All 14 deterministic CPU acceptance tests in `tests/test_safeadapt_c1_monitor.py` pass with **100% success (14/14)**, while preserving all 14 Stage B tests (14/14) and the complete regression suite (85 passed, 2 skipped).

---

## 2. Inventory of File Changes

### 2.1 Created or Extended Modules (Isolated `src/v2/` Namespace)
1. [`src/v2/safeadapt/monitor.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/monitor.py)
   - Core implementation of `CollapseMonitor`, `SlideState`, `MonitorWindowStats`, `AlarmLevel`, and `MonitorMode`.
   - Streaming rolling-window buffer with numerically stable statistics:
     - Predicted-positive fraction ($\hat{y} \ge \tau$)
     - Binary predictive entropy ($-\sum p \log p$)
     - Logit mean and empirical variance ($\mathbb{E}[z], \mathrm{Var}(z)$)
     - Mean continuous and binary source-versus-adapted disagreement ($\frac{1}{N}\sum |p - p_0|, \frac{1}{N}\sum \mathbb{I}(\hat{y} \neq \hat{y}_0)$)
     - 10-bin probability histogram over $[0.0, 1.0]$
     - Consecutive suspicious window counter for multi-signal persistence.
2. [`src/v2/safeadapt/config.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/config.py)
   - Extended `SafeAdaptConfig` with nested `monitor: CollapseMonitorConfig`.
   - Supports observation mode (`MonitorMode.OBSERVATION`), active enforcement (`MonitorMode.ACTIVE`), and disabled mode (`MonitorMode.DISABLED`).
   - Clean parameter validation with backward compatibility.
3. [`src/v2/safeadapt/controller.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/controller.py)
   - Integrated `CollapseMonitor` into the atomic step pipeline.
   - Post-step inference evaluates collapse rules:
     - `AlarmLevel.WARNING`: Logs structured telemetry event; allows update.
     - `AlarmLevel.REJECT_UPDATE`: Triggers atomic rollback of model parameters and Adam state; recomputes and emits safe frozen-source predictions.
     - `AlarmLevel.FALLBACK`: Permanently locks controller into frozen-source fallback; reloads immutable source weights.
   - Slide-aware API: `set_slide(slide_id)` and `reset_boundary(slide_id, restore_source_weights)`.
4. [`src/v2/safeadapt/telemetry.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/telemetry.py)
   - Added `fallback_triggered` property and structured alarm event logging (`ALARM_WARNING`, `ROLLBACK`, `RESET`).
5. [`src/v2/safeadapt/__init__.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/src/v2/safeadapt/__init__.py)
   - Exported `CollapseMonitor`, `CollapseMonitorConfig`, `AlarmLevel`, `MonitorMode`, `SlideState`, and `MonitorWindowStats`.
6. [`configs/v2/safeadapt_c1_monitor.yaml`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/configs/v2/safeadapt_c1_monitor.yaml)
   - Canonical Stage C1 configuration file defaulting to `OBSERVATION` mode until calibrated on Hospital-1 validation data.
7. [`tests/test_safeadapt_c1_monitor.py`](file:///c:/Users/avina/OneDrive/Desktop/UTTA-Med/tests/test_safeadapt_c1_monitor.py)
   - 14 deterministic CPU verification tests covering all failure and boundary scenarios.

### 2.2 Preservation of Frozen Scientific Artifacts
- **Checkpoints:** Unmodified. All source ResNet-18 checkpoints retain identical MD5 checksums.
- **Tables and Raw Metrics:** Unmodified. `results/tables/table_paired_statistics_final.csv`, `results/five_seed_final_stats.json`, and `results/master_results.csv` remain intact.
- **v1 Source Scripts:** Unmodified. No references to `src.v2` leak into frozen v1 evaluation or training code.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Rolling Window Statistics
For a Whole-Slide Image $S$, let the sliding window of the most recent $W$ predictions be $\{p_i\}_{i=1}^W$, with corresponding source predictions $\{p_{0, i}\}_{i=1}^W$ and adapted logits $\{z_i\}_{i=1}^W$.

1. **Predicted-Positive Fraction:**
   $$\hat{F}_+ = \frac{1}{W} \sum_{i=1}^W \mathbb{I}(p_i \ge \tau)$$
2. **Mean Predictive Entropy:**
   $$\bar{H} = -\frac{1}{W} \sum_{i=1}^W \left[ p_i \ln(p_i + \epsilon) + (1 - p_i) \ln(1 - p_i + \epsilon) \right]$$
3. **Logit Mean & Variance:**
   $$\bar{z} = \frac{1}{W} \sum_{i=1}^W z_i, \quad \sigma_z^2 = \frac{1}{W - 1} \sum_{i=1}^W (z_i - \bar{z})^2$$
4. **Source Disagreement:**
   $$\Delta_{\mathrm{cont}} = \frac{1}{W} \sum_{i=1}^W |p_i - p_{0, i}|, \quad \Delta_{\mathrm{bin}} = \frac{1}{W} \sum_{i=1}^W \mathbb{I}\left( (p_i \ge \tau) \neq (p_{0, i} \ge \tau) \right)$$

### 3.2 Pathology Class-Imbalance Guardrail
In histopathological lymph node metastasis detection (Camelyon17), normal tissue slides contain **0% tumor patches**. A model predicting 0% positive patches is displaying accurate clinical discernment, not mode collapse.

Therefore, the monitor enforces the fundamental invariant:
$$\hat{F}_+ \approx 0 \implies \mathbf{NOT\ ALARM}$$
An alarm on low positive fraction is permitted **only** if coupled with:
1. Severe source disagreement: source model predicted significant tumor presence ($F_{0, +} > 0.20$), but adapted model suppressed all detections ($\Delta_{\mathrm{bin}} > \Delta_{\max}$); AND
2. Logit variance collapse ($\sigma_z^2 < \sigma_{\min}^2$).

### 3.3 Escalation Ladder
The alarm level escalates strictly through multi-window persistence:
- **Normal (`AlarmLevel.OK` / `NORMAL`):** No anomaly detected.
- **Warning (`AlarmLevel.WARNING`):** Transient anomaly ($C_{\mathrm{susp}} < C_{\mathrm{rollback}}$). Logged to telemetry; adaptation update proceeds.
- **Update Rejection (`AlarmLevel.REJECT_UPDATE`):** Sustained anomaly ($C_{\mathrm{susp}} \ge C_{\mathrm{rollback}}$). Current gradient update is discarded; weights and Adam moments rolled back to pre-step state; source predictions emitted.
- **Fallback (`AlarmLevel.FALLBACK`):** Persistent anomaly ($C_{\mathrm{susp}} \ge C_{\mathrm{fallback}}$). Controller permanently disables adaptation; model state permanently restored to frozen source weights.

---

## 4. Alarm Limitations and Remaining Risks

1. **Uncalibrated Thresholds:**
   Current thresholds (`max_disagreement_rate = 0.40`, `min_logit_variance = 0.05`, `consecutive_for_rollback = 2`) are provisional working hypotheses. They must be calibrated exclusively on Hospital-1 out-of-distribution validation data before deployment.
2. **Observation-Only Default:**
   To prevent premature adaptation rejections during development, `configs/v2/safeadapt_c1_monitor.yaml` sets `mode: "observation"`.
3. **Target Isolation Guarantee:**
   No target labels ($y$) are ingested or accessible anywhere within the monitor or controller pipeline.
4. **Remaining Scientific Risk:**
   While the monitor successfully detects single-class collapse and catastrophic divergence, subtle drift that maintains artificial variance without collapsing may evade detection without task-specific calibration.
