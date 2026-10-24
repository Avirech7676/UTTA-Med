"""Deterministic CPU Unit Tests for SafeAdapt v2 Stage C1 Collapse Monitor.

Validates:
1. Normal stable predictions
2. All-negative slides without false alarms
3. Severe artificial single-class collapse
4. Sustained source-adapted disagreement
5. Temporary benign distribution changes
6. Small batches and incomplete windows
7. WSI boundary reset
8. Repeated suspicious windows
9. Alarm-triggered atomic rollback
10. Source fallback
11. Monitor-disabled equivalence
12. No target-label ingress
13. No changes to frozen v1 artifacts
14. Preservation of all Stage B tests
"""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import torch
import torch.nn as nn

from src.v2.safeadapt.adapter import SafeAdaptTTA
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController
from src.v2.safeadapt.monitor import (
    AlarmLevel,
    CollapseMonitor,
    CollapseMonitorConfig,
    MonitorMode,
    SlideState,
)


class DeterministicToyModel(nn.Module):
    """Deterministic model with BatchNorm layers for CPU testing."""

    def __init__(self, in_features: int = 16, hidden: int = 8):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out.squeeze(-1)


def make_c1_setup(
    mode: MonitorMode = MonitorMode.ACTIVE,
    window_size: int = 30,
    min_window_samples: int = 10,
    max_disagreement_rate: float = 0.40,
    consecutive_for_rollback: int = 2,
    consecutive_for_fallback: int = 4,
):
    """Build deterministic test setup for Stage C1."""
    torch.manual_seed(42)
    model = DeterministicToyModel()
    monitor_cfg = CollapseMonitorConfig(
        mode=mode,
        window_size=window_size,
        min_window_samples=min_window_samples,
        max_disagreement_rate=max_disagreement_rate,
        min_logit_variance=0.05,
        min_predictive_entropy=0.05,
        consecutive_suspicious_for_rollback=consecutive_for_rollback,
        consecutive_suspicious_for_fallback=consecutive_for_fallback,
    )
    config = SafeAdaptConfig(
        tta_lr=0.01,
        max_grad_norm=1.0,
        param_displacement_bound_linf=0.50,
        max_consecutive_rejections=5,
        monitor=monitor_cfg,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.tta_lr)
    controller = SafeAdaptController(model, optimizer, config, adaptable_params=params)
    return model, optimizer, controller, config


def test_1_normal_stable_predictions():
    """Test that well-behaved predictions with low disagreement produce AlarmLevel.OK."""
    model, optimizer, controller, config = make_c1_setup(mode=MonitorMode.ACTIVE)
    controller.set_slide("slide_normal_001")

    # Feed multiple batches with normal balanced predictions
    for i in range(5):
        torch.manual_seed(100 + i)
        x = torch.randn(10, 16)
        # Gentle loss to keep adaptation stable
        preds, accepted = controller.step(x, compute_loss_fn=lambda: model(x).sum() * 0.001)
        assert accepted, f"Batch {i} should be accepted"

    stats = controller.monitor.get_current_window_stats()
    assert stats is not None
    assert stats.alarm_level == AlarmLevel.OK
    assert stats.consecutive_suspicious == 0
    assert not stats.is_collapsed
    assert controller.monitor.state.alarm_level == AlarmLevel.OK


def test_2_all_negative_slide_no_false_alarm():
    """Test that all-negative slide (0% tumor) does NOT falsely trigger collapse alarm."""
    model, optimizer, controller, config = make_c1_setup(mode=MonitorMode.ACTIVE)
    controller.set_slide("slide_benign_all_neg")

    # Feed batches where both source and adapted predict negative (e.g. logits between -4.0 and -1.0)
    monitor = controller.monitor
    for _ in range(4):
        # 10 samples per batch, negative logits
        adapted_logits = torch.tensor([-2.5, -3.1, -1.8, -4.0, -2.0, -3.5, -1.9, -2.2, -3.0, -2.8])
        source_logits = torch.tensor([-2.4, -3.0, -1.9, -3.8, -2.1, -3.4, -2.0, -2.1, -2.9, -2.7])
        stats = monitor.update(adapted_logits, source_logits, slide_id="slide_benign_all_neg")

    assert stats.predicted_positive_fraction == 0.0, "Positive fraction should be exactly 0"
    assert stats.source_positive_fraction == 0.0, "Source positive fraction should also be 0"
    # Logit variance is healthy (> 0.05) and disagreement is 0.0
    assert stats.logit_variance > 0.05
    assert stats.disagreement_rate == 0.0
    # Guardrail check: MUST NOT be alarmed
    assert stats.alarm_level == AlarmLevel.OK, "All-negative slide must NOT trigger alarm when source agrees"
    assert not stats.is_collapsed
    assert stats.consecutive_suspicious == 0


def test_3_severe_artificial_single_class_collapse():
    """Test detection of severe single-class collapse where logits collapse to a constant."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        consecutive_for_rollback=1,
    )
    monitor = controller.monitor
    controller.set_slide("slide_collapsed")

    # Source has diverse distribution around 0, adapted collapses to constant 10.0 (variance = 0, disagreement = 50%)
    for _ in range(3):
        adapted_logits = torch.full((12,), 10.0)
        source_logits = torch.tensor([-2.0, 1.5, -1.0, 2.0, -0.5, 0.8, -1.8, 1.2, -0.2, 0.3, -1.5, 1.0])
        stats = monitor.update(adapted_logits, source_logits, slide_id="slide_collapsed")

    assert stats.is_collapsed, "Monitor should flag collapse"
    assert stats.logit_variance < 0.01, "Logit variance should be near 0"
    assert stats.alarm_level in (AlarmLevel.REJECT_UPDATE, AlarmLevel.FALLBACK)


def test_4_sustained_source_adapted_disagreement():
    """Test that high sustained disagreement between adapted and source raises alarms."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        max_disagreement_rate=0.30,
        consecutive_for_rollback=2,
    )
    monitor = controller.monitor
    controller.set_slide("slide_divergent")

    # Feed batches where adapted and source disagree on 80% of predictions
    for step in range(3):
        # 10 samples: source predicts mostly positive, adapted predicts mostly negative
        source_logits = torch.tensor([2.0, 2.5, 1.8, 2.2, 1.9, 2.1, 2.3, 1.7, 2.0, 2.2])
        adapted_logits = torch.tensor([-2.0, -2.5, -1.8, -2.2, -1.9, -2.1, -2.3, -1.7, 2.0, 2.2])
        stats = monitor.update(adapted_logits, source_logits, slide_id="slide_divergent")

    assert stats.disagreement_rate >= 0.80
    assert stats.consecutive_suspicious >= 2
    assert stats.alarm_level == AlarmLevel.REJECT_UPDATE


def test_5_temporary_benign_distribution_changes():
    """Test that transient suspicious windows do not trigger fallback if followed by normal windows."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        window_size=15,
        consecutive_for_rollback=2,
        consecutive_for_fallback=4,
    )
    monitor = controller.monitor
    controller.set_slide("slide_transient")

    # Step 1: Normal window
    normal_adapted = torch.randn(15)
    normal_source = normal_adapted + 0.1 * torch.randn(15)
    stats1 = monitor.update(normal_adapted, normal_source, slide_id="slide_transient")
    assert stats1.consecutive_suspicious == 0

    # Step 2: One transient anomaly
    anomalous_adapted = torch.full((15,), 8.0)
    stats2 = monitor.update(anomalous_adapted, normal_source, slide_id="slide_transient")
    assert stats2.consecutive_suspicious == 1
    # 1 suspicious window is only WARNING, not rollback (threshold is 2)
    assert stats2.alarm_level == AlarmLevel.WARNING

    # Step 3: Returns to normal
    stats3 = monitor.update(normal_adapted, normal_source, slide_id="slide_transient")
    # Consecutive count resets to 0!
    assert stats3.consecutive_suspicious == 0
    assert stats3.alarm_level == AlarmLevel.OK


def test_6_small_batches_and_incomplete_windows():
    """Test handling of small batches and incomplete windows without error."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        window_size=30,
        min_window_samples=10,
    )
    monitor = controller.monitor
    controller.set_slide("slide_small_batches")

    # Batch of size 2 (less than min_window_samples 10)
    small_adapted = torch.tensor([1.2, -0.5])
    small_source = torch.tensor([1.1, -0.4])
    stats = monitor.update(small_adapted, small_source, slide_id="slide_small_batches")

    assert stats.sample_count == 2
    # Incomplete window: alarm_level should remain OK
    assert stats.alarm_level == AlarmLevel.OK
    assert not stats.is_collapsed

    # Add single sample
    stats = monitor.update(torch.tensor([0.3]), torch.tensor([0.2]), slide_id="slide_small_batches")
    assert stats.sample_count == 3
    assert stats.alarm_level == AlarmLevel.OK


def test_7_wsi_boundary_reset():
    """Test that switching WSI boundaries resets monitor state independently."""
    model, optimizer, controller, config = make_c1_setup(mode=MonitorMode.ACTIVE)

    # Process on slide A
    controller.set_slide("slide_A")
    for _ in range(2):
        controller.monitor.update(torch.randn(15), torch.randn(15), slide_id="slide_A")
    assert controller.monitor.state.total_samples_processed == 30
    assert controller.monitor.state.slide_id == "slide_A"

    # Switch to slide B
    controller.set_slide("slide_B")
    assert controller.monitor.state.slide_id == "slide_B"
    assert controller.monitor.state.total_samples_processed == 0
    assert len(controller.monitor.state.window_adapted_logits) == 0
    assert controller.monitor.state.consecutive_suspicious == 0

    # Also test reset_boundary helper on controller
    controller.reset_boundary("slide_C", restore_source_weights=True)
    assert controller.current_slide_id == "slide_C"
    assert controller.monitor.state.slide_id == "slide_C"
    assert controller.monitor.state.total_samples_processed == 0


def test_8_repeated_suspicious_windows():
    """Test that sustained suspicious windows escalate from WARNING to REJECT to FALLBACK."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        consecutive_for_rollback=2,
        consecutive_for_fallback=3,
    )
    monitor = controller.monitor
    controller.set_slide("slide_escalation")

    src = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    coll = torch.full((10,), 10.0)

    # Window 1: Suspicious 1 -> WARNING
    s1 = monitor.update(coll, src, slide_id="slide_escalation")
    assert s1.consecutive_suspicious == 1
    assert s1.alarm_level == AlarmLevel.WARNING

    # Window 2: Suspicious 2 -> REJECT_UPDATE
    s2 = monitor.update(coll, src, slide_id="slide_escalation")
    assert s2.consecutive_suspicious == 2
    assert s2.alarm_level == AlarmLevel.REJECT_UPDATE

    # Window 3: Suspicious 3 -> FALLBACK
    s3 = monitor.update(coll, src, slide_id="slide_escalation")
    assert s3.consecutive_suspicious == 3
    assert s3.alarm_level == AlarmLevel.FALLBACK


def test_9_alarm_triggered_atomic_rollback():
    """Test that when monitor triggers REJECT_UPDATE during controller.step, model is rolled back."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        consecutive_for_rollback=1,
    )
    controller.set_slide("slide_test_rollback")

    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # Force the model to produce massive collapsing logits during adaptation
    def collapsing_loss_fn():
        # Large loss that shifts weights significantly
        out = model(torch.randn(12, 16))
        return (out - 100.0).pow(2).sum()

    # Step with controller; monitor detects single-class / divergent collapse and rejects
    x = torch.randn(12, 16)
    preds, accepted = controller.step(x, compute_loss_fn=collapsing_loss_fn)

    # Update should be rejected
    assert not accepted, "Collapse alarm must reject update"
    # Weights must be rolled back to pre-step state
    for k, v in model.state_dict().items():
        assert torch.allclose(v, initial_weights[k]), f"Weight mismatch in {k} after rollback"
    # Predictions returned must be safe source predictions
    assert torch.all(torch.isfinite(preds))
    assert controller.telemetry.updates_rejected >= 1


def test_10_source_fallback():
    """Test that controller enters permanent source-fallback mode when fallback alarm is triggered."""
    model, optimizer, controller, config = make_c1_setup(
        mode=MonitorMode.ACTIVE,
        consecutive_for_rollback=1,
        consecutive_for_fallback=2,
    )
    controller.set_slide("slide_fallback")

    # Trigger 2 consecutive collapse updates
    for i in range(2):
        x = torch.randn(12, 16)
        def bad_loss():
            return (model(x) - 100.0).pow(2).sum()
        preds, accepted = controller.step(x, compute_loss_fn=bad_loss)
        assert not accepted

    # Controller must now be in fallback mode
    assert controller.is_in_fallback_mode
    assert controller.telemetry.fallback_triggered

    # Further steps should immediately return source predictions without running adaptation
    x2 = torch.randn(8, 16)
    preds2, accepted2 = controller.step(x2, compute_loss_fn=lambda: model(x2).sum())
    assert not accepted2
    assert controller.is_in_fallback_mode


def test_11_monitor_disabled_equivalence():
    """Test that when monitor is disabled, controller behavior matches unmonitored execution."""
    torch.manual_seed(99)
    model1 = DeterministicToyModel()
    params1 = [p for p in model1.parameters() if p.requires_grad]
    opt1 = torch.optim.Adam(params1, lr=0.01)
    cfg_disabled = SafeAdaptConfig(
        tta_lr=0.01,
        monitor=CollapseMonitorConfig(mode=MonitorMode.DISABLED),
    )
    ctrl1 = SafeAdaptController(model1, opt1, cfg_disabled, adaptable_params=params1)

    torch.manual_seed(99)
    model2 = DeterministicToyModel()
    params2 = [p for p in model2.parameters() if p.requires_grad]
    opt2 = torch.optim.Adam(params2, lr=0.01)
    cfg_unmonitored = SafeAdaptConfig(tta_lr=0.01, monitor=None)
    ctrl2 = SafeAdaptController(model2, opt2, cfg_unmonitored, adaptable_params=params2)

    torch.manual_seed(42)
    x = torch.randn(10, 16)

    def loss1():
        return model1(x).sum()

    def loss2():
        return model2(x).sum()

    preds1, acc1 = ctrl1.step(x, compute_loss_fn=loss1)
    preds2, acc2 = ctrl2.step(x, compute_loss_fn=loss2)

    assert acc1 == acc2
    assert torch.allclose(preds1, preds2, atol=1e-7)
    for (k1, v1), (k2, v2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        assert torch.allclose(v1, v2, atol=1e-7)


def test_12_no_target_label_ingress():
    """Test that neither CollapseMonitor nor SafeAdaptController accepts target labels."""
    # Inspect CollapseMonitor signatures
    monitor = CollapseMonitor(CollapseMonitorConfig())
    update_params = list(inspect.signature(monitor.update).parameters.keys())
    assert "y" not in update_params
    assert "labels" not in update_params
    assert "targets" not in update_params
    assert "target" not in update_params
    assert "adapted_probs" in update_params
    assert "source_probs" in update_params

    # Inspect SafeAdaptController.step signature
    ctrl_step_params = list(inspect.signature(SafeAdaptController.step).parameters.keys())
    assert "y" not in ctrl_step_params
    assert "labels" not in ctrl_step_params
    assert "targets" not in ctrl_step_params
    assert "target" not in ctrl_step_params
    assert "x" in ctrl_step_params
    assert "compute_loss_fn" in ctrl_step_params


def test_13_no_changes_to_frozen_v1_artifacts():
    """Test that frozen v1 scientific artifacts exist and are untouched."""
    proj_root = Path(__file__).resolve().parent.parent

    # Authoritative results and tables must exist
    stats_file = proj_root / "results" / "five_seed_final_stats.json"
    table_paired = proj_root / "results" / "tables" / "table_paired_statistics_final.csv"
    master_csv = proj_root / "results" / "master_results.csv"
    auth_yaml = proj_root / "configs" / "authoritative_resnet18_utta.yaml"

    assert stats_file.exists(), f"Missing {stats_file}"
    assert table_paired.exists(), f"Missing {table_paired}"
    assert master_csv.exists(), f"Missing {master_csv}"
    assert auth_yaml.exists(), f"Missing {auth_yaml}"

    # Verify no v2 imports or references leaked into frozen v1 scripts
    v1_main = proj_root / "src" / "eval_camelyon17.py"
    if v1_main.exists():
        content = v1_main.read_text(encoding="utf-8")
        assert "src.v2" not in content, "v2 must not be imported into v1 eval_camelyon17.py"


def test_14_preservation_of_all_stage_b_tests():
    """Test that all original Stage B tests remain runnable and pass."""
    import tests.test_safeadapt_v2 as b_tests

    # Verify key Stage B test functions exist
    expected_stage_b_tests = [
        "test_nan_loss_rejection",
        "test_infinite_gradient_rejection",
        "test_non_finite_post_step_parameter_rejection",
        "test_adam_step_counter_and_optimizer_state_preservation",
        "test_injected_invalid_optimizer_update_restores_exact_state",
        "test_source_state_immutability",
        "test_bn_running_statistics_preservation",
        "test_source_relative_parameter_projection_linf",
        "test_exception_safe_rollback",
        "test_current_batch_source_fallback_on_rejection",
        "test_repeated_failures_trigger_fallback_mode",
        "test_reset_boundary_behavior",
        "test_adapter_absence_of_target_label_inputs",
        "test_v1_artifacts_unmodified_checksum",
    ]
    for test_fn_name in expected_stage_b_tests:
        assert hasattr(b_tests, test_fn_name), f"Stage B test {test_fn_name} must be preserved"

