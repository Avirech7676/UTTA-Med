"""Unit tests for DetectionSafetyMonitor with cellularity & concordance awareness."""
import pytest
from src.v2.safeadapt.detection_monitor import (
    DetectionSafetyMonitor,
    DetectionSafetyMonitorConfig,
)


def test_normal_detection_flow():
    monitor = DetectionSafetyMonitor(silence_window=10, max_per_patch=4, source_ratio=3.0)
    for _ in range(5):
        alarm, alarm_type = monitor.check_patch(n_pred=2, n_source=2)
        assert not alarm
        assert alarm_type == "NORMAL"
        assert monitor.empty_streak == 0
        assert monitor.consecutive_rejections == 0


def test_absolute_proliferation_alarm():
    monitor = DetectionSafetyMonitor(silence_window=10, max_per_patch=4, source_ratio=3.0)
    # 5 > 4 -> Absolute Proliferation
    alarm, alarm_type = monitor.check_patch(n_pred=5, n_source=2)
    assert alarm
    assert alarm_type == "PROLIFERATION_ABS_ALARM"
    assert monitor.consecutive_rejections == 1
    assert monitor.alarms_prolif_abs == 1


def test_relative_proliferation_alarm():
    monitor = DetectionSafetyMonitor(silence_window=10, max_per_patch=4, source_ratio=3.0)
    # 4 <= 4 (not abs), but 4 > 3.0 * 1 -> Relative Proliferation
    alarm, alarm_type = monitor.check_patch(n_pred=4, n_source=1)
    assert alarm
    assert alarm_type == "PROLIFERATION_REL_ALARM"
    assert monitor.consecutive_rejections == 1
    assert monitor.alarms_prolif_rel == 1


def test_acellular_stroma_concordance_invariance():
    """Verify that acellular stroma or concordant zero mitoses does NOT trigger silence alarms."""
    monitor = DetectionSafetyMonitor(silence_window=10, max_per_patch=4, source_ratio=3.0)

    # 15 consecutive patches with 0 detections on acellular stroma (or concordant n_source == 0)
    for _ in range(15):
        alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=0, is_acellular=True)
        assert not alarm
        assert alarm_type == "NORMAL"

    # Invariance check: empty streak must remain 0
    assert monitor.empty_streak == 0
    assert monitor.alarms_silence == 0
    assert monitor.benign_acellular_skips == 15


def test_true_silence_alarm_on_cellular_tissue():
    """Verify that silence alarms correctly trigger when source expects mitoses or tissue is cellular."""
    monitor = DetectionSafetyMonitor(silence_window=10, max_per_patch=4, source_ratio=3.0)

    # 9 consecutive empty patches where source expected 1 mitosis
    for i in range(9):
        alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, is_acellular=False)
        assert not alarm
        assert monitor.empty_streak == i + 1

    # 10th patch reaches silence window -> SILENCE ALARM
    alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, is_acellular=False)
    assert alarm
    assert alarm_type == "SILENCE_ALARM"
    assert monitor.alarms_silence == 1
    assert monitor.consecutive_rejections == 1
    assert monitor.local_rollbacks == 1
    assert not monitor.is_latched


def test_cmr_calculation():
    """Verify Cellularity Mitosis Ratio (CMR) calculation."""
    from src.v2.safeadapt.detection_monitor import compute_cmr
    # Zero mitoses
    assert compute_cmr(0, 0.20) == 0.0
    # 2 mitoses on 0.25 cellularity -> 8.0
    assert abs(compute_cmr(2, 0.25) - 8.0) < 1e-5
    # Near-zero cellularity clamped by epsilon
    assert compute_cmr(1, 0.0, epsilon=1e-3) == 1000.0


def test_local_rollback_and_latching_fallback():
    """Verify that transient silence triggers local rollback, and >= 5 collapse events latch global fallback."""
    monitor = DetectionSafetyMonitor(silence_window=2, max_collapse_streak=5)
    
    # 4 consecutive rejections trigger local rollbacks without latching
    for streak in range(4):
        alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, tissue_cellularity=0.20)
        assert not alarm
        alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, tissue_cellularity=0.20)
        assert alarm
        assert alarm_type == "SILENCE_ALARM"
        assert not monitor.is_latched
        assert monitor.local_rollbacks == streak + 1

    # 5th consecutive rejection trips SILENCE_LATCH_COLLAPSE
    alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, tissue_cellularity=0.20)
    assert not alarm
    alarm, alarm_type = monitor.check_patch(n_pred=0, n_source=1, tissue_cellularity=0.20)
    assert alarm
    assert alarm_type == "SILENCE_LATCH_COLLAPSE"
    assert monitor.is_latched
    assert monitor.latching_fallbacks == 1

