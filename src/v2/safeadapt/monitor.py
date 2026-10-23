"""SafeAdapt v2 Prediction-Collapse and Slide-Aware Monitoring Module.

Stage C1: Implements rolling-window metrics for predicted-positive fraction,
entropy, logit mean/variance, source-adapted disagreement, probability histograms,
and slide-aware boundary resets without using target ground truth labels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AlarmLevel(Enum):
    """Alarm severity levels for collapse monitoring."""

    NORMAL = "NORMAL"
    OK = "NORMAL"  # Alias for NORMAL
    WARNING = "WARNING"
    REJECT_UPDATE = "REJECT_UPDATE"
    ROLLBACK = "ROLLBACK"
    FALLBACK = "FALLBACK"


class MonitorMode(Enum):
    """Operating modes for collapse monitor."""

    DISABLED = "DISABLED"        # Completely inactive, zero computation
    OBSERVATION = "OBSERVATION"  # Compute and log telemetry, no interventions
    ACTIVE = "ACTIVE"            # Actively enforce alarm policy (rejection / fallback)


@dataclass
class MonitorWindowStats:
    """Statistics computed over a rolling evaluation window."""

    window_size: int
    predicted_positive_fraction: float
    mean_predictive_entropy: float
    logit_mean: float
    logit_variance: float
    mean_source_disagreement: float
    binary_disagreement_rate: float
    probability_histogram: List[int] = field(default_factory=list)  # 10 bins: [0, 0.1), ...
    is_sufficient_samples: bool = True
    source_positive_fraction: float = 0.0
    consecutive_suspicious: int = 0
    alarm_level: AlarmLevel = AlarmLevel.NORMAL
    is_collapsed: bool = False
    disagreement_rate: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert window stats to dictionary."""
        return {
            "window_size": self.window_size,
            "sample_count": self.sample_count,
            "predicted_positive_fraction": round(self.predicted_positive_fraction, 6),
            "source_positive_fraction": round(self.source_positive_fraction, 6),
            "mean_predictive_entropy": round(self.mean_predictive_entropy, 6),
            "logit_mean": round(self.logit_mean, 6),
            "logit_variance": round(self.logit_variance, 6),
            "mean_source_disagreement": round(self.mean_source_disagreement, 6),
            "binary_disagreement_rate": round(self.binary_disagreement_rate, 6),
            "disagreement_rate": round(self.disagreement_rate, 6),
            "probability_histogram": list(self.probability_histogram),
            "is_sufficient_samples": self.is_sufficient_samples,
            "consecutive_suspicious": self.consecutive_suspicious,
            "alarm_level": self.alarm_level.value,
            "is_collapsed": self.is_collapsed,
        }


@dataclass
class CollapseMonitorConfig:
    """Configuration for slide-aware prediction-collapse monitor."""

    # Operational mode
    mode: MonitorMode = MonitorMode.OBSERVATION

    # Rolling window size (in samples)
    window_size: int = 500
    min_samples_for_alarm: int = 100
    min_window_samples: Optional[int] = None

    # Decision threshold for binary classification
    decision_threshold: float = 0.5

    # Candidate alarm thresholds (Uncalibrated: observation-only by default)
    max_mean_source_disagreement: float = 0.40
    max_binary_disagreement_rate: float = 0.35
    max_disagreement_rate: Optional[float] = None

    # Logit & entropy thresholds
    min_logit_variance: float = 0.05
    min_predictive_entropy: float = 0.05

    # Persistence / escalation counts
    min_consecutive_suspicious_windows: int = 3
    consecutive_suspicious_for_rollback: int = 2
    consecutive_suspicious_for_fallback: int = 4

    # Slide-aware policies
    slide_reset_policy: str = "RESET_MONITOR_ONLY"
    slide_aware_reset: bool = True
    reset_weights_on_slide: bool = False

    def __post_init__(self) -> None:
        """Synchronize optional alias attributes."""
        if self.min_window_samples is not None:
            self.min_samples_for_alarm = self.min_window_samples
        else:
            self.min_window_samples = self.min_samples_for_alarm

        if self.max_disagreement_rate is not None:
            self.max_mean_source_disagreement = self.max_disagreement_rate
            self.max_binary_disagreement_rate = self.max_disagreement_rate
        else:
            self.max_disagreement_rate = self.max_mean_source_disagreement


class SlideState:
    """Tracks state and sliding buffers for an individual Whole-Slide Image."""

    def __init__(self, slide_id: str, max_buffer: int = 1000):
        self.slide_id = slide_id
        self.max_buffer = max_buffer
        self.reset()

    def reset(self) -> None:
        """Reset sliding buffers for this slide."""
        self.probs: deque = deque(maxlen=self.max_buffer)
        self.source_probs: deque = deque(maxlen=self.max_buffer)
        self.logits: deque = deque(maxlen=self.max_buffer)
        self.consecutive_suspicious_windows: int = 0
        self.total_patches_seen: int = 0
        self.total_batches_seen: int = 0
        self.alarms_tripped: int = 0
        self.alarm_level: AlarmLevel = AlarmLevel.NORMAL

    @property
    def total_samples_processed(self) -> int:
        """Alias for total_patches_seen."""
        return self.total_patches_seen

    @property
    def consecutive_suspicious(self) -> int:
        """Alias for consecutive_suspicious_windows."""
        return self.consecutive_suspicious_windows

    @property
    def window_adapted_logits(self) -> deque:
        """Alias for logits buffer."""
        return self.logits

    def append_batch(
        self,
        probs: np.ndarray,
        source_probs: np.ndarray,
        logits: np.ndarray,
    ) -> None:
        """Append batch predictions to slide buffer."""
        n = len(probs)
        self.total_patches_seen += n
        self.total_batches_seen += 1
        for p, sp, z in zip(probs, source_probs, logits):
            self.probs.append(float(p))
            self.source_probs.append(float(sp))
            self.logits.append(float(z))


class UpdateResult:
    """Wrapper that can be unpacked as (alarm_level, stats, reason) or accessed as stats object."""

    def __init__(
        self,
        alarm_level: AlarmLevel,
        stats: MonitorWindowStats,
        reason: str,
    ):
        self.alarm_level = alarm_level
        self.stats = stats
        self.reason = reason

    def __iter__(self) -> Iterator[Any]:
        """Allow unpacking: alarm_level, stats, reason = update_result."""
        yield self.alarm_level
        yield self.stats
        yield self.reason

    def __getitem__(self, idx: int) -> Any:
        """Support index-based access."""
        items = (self.alarm_level, self.stats, self.reason)
        return items[idx]

    def __len__(self) -> int:
        return 3

    def __getattr__(self, name: str) -> Any:
        """Forward any attribute lookup to the underlying MonitorWindowStats."""
        if hasattr(self.stats, name):
            return getattr(self.stats, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class CollapseMonitor:
    """Label-free, slide-aware prediction-collapse monitor."""

    def __init__(self, config: Optional[CollapseMonitorConfig] = None):
        self.config = config or CollapseMonitorConfig()
        self.current_slide_id: str = "default_slide"
        self.slides: Dict[str, SlideState] = {
            self.current_slide_id: SlideState(
                self.current_slide_id, max_buffer=self.config.window_size * 2
            )
        }
        self.last_window_stats: Optional[MonitorWindowStats] = None

    @property
    def current_slide(self) -> SlideState:
        """Return state for active slide."""
        if self.current_slide_id not in self.slides:
            self.slides[self.current_slide_id] = SlideState(
                self.current_slide_id, max_buffer=self.config.window_size * 2
            )
        return self.slides[self.current_slide_id]

    @property
    def state(self) -> SlideState:
        """Alias for active slide state."""
        return self.current_slide

    def set_slide(self, slide_id: str) -> None:
        """Switch to a new Whole-Slide Image, isolating monitoring state."""
        self.current_slide_id = str(slide_id)
        if self.current_slide_id not in self.slides:
            self.slides[self.current_slide_id] = SlideState(
                self.current_slide_id, max_buffer=self.config.window_size * 2
            )

    def reset_slide(self, slide_id: Optional[str] = None) -> None:
        """Explicitly reset state for a given slide or active slide."""
        sid = str(slide_id) if slide_id is not None else self.current_slide_id
        if sid in self.slides:
            self.slides[sid].reset()

    def get_current_window_stats(self) -> Optional[MonitorWindowStats]:
        """Return the most recently computed window statistics."""
        return self.last_window_stats

    def update(
        self,
        adapted_probs: torch.Tensor,
        source_probs: torch.Tensor,
        adapted_logits: Optional[torch.Tensor] = None,
        slide_id: Optional[str] = None,
    ) -> UpdateResult:
        """Update monitor with new batch and evaluate collapse policy.

        Notice: Accepts ONLY predictions and logits. No target labels 'y' are accepted.

        Returns:
            UpdateResult unpackable as (AlarmLevel, MonitorWindowStats, reason_string)
            or directly queryable for stats attributes.
        """
        if slide_id is not None and slide_id != self.current_slide_id:
            self.set_slide(slide_id)

        # Detach and convert to CPU numpy
        p_tensor = adapted_probs.detach().cpu().view(-1)
        sp_tensor = source_probs.detach().cpu().view(-1)

        # Detect if input is logits rather than probabilities (values outside [0, 1] or negative)
        if (p_tensor < 0.0).any() or (p_tensor > 1.0).any():
            z_np = p_tensor.numpy().astype(np.float64)
            p_np = torch.sigmoid(p_tensor).numpy().astype(np.float64)
        else:
            p_np = p_tensor.numpy().astype(np.float64)
            if adapted_logits is not None:
                z_np = adapted_logits.detach().cpu().view(-1).numpy().astype(np.float64)
            else:
                p_clamped = np.clip(p_np, 1e-7, 1.0 - 1e-7)
                z_np = np.log(p_clamped / (1.0 - p_clamped))

        if (sp_tensor < 0.0).any() or (sp_tensor > 1.0).any():
            sp_np = torch.sigmoid(sp_tensor).numpy().astype(np.float64)
        else:
            sp_np = sp_tensor.numpy().astype(np.float64)

        slide = self.current_slide
        slide.append_batch(p_np, sp_np, z_np)

        if self.config.mode == MonitorMode.DISABLED:
            stats = self._compute_window_stats(slide)
            self.last_window_stats = stats
            slide.alarm_level = AlarmLevel.NORMAL
            return UpdateResult(AlarmLevel.NORMAL, stats, "monitor_disabled")

        stats = self._compute_window_stats(slide)

        if not stats.is_sufficient_samples:
            stats.alarm_level = AlarmLevel.NORMAL
            self.last_window_stats = stats
            slide.alarm_level = AlarmLevel.NORMAL
            return UpdateResult(AlarmLevel.NORMAL, stats, "insufficient_window_samples")

        alarm_level, reason = self._evaluate_alarm_policy(stats, slide)

        if alarm_level != AlarmLevel.NORMAL and alarm_level != AlarmLevel.OK:
            slide.consecutive_suspicious_windows += 1
            slide.alarms_tripped += 1
        else:
            slide.consecutive_suspicious_windows = 0

        # Update stats metadata
        stats.consecutive_suspicious = slide.consecutive_suspicious_windows
        stats.alarm_level = alarm_level
        stats.is_collapsed = alarm_level in (AlarmLevel.REJECT_UPDATE, AlarmLevel.FALLBACK)
        slide.alarm_level = alarm_level
        self.last_window_stats = stats

        # In observation mode, never trigger active interventions
        if self.config.mode == MonitorMode.OBSERVATION:
            if alarm_level in (AlarmLevel.REJECT_UPDATE, AlarmLevel.FALLBACK):
                return UpdateResult(AlarmLevel.WARNING, stats, f"observed_{reason}")
            return UpdateResult(alarm_level, stats, f"observed_{reason}")

        return UpdateResult(alarm_level, stats, reason)

    def _compute_window_stats(self, slide: SlideState) -> MonitorWindowStats:
        """Compute rolling-window statistics using available samples."""
        n_avail = len(slide.probs)
        w = min(n_avail, self.config.window_size)
        if w < 1:
            return MonitorWindowStats(
                window_size=0,
                predicted_positive_fraction=0.0,
                mean_predictive_entropy=0.0,
                logit_mean=0.0,
                logit_variance=0.0,
                mean_source_disagreement=0.0,
                binary_disagreement_rate=0.0,
                is_sufficient_samples=False,
                sample_count=0,
            )

        # Slice the most recent w items
        p_slice = np.array(list(slide.probs)[-w:], dtype=np.float64)
        sp_slice = np.array(list(slide.source_probs)[-w:], dtype=np.float64)
        z_slice = np.array(list(slide.logits)[-w:], dtype=np.float64)

        thresh = self.config.decision_threshold
        pos_frac = float((p_slice >= thresh).mean())
        src_pos_frac = float((sp_slice >= thresh).mean())

        # 2. Binary entropy: -p log p - (1-p) log (1-p)
        p_c = np.clip(p_slice, 1e-7, 1.0 - 1e-7)
        entropy = -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))
        mean_entropy = float(entropy.mean())

        # 3. Logit mean and variance
        logit_mean = float(z_slice.mean())
        logit_var = float(z_slice.var()) if len(z_slice) > 1 else 0.0

        # 4. Source disagreement
        abs_diff = np.abs(p_slice - sp_slice)
        mean_disagreement = float(abs_diff.mean())
        bin_diff = (p_slice >= thresh) != (sp_slice >= thresh)
        bin_disagreement = float(bin_diff.mean())

        # 5. 10-bin probability histogram
        hist, _ = np.histogram(p_slice, bins=10, range=(0.0, 1.0))
        hist_list = [int(h) for h in hist]

        is_sufficient = w >= self.config.min_samples_for_alarm

        return MonitorWindowStats(
            window_size=w,
            sample_count=w,
            predicted_positive_fraction=pos_frac,
            source_positive_fraction=src_pos_frac,
            mean_predictive_entropy=mean_entropy,
            logit_mean=logit_mean,
            logit_variance=logit_var,
            mean_source_disagreement=mean_disagreement,
            binary_disagreement_rate=bin_disagreement,
            disagreement_rate=bin_disagreement,
            probability_histogram=hist_list,
            is_sufficient_samples=is_sufficient,
            consecutive_suspicious=slide.consecutive_suspicious_windows,
        )

    def _evaluate_alarm_policy(
        self, stats: MonitorWindowStats, slide: SlideState
    ) -> Tuple[AlarmLevel, str]:
        """Evaluate multi-signal alarm rules.

        Key Pathology Guardrail:
        - In digital pathology, a normal lymph node legitimately contains 0% tumor patches.
        - Therefore, predicted_positive_fraction near zero is NEVER an alarm on its own.
        - It is ONLY considered an alarm indicator if BOTH:
          1. The source model disagreed significantly (source predicted tumors, but adapted crushed them);
          2. Logit variance has collapsed (indicating constant flattened outputs).
        """
        suspicious_signals: List[str] = []

        # Signal 1: Severe source-adapted disagreement
        if stats.mean_source_disagreement > self.config.max_mean_source_disagreement:
            suspicious_signals.append("high_continuous_disagreement")
        if stats.binary_disagreement_rate > self.config.max_binary_disagreement_rate:
            suspicious_signals.append("high_binary_decision_flip")

        # Signal 2: Logit variance collapse
        if stats.logit_variance < self.config.min_logit_variance:
            suspicious_signals.append("logit_variance_collapse")

        # Signal 3: Predictive entropy collapse
        if stats.mean_predictive_entropy < self.config.min_predictive_entropy:
            # Low entropy with extreme single class
            if stats.predicted_positive_fraction >= 0.99 or (
                stats.predicted_positive_fraction <= 0.01 and stats.source_positive_fraction > 0.20
            ):
                suspicious_signals.append("predictive_entropy_collapse")

        # If no suspicious signals, state is normal
        if not suspicious_signals:
            return AlarmLevel.NORMAL, "normal"

        # Signal combination logic:
        # Require multiple concurrent signals OR sustained persistence
        is_multi_signal = len(suspicious_signals) >= 2
        consec = slide.consecutive_suspicious_windows + 1
        is_rollback_persistent = consec >= self.config.consecutive_suspicious_for_rollback
        is_fallback_persistent = consec >= self.config.consecutive_suspicious_for_fallback

        reason_str = "+".join(suspicious_signals)

        if is_fallback_persistent:
            return AlarmLevel.FALLBACK, f"sustained_collapse_fallback:{reason_str}"
        elif is_rollback_persistent:
            return AlarmLevel.REJECT_UPDATE, f"collapse_alarm_rollback:{reason_str}"
        else:
            return AlarmLevel.WARNING, f"isolated_warning:{reason_str}"
