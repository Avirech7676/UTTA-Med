"""Detection Safety Monitor for Dense Object Detection Under Test-Time Adaptation.

Provides label-free, fail-closed runtime monitoring for dense mitotic figure detection
under test-time adaptation (MIDOG 2021 / Faster R-CNN).

Key Clinical & Domain Innovations:
1. Histopathological Cellularity Pre-Filter: In digital pathology, whole-slide images
   frequently contain extensive regions of acellular stroma, adipose tissue, or normal
   connective architecture where mitotic figures are biologically absent.
   The monitor explicitly decouples physiological absence of mitoses from detector collapse.
2. Concordant Negative Invariance: When both the frozen source detector and the adapted
   detector observe 0 mitoses on a patch (n_source == 0, n_pred == 0), the silence streak
   is NOT advanced, preventing premature latching and fallback on benign tissue.
3. Proliferation Safeguards: Bounded absolute and relative proliferation detection prevents
   pathological parameter mutation that produces spurious dense false positives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DetectionSafetyMonitorConfig:
    """Configuration parameters for detection safety monitoring."""
    silence_window: int = 30
    max_per_patch: int = 4
    source_ratio: float = 3.0
    min_cellularity_threshold: float = 0.15
    max_collapse_streak: int = 5


def compute_cmr(n_mitoses: int, cellularity: float, epsilon: float = 1e-5) -> float:
    """Compute the Cellularity Mitosis Ratio (CMR).
    
    CMR(x) = n_mitoses(x) / max(rho_cell(x), epsilon)
    """
    return float(n_mitoses) / max(float(cellularity), epsilon)


class DetectionSafetyMonitor:
    """Runtime safety monitor for dense object detection TTA with CMR and local rollback."""

    def __init__(
        self,
        silence_window: int = 30,
        max_per_patch: int = 4,
        source_ratio: float = 3.0,
        min_cellularity_threshold: float = 0.15,
        max_collapse_streak: int = 5,
    ) -> None:
        self.silence_window = silence_window
        self.max_per_patch = max_per_patch
        self.source_ratio = source_ratio
        self.min_cellularity_threshold = min_cellularity_threshold
        self.max_collapse_streak = max_collapse_streak

        self.empty_streak: int = 0
        self.consecutive_rejections: int = 0
        self.total_patches_checked: int = 0
        self.total_alarms: int = 0
        self.alarms_silence: int = 0
        self.alarms_prolif_abs: int = 0
        self.alarms_prolif_rel: int = 0
        self.benign_acellular_skips: int = 0
        self.local_rollbacks: int = 0
        self.latching_fallbacks: int = 0
        self.is_latched: bool = False

    def reset(self) -> None:
        """Reset monitoring counters."""
        self.empty_streak = 0
        self.consecutive_rejections = 0
        self.is_latched = False

    def compute_cmr(self, n_mitoses: int, cellularity: float, epsilon: float = 1e-5) -> float:
        """Helper method to compute CMR."""
        return compute_cmr(n_mitoses, cellularity, epsilon)

    def check_patch(
        self,
        n_pred: int,
        n_source: Optional[int] = None,
        is_acellular: bool = False,
        tissue_cellularity: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Evaluate patch detection count against safety invariants.

        Conditioning criteria for silence:
        The silence counter increments if and only if:
        (i) n_pred == 0,
        (ii) nuclear area fraction rho_cell(x) >= tau_nuc (default 0.15, hypercellular tumor),
        (iii) source detector predicted mitosis (n_source > 0).

        Args:
            n_pred: Number of detections predicted by adapted model above threshold tau.
            n_source: Optional number of detections predicted by frozen source model.
            is_acellular: Boolean flag indicating if tissue patch is confirmed acellular stroma.
            tissue_cellularity: Optional estimated cellular area fraction in [0.0, 1.0].

        Returns:
            Tuple of (is_alarm: bool, alarm_type: str).
            alarm_type is one of: 'NORMAL', 'SILENCE_ALARM', 'SILENCE_LATCH_COLLAPSE',
            'PROLIFERATION_ABS_ALARM', 'PROLIFERATION_REL_ALARM'.
        """
        self.total_patches_checked += 1

        # Evaluate silence streak
        if n_pred == 0:
            # Check if this patch is clinically expected to have zero mitoses
            is_benign_negative = is_acellular or (
                tissue_cellularity is not None
                and tissue_cellularity < self.min_cellularity_threshold
            ) or (n_source is not None and n_source == 0)

            if is_benign_negative:
                # Concordant absence of mitoses or acellular stroma: do not increment silence streak
                self.benign_acellular_skips += 1
            else:
                # Tissue is cellular or source expects mitoses, yet adapted predicts 0:
                self.empty_streak += 1
                if self.empty_streak >= self.silence_window:
                    self.consecutive_rejections += 1
                    self.total_alarms += 1
                    self.alarms_silence += 1
                    self.local_rollbacks += 1
                    self.empty_streak = 0

                    if self.consecutive_rejections >= self.max_collapse_streak:
                        self.latching_fallbacks += 1
                        self.is_latched = True
                        return True, "SILENCE_LATCH_COLLAPSE"

                    return True, "SILENCE_ALARM"
        else:
            self.empty_streak = 0

        # Absolute proliferation check
        if n_pred > self.max_per_patch:
            self.consecutive_rejections += 1
            self.total_alarms += 1
            self.alarms_prolif_abs += 1
            self.local_rollbacks += 1
            return True, "PROLIFERATION_ABS_ALARM"

        # Relative proliferation check (relative to frozen source baseline)
        if n_source is not None and n_source > 0:
            if n_pred > (self.source_ratio * n_source):
                self.consecutive_rejections += 1
                self.total_alarms += 1
                self.alarms_prolif_rel += 1
                self.local_rollbacks += 1
                return True, "PROLIFERATION_REL_ALARM"

        # If we reached here with n_pred > 0 and no proliferation alarm, reset consecutive rejections
        if n_pred > 0:
            self.consecutive_rejections = 0

        return False, "NORMAL"

