"""SafeAdapt v2 Configuration Specification.

Stage B & C1: SafeAdapt configuration with decoupled numerical validation,
independent gradient clipping, parameter bounding, and slide-aware collapse monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.v2.safeadapt.monitor import CollapseMonitorConfig, MonitorMode


@dataclass
class SafeAdaptConfig:
    """Configuration for SafeAdapt test-time adaptation controller.

    Scientific Notice:
    - Default hyperparameters (e.g. param_displacement_bound_linf = 0.20,
      min_logit_variance = 0.05) are candidate working hypotheses for development
      and MUST be calibrated on validation data (Hospital 1).
    - They are NOT scientifically validated universal constants.
    """

    # Update scope
    bn_affine_only: bool = True
    freeze_running_stats: bool = True

    # Learning rate (authoritative Camelyon17 TTA learning rate)
    tta_lr: float = 1e-5

    # Finite-value and numerical validation
    check_finite_loss: bool = True
    check_finite_gradients: bool = True
    check_finite_parameters: bool = True
    check_finite_predictions: bool = True

    # Safeguards: independently configurable
    max_grad_norm: Optional[float] = 1.0  # Optional L2 gradient clipping norm
    param_displacement_bound_linf: Optional[float] = 0.20  # Optional source-relative L_inf bound (epsilon)

    # Rollback & Fallback policies
    max_consecutive_rejections: int = 5
    fallback_on_max_rejections: bool = True
    fallback_on_unhandled_exception: bool = True

    # Reset boundaries
    reset_on_boundary: bool = False  # Reset parameters to source at episode/WSI boundary

    # Stage C1: Collapse Monitor Configuration
    monitor: CollapseMonitorConfig = field(default_factory=CollapseMonitorConfig)

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.tta_lr <= 0:
            raise ValueError(f"tta_lr must be positive, got {self.tta_lr}")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.param_displacement_bound_linf is not None and self.param_displacement_bound_linf <= 0:
            raise ValueError(
                f"param_displacement_bound_linf must be positive, got {self.param_displacement_bound_linf}"
            )
        if self.max_consecutive_rejections < 1:
            raise ValueError(
                f"max_consecutive_rejections must be >= 1, got {self.max_consecutive_rejections}"
            )
        if self.monitor is not None and self.monitor.window_size < 10:
            raise ValueError(f"monitor.window_size must be >= 10, got {self.monitor.window_size}")
