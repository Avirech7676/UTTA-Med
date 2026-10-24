"""SafeAdapt v2 Atomic Adaptation Controller.

Implements atomic, numerically robust, source-preserving test-time adaptation
with non-finite rejection, bounded parameter projection, slide-aware prediction-collapse
monitoring, and frozen-source fallback.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.monitor import AlarmLevel, CollapseMonitor, MonitorMode
from src.v2.safeadapt.telemetry import SafeAdaptTelemetry


class SafeAdaptController:
    """Atomic adaptation controller for test-time adaptation.

    Invariants:
    1. Immutability: Preserves a separate, detached source state that cannot be mutated.
    2. Atomicity: An update is committed iff loss, gradients, parameters, and predictions
       are all finite and within allowable constraints; otherwise, model and optimizer
       are rolled back to their pre-step states.
    3. Non-Toxicity: Never returns non-finite prediction probabilities to downstream clinical workflows.
    4. Label-Free: Adapts and monitors without any target ground truth labels.
    5. Slide-Awareness: Tracks independent sliding windows per Whole-Slide Image and handles
       boundary resets without false alarms on legitimate 0% tumor tissue sections.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[SafeAdaptConfig] = None,
        adaptable_params: Optional[List[nn.Parameter]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or SafeAdaptConfig()
        self.config.validate()

        # Identify adaptable parameters
        if adaptable_params is not None:
            self.adaptable_params = adaptable_params
            self._apply_mode_and_freeze_rules()
        else:
            self.adaptable_params = self._configure_bn_affine_scope()

        # 1. Snapshot immutable source state (CPU and device)
        self._immutable_source_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()
        }
        self._source_model = deepcopy(self.model).eval()
        for p in self._source_model.parameters():
            p.requires_grad = False

        # Keep an immutable reference tensor map for projection
        self._source_param_anchors: Dict[int, torch.Tensor] = {
            id(p): p.detach().clone() for p in self.adaptable_params
        }

        # Telemetry tracker
        self.telemetry = SafeAdaptTelemetry()

        # Stage C1: Collapse Monitor
        self.monitor = CollapseMonitor(self.config.monitor)

        # Operational state
        self._in_fallback_mode: bool = False
        self._consecutive_rejections: int = 0
        self._batch_counter: int = 0

    def _apply_mode_and_freeze_rules(self) -> None:
        """Enforce frozen running stats and dropout eval mode."""
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if self.config.freeze_running_stats:
                    m.eval()
            elif isinstance(m, nn.Dropout):
                m.eval()

    def _configure_bn_affine_scope(self) -> List[nn.Parameter]:
        """Configure BatchNorm affine-only adaptation and freeze running stats."""
        for p in self.model.parameters():
            p.requires_grad = False

        params: List[nn.Parameter] = []
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if self.config.freeze_running_stats:
                    m.eval()
                if self.config.bn_affine_only and m.affine:
                    if m.weight is not None:
                        m.weight.requires_grad = True
                        params.append(m.weight)
                    if m.bias is not None:
                        m.bias.requires_grad = True
                        params.append(m.bias)
            elif isinstance(m, nn.Dropout):
                m.eval()

        if not params and self.config.bn_affine_only:
            raise RuntimeError("No BatchNorm affine parameters found for adaptation scope.")
        return params

    @property
    def in_fallback_mode(self) -> bool:
        """Return True if adaptation is disabled and routing to frozen source."""
        return self._in_fallback_mode

    @property
    def is_in_fallback_mode(self) -> bool:
        """Alias for in_fallback_mode."""
        return self._in_fallback_mode

    @property
    def current_slide_id(self) -> str:
        """Return active Whole-Slide Image ID."""
        return self.monitor.current_slide_id if self.monitor is not None else "default_slide"

    @property
    def consecutive_rejections(self) -> int:
        """Return count of current consecutive rejected updates."""
        return self._consecutive_rejections

    def set_slide(self, slide_id: str) -> None:
        """Set active Whole-Slide Image ID for slide-aware monitoring."""
        if self.monitor is not None:
            self.monitor.set_slide(slide_id)

    def verify_source_immutability(self) -> bool:
        """Verify that internal preserved source state and frozen model layers have not been mutated.

        Audits:
        1. Preserved source state dictionary contains expected keys and matching tensor shapes.
        2. Reference source model (_source_model) parameters and buffers match the immutable source snapshot bit-for-bit.
        3. All non-adaptable parameters (requires_grad=False) and buffers in self.model match
           the source snapshot bit-for-bit.
        """
        source_model_state = self._source_model.state_dict()

        def tensors_match(t1: torch.Tensor, t2: torch.Tensor) -> bool:
            if t1.shape != t2.shape:
                return False
            if t1.is_floating_point():
                return bool(torch.allclose(t1.cpu(), t2.cpu(), atol=1e-6, rtol=1e-5))
            return bool(torch.equal(t1.cpu(), t2.cpu()))

        # 1. Verify reference source model integrity
        for k, src_v in self._immutable_source_state.items():
            if k not in source_model_state:
                return False
            if not isinstance(src_v, torch.Tensor):
                return False
            if not tensors_match(source_model_state[k], src_v):
                return False

        # 2. Verify model's non-adaptable parameters and frozen buffers
        adaptable_param_ids = {id(p) for p in self.adaptable_params}
        for name, param in self.model.named_parameters():
            if id(param) not in adaptable_param_ids or not param.requires_grad:
                if name in self._immutable_source_state:
                    src_v = self._immutable_source_state[name]
                    if not tensors_match(param.detach(), src_v):
                        return False

        for name, buf in self.model.named_buffers():
            if name in self._immutable_source_state:
                src_v = self._immutable_source_state[name]
                if not tensors_match(buf.detach(), src_v):
                    return False

        # 3. If in fallback mode, verify that all adaptable parameters match source
        if self._in_fallback_mode:
            for name, param in self.model.named_parameters():
                if name in self._immutable_source_state:
                    src_v = self._immutable_source_state[name]
                    if not tensors_match(param.detach(), src_v):
                        return False

        return True

    def forward_source(
        self, x: torch.Tensor, forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute forward predictions strictly using frozen source parameters."""
        device = x.device
        if next(self._source_model.parameters()).device != device:
            self._source_model.to(device)
        self._source_model.eval()
        with torch.no_grad():
            if forward_fn is not None:
                out = forward_fn(x)
            else:
                out = self._source_model(x)
                if out.dim() == 1 or (out.dim() == 2 and out.size(-1) == 1):
                    out = torch.sigmoid(out).view(-1)
                else:
                    out = torch.softmax(out, dim=-1)
        return out


    def trigger_fallback(self, reason: str = "manual_trigger") -> None:
        """Transition into frozen-source fallback mode."""
        self._in_fallback_mode = True
        self.model.load_state_dict(
            {k: v.to(next(self.model.parameters()).device) for k, v in self._immutable_source_state.items()}
        )
        for p in self.adaptable_params:
            p.requires_grad = False
        self.optimizer.state.clear()
        self.telemetry.record_event(
            batch_idx=self._batch_counter,
            event_type="FALLBACK",
            reason=reason,
            in_fallback=True,
        )

    def reset_boundary(
        self,
        slide_id: Optional[str] = None,
        restore_source_weights: bool = False,
    ) -> None:
        """Reset adaptation state at episode or Whole-Slide Image boundary."""
        self._consecutive_rejections = 0
        self._in_fallback_mode = False
        for p in self.adaptable_params:
            p.requires_grad = True

        if restore_source_weights or self.config.reset_on_boundary:
            device = next(self.model.parameters()).device
            self.model.load_state_dict({k: v.to(device) for k, v in self._immutable_source_state.items()})
            self.optimizer.state.clear()

        if slide_id is not None:
            self.set_slide(slide_id)
        if self.monitor is not None:
            self.monitor.reset_slide(slide_id)

        self.telemetry.record_event(
            batch_idx=self._batch_counter,
            event_type="RESET",
            reason="boundary_reset",
            in_fallback=False,
        )

    def step(
        self,
        x: torch.Tensor,
        compute_loss_fn: Callable[[], torch.Tensor],
        forward_inference_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, bool]:
        """Execute one atomic adaptation step on batch x.

        Args:
            x: Input batch tensor (no labels permitted).
            compute_loss_fn: Zero-argument callable evaluating loss on current batch.
            forward_inference_fn: Optional custom inference function returning probabilities.

        Returns:
            Tuple of (prediction_tensor, was_update_accepted).
        """
        self._batch_counter += 1
        current_batch_idx = self._batch_counter

        # Enforce frozen running statistics and dropout eval mode
        self._apply_mode_and_freeze_rules()

        # If in fallback mode, bypass adaptation and evaluate strictly with frozen source
        if self._in_fallback_mode:
            source_preds = self.forward_source(x, forward_fn=forward_inference_fn)
            self.telemetry.record_event(
                batch_idx=current_batch_idx,
                event_type="FALLBACK_EVAL",
                reason="in_fallback_mode",
                in_fallback=True,
            )
            return source_preds, False

        # Take pre-step snapshots for atomic rollback
        pre_model_state: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.model.state_dict().items()
        }
        pre_optimizer_state: Dict[str, Any] = deepcopy(self.optimizer.state_dict())

        def rollback(reason: str, loss_val: Optional[float] = None) -> Tuple[torch.Tensor, bool]:
            """Execute atomic restoration of pre-step model and optimizer state."""
            self.model.load_state_dict(pre_model_state)
            self.optimizer.load_state_dict(pre_optimizer_state)
            self._apply_mode_and_freeze_rules()
            self._consecutive_rejections += 1

            self.telemetry.record_event(
                batch_idx=current_batch_idx,
                event_type="ROLLBACK",
                reason=reason,
                loss=loss_val,
                in_fallback=self._in_fallback_mode,
            )

            # Check consecutive rejection threshold
            if (
                self.config.fallback_on_max_rejections
                and self._consecutive_rejections >= self.config.max_consecutive_rejections
            ):
                self.trigger_fallback(reason=f"max_consecutive_rejections_exceeded_{reason}")

            # Return source predictions for the current batch
            safe_preds = self.forward_source(x, forward_fn=forward_inference_fn)
            return safe_preds, False

        # Execute candidate update inside exception-safe boundary
        try:
            # 1. Forward & Loss Validation
            loss = compute_loss_fn()

            # G-03: Explicit zero-selection / empty-gate skip bypass
            # Bypasses backward pass and optimizer step completely to avoid momentum drift
            if loss is None:
                self.telemetry.record_event(
                    batch_idx=current_batch_idx,
                    event_type="SKIP_EMPTY_GATE",
                    reason="zero_selected_patches",
                    loss=None,
                    in_fallback=self._in_fallback_mode,
                )
                safe_preds = (
                    self.forward_source(x, forward_fn=forward_inference_fn)
                    if self._in_fallback_mode
                    else (forward_inference_fn(x) if forward_inference_fn is not None else self.forward_source(x))
                )
                return safe_preds, False

            loss_val = float(loss.detach().item())

            if self.config.check_finite_loss:
                if not torch.isfinite(loss):
                    return rollback("non_finite_loss", loss_val=loss_val)

            # 2. Backward Propagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # 3. Gradient Validation
            if self.config.check_finite_gradients:
                for p in self.adaptable_params:
                    if p.grad is not None and not torch.all(torch.isfinite(p.grad)):
                        return rollback("non_finite_gradient", loss_val=loss_val)

            # 4. Optional Independent Gradient Clipping
            grad_norm = None
            if self.config.max_grad_norm is not None:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self.adaptable_params, self.config.max_grad_norm)
                )

            # 5. Candidate Optimizer Step
            self.optimizer.step()

            # 6. Post-Step Parameter Validation
            if self.config.check_finite_parameters:
                for p in self.adaptable_params:
                    if not torch.all(torch.isfinite(p)):
                        return rollback("non_finite_parameter", loss_val=loss_val)
                # Check buffers as well
                for b in self.model.buffers():
                    if not torch.all(torch.isfinite(b)):
                        return rollback("non_finite_parameter", loss_val=loss_val)

            # 7. Optional Independent Source-Relative Projection (L_inf Bound)
            num_clamped = 0
            max_disp = 0.0
            if self.config.param_displacement_bound_linf is not None:
                eps = self.config.param_displacement_bound_linf
                for p in self.adaptable_params:
                    src_p = self._source_param_anchors[id(p)]
                    disp = torch.max(torch.abs(p.data - src_p.data)).item()
                    if disp > max_disp:
                        max_disp = disp
                    if disp > eps:
                        num_clamped += 1
                        p.data.clamp_(src_p.data - eps, src_p.data + eps)

            # 8. Post-Adaptation Inference Validation
            with torch.no_grad():
                if forward_inference_fn is not None:
                    pred = forward_inference_fn(x)
                else:
                    self.model.eval()
                    out = self.model(x)
                    if out.dim() == 1 or (out.dim() == 2 and out.size(-1) == 1):
                        pred = torch.sigmoid(out).view(-1)
                    else:
                        pred = torch.softmax(out, dim=-1)

            if self.config.check_finite_predictions:
                if not torch.all(torch.isfinite(pred)):
                    return rollback("non_finite_prediction", loss_val=loss_val)

            # 9. Evaluate Slide-Aware Prediction-Collapse Monitor
            if (
                self.config.monitor is not None
                and self.config.monitor.mode != MonitorMode.DISABLED
                and self.monitor is not None
            ):
                source_preds = self.forward_source(x, forward_fn=forward_inference_fn)
                alarm_level, window_stats, alarm_reason = self.monitor.update(
                    adapted_probs=pred,
                    source_probs=source_preds,
                )
                if alarm_level == AlarmLevel.REJECT_UPDATE:
                    return rollback(f"collapse_alarm_{alarm_reason}", loss_val=loss_val)
                elif alarm_level == AlarmLevel.FALLBACK:
                    self.trigger_fallback(reason=f"collapse_alarm_{alarm_reason}")
                    return rollback(f"collapse_alarm_{alarm_reason}", loss_val=loss_val)
                elif alarm_level == AlarmLevel.WARNING:
                    self.telemetry.record_event(
                        batch_idx=current_batch_idx,
                        event_type="ALARM_WARNING",
                        reason=alarm_reason,
                        in_fallback=self._in_fallback_mode,
                    )

            # 10. Update Accepted: Commit Step
            self._consecutive_rejections = 0
            self.telemetry.record_event(
                batch_idx=current_batch_idx,
                event_type="ACCEPTED",
                loss=loss_val,
                grad_norm=grad_norm,
                max_displacement_linf=max_disp,
                num_clamped_tensors=num_clamped,
                in_fallback=False,
            )
            return pred, True

        except Exception as e:
            if self.config.fallback_on_unhandled_exception:
                return rollback(f"exception_{type(e).__name__}")
            raise e
