"""SafeAdapt v2 Gated Test-Time Adaptation Adapter.

Stage D1: Bridges patch-level GatingPolicies with the SafeAdaptController.
Ensures:
- Patch-level loss masking: only selected patches contribute to entropy loss.
- Zero-selected-patch batches skip backward and optimizer steps cleanly.
- Strict frozen BatchNorm running statistics.
- Atomic state rollback on non-finite loss, gradients, parameters, or collapse alarms.
- Granular telemetry tracking distinguishing patch coverage, update coverage, and MC compute ratio.
- Zero target-label ingress.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.v2.gating.policies import BaseGatingPolicy, FrozenSourcePolicy, GatingDecision
from src.v2.gating.telemetry import GatingTelemetryTracker
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController


class GatedSafeAdaptAdapter:
    """Safe test-time adaptation adapter with pluggable patch-level gating policies."""

    def __init__(
        self,
        model: nn.Module,
        policy: BaseGatingPolicy,
        config: Optional[SafeAdaptConfig] = None,
        policy_name: str = "gating_policy",
    ):
        self.model = model
        self.policy = policy
        self.config = config or SafeAdaptConfig()
        self.config.validate()

        # Isolate adaptable parameters strictly (BatchNorm affine by default)
        for p in self.model.parameters():
            p.requires_grad = False
        adaptable_params = []
        for mod in self.model.modules():
            if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and mod.affine:
                if mod.weight is not None:
                    mod.weight.requires_grad = True
                    adaptable_params.append(mod.weight)
                if mod.bias is not None:
                    mod.bias.requires_grad = True
                    adaptable_params.append(mod.bias)

        if not adaptable_params:
            # Fallback for non-BN architectures: all parameters with requires_grad
            for p in self.model.parameters():
                p.requires_grad = True
            adaptable_params = list(self.model.parameters())

        self.optimizer = torch.optim.Adam(adaptable_params, lr=self.config.tta_lr)
        self.controller = SafeAdaptController(
            model=self.model,
            optimizer=self.optimizer,
            config=self.config,
            adaptable_params=adaptable_params,
        )

        self.telemetry = GatingTelemetryTracker(policy_name=policy_name)
        self._source_anchors: Dict[int, torch.Tensor] = {
            id(p): p.detach().clone() for p in adaptable_params
        }

    @property
    def in_fallback_mode(self) -> bool:
        """Check if controller is currently in permanent source fallback mode."""
        return self.controller.in_fallback_mode

    def set_slide(self, slide_id: str) -> None:
        """Switch to a new Whole-Slide Image."""
        self.controller.set_slide(slide_id)

    def reset_boundary(self, slide_id: Optional[str] = None, restore_source_weights: bool = False) -> None:
        """Reset adaptation state at slide or episode boundary."""
        self.controller.reset_boundary(slide_id=slide_id, restore_source_weights=restore_source_weights)

    def adapt_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, GatingDecision]:
        """Execute one gated adaptation step on input batch x.

        Notice: Accepts ONLY input batch tensor x. Target labels y are strictly prohibited.

        Returns:
            Tuple of (prediction_probabilities, gating_decision).
        """
        b = x.size(0)

        # Baseline source predictions for telemetry delta tracking
        with torch.no_grad():
            self.model.eval()
            p_source = torch.sigmoid(self.model(x)).view(-1)

        # 1. Evaluate Gating Policy
        t_gate_start = time.perf_counter()
        decision = self.policy.select(self.model, x)
        t_gate_end = time.perf_counter()
        gating_latency_ms = (t_gate_end - t_gate_start) * 1000.0

        n_selected = decision.num_selected
        w = decision.keep_mask.float()

        # Handle Policy A (Frozen Source) or zero selected patches
        if n_selected == 0:
            t_step_start = time.perf_counter()
            with torch.no_grad():
                self.model.eval()
                preds = torch.sigmoid(self.model(x)).view(-1)
            t_step_end = time.perf_counter()
            step_latency_ms = (t_step_end - t_step_start) * 1000.0

            # Record telemetry for skipped batch
            self.telemetry.record_batch(
                eligible_patches=b,
                selected_patches=0,
                routed_to_mc_patches=decision.num_routed_to_mc,
                mc_passes_executed=decision.mc_passes_executed,
                attempted_update=False,
                accepted_update=False,
                rejected_update=False,
                in_fallback=self.controller.in_fallback_mode,
                gating_latency_ms=gating_latency_ms,
                step_latency_ms=step_latency_ms,
                max_param_displacement=self._compute_max_displacement(),
                mean_abs_prob_diff=0.0,
            )
            return preds, decision

        # 2. Define Masked Adaptation Loss
        def compute_masked_loss() -> torch.Tensor:
            self.model.train()
            # Enforce frozen running statistics
            if self.config.freeze_running_stats:
                for m in self.model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()
                    elif isinstance(m, nn.Dropout):
                        m.eval()

            logits = self.model(x)
            p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).view(-1)
            # Masked mean entropy over selected patches only
            return (w * ent).sum() / w.sum()

        def forward_inference(x_eval: torch.Tensor) -> torch.Tensor:
            self.model.eval()
            out = self.model(x_eval)
            return torch.sigmoid(out).view(-1)

        # 3. Execute Candidate Update inside SafeAdapt Controller
        t_step_start = time.perf_counter()
        preds, accepted = self.controller.step(
            x=x,
            compute_loss_fn=compute_masked_loss,
            forward_inference_fn=forward_inference,
        )
        t_step_end = time.perf_counter()
        step_latency_ms = (t_step_end - t_step_start) * 1000.0

        # Calculate parameter displacement and probability shift
        max_disp = self._compute_max_displacement()
        mean_prob_diff = float((preds - p_source).abs().mean().item())

        self.telemetry.record_batch(
            eligible_patches=b,
            selected_patches=n_selected,
            routed_to_mc_patches=decision.num_routed_to_mc,
            mc_passes_executed=decision.mc_passes_executed,
            attempted_update=True,
            accepted_update=accepted,
            rejected_update=not accepted,
            in_fallback=self.controller.in_fallback_mode,
            gating_latency_ms=gating_latency_ms,
            step_latency_ms=step_latency_ms,
            max_param_displacement=max_disp,
            mean_abs_prob_diff=mean_prob_diff,
        )

        return preds, decision

    def _compute_max_displacement(self) -> float:
        """Compute maximum Linf parameter displacement from immutable source weights."""
        max_disp = 0.0
        for p in self.controller.adaptable_params:
            src_p = self._source_anchors[id(p)]
            disp = float((p.data - src_p.data).abs().max().item())
            if disp > max_disp:
                max_disp = disp
        return max_disp
