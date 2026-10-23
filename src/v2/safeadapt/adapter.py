"""SafeAdapt v2 TTA Adapter Interface.

Provides a clean test-time adaptation wrapper wrapping SafeAdaptController for streaming
patch-level digital pathology inference without requiring target labels.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController


def binary_entropy_from_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute binary Shannon entropy from logits."""
    p = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).view(-1)


class SafeAdaptTTA:
    """Safe test-time adaptation wrapper.

    Works seamlessly across ResNet-18, ResNet-50, or custom backbones.
    Operates strictly on test inputs `x` without accepting ground truth labels `y`.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[SafeAdaptConfig] = None,
        loss_type: str = "entropy",
        gate_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.model = model
        self.config = config or SafeAdaptConfig()
        self.config.validate()

        # Build optimizer strictly over adaptable parameters
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            # Let controller configure scope
            temp_ctrl = SafeAdaptController(model, torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=self.config.tta_lr), self.config)
            params = temp_ctrl.adaptable_params

        self.optimizer = torch.optim.Adam(params, lr=self.config.tta_lr)
        self.controller = SafeAdaptController(
            model=model,
            optimizer=self.optimizer,
            config=self.config,
            adaptable_params=params,
        )

        self.loss_type = loss_type
        self.gate_fn = gate_fn

    @property
    def telemetry(self):
        """Return controller telemetry tracker."""
        return self.controller.telemetry

    def reset_boundary(self, restore_source_weights: bool = False) -> None:
        """Reset adaptation state at slide or episode boundary."""
        self.controller.reset_boundary(restore_source_weights=restore_source_weights)

    def adapt_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Process incoming batch x and return prediction probabilities.

        Notice: Signature strictly accepts ONLY input batch tensor `x`.
        Target labels `y` are strictly prohibited.
        """
        def compute_loss() -> torch.Tensor:
            self.model.train()
            # Preserve frozen running stats
            if self.config.freeze_running_stats:
                for m in self.model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        m.eval()
                    elif isinstance(m, nn.Dropout):
                        m.eval()

            logits = self.model(x)
            if self.loss_type == "entropy":
                ent = binary_entropy_from_logits(logits)
                if self.gate_fn is not None:
                    mask = self.gate_fn(logits, x)
                    if mask.sum() < 1:
                        # Return None to explicitly skip backward and optimizer step
                        return None
                    return (mask * ent).sum() / mask.sum()
                return ent.mean()
            raise NotImplementedError(f"Unsupported loss_type: {self.loss_type}")

        def forward_inference(x_eval: torch.Tensor) -> torch.Tensor:
            self.model.eval()
            out = self.model(x_eval)
            return torch.sigmoid(out).view(-1)

        preds, _ = self.controller.step(
            x=x,
            compute_loss_fn=compute_loss,
            forward_inference_fn=forward_inference,
        )
        return preds
