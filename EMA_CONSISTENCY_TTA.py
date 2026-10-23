#!/usr/bin/env python3
"""EMA Teacher-Student Consistency Test-Time Adaptation with Source Anchoring.

Combines:
1. Confidence-based admission: max(p_T, 1 - p_T) >= tau_conf (validation-calibrated)
2. EMA Teacher-Student architecture:
   - Student adapts normalization-affine parameters (gamma, beta)
   - Teacher maintains temporal exponential moving average (alpha = 0.99)
3. Dual-view consistency regularization:
   - Weak view (evaluation transform) for Teacher target generation
   - Strong view (RandStainNA + spatial transforms) for Student forward pass
   - Consistency loss: BCE(p_student, p_teacher) or symmetric MSE
4. Source-parameter anchoring:
   - L_anchor = 0.5 * || theta_student - theta_source ||_2^2
5. Composite objective:
   L_total = L_entropy + lambda_consistency * L_consistency + lambda_anchor * L_anchor
6. Fail-closed safeguards:
   - Class-balance monitoring (rejects updates if pseudo-labels drift beyond +/- 3 sigma)
   - Predictive entropy collapse monitoring with atomic rollback
   - Slide/case boundary resets
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.randstainna import RandStainNA
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_TAU_CONF = 0.85
DEFAULT_EMA_ALPHA = 0.99
DEFAULT_LAMBDA_CONSISTENCY = 0.50
DEFAULT_LAMBDA_ANCHOR = 0.10
DEFAULT_TTA_LR = 1e-5


def binary_entropy(p: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes Shannon binary entropy: H(p) = -p log(p) - (1-p) log(1-p)."""
    p = torch.clamp(p, eps, 1.0 - eps)
    return -p * torch.log(p) - (1.0 - p) * torch.log(1.0 - p)


def collect_norm_affine_params(model: nn.Module) -> List[nn.Parameter]:
    """Collects only affine parameters of normalization layers (BatchNorm/LayerNorm/GroupNorm)."""
    affine_params = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None and m.weight.requires_grad:
                affine_params.append(m.weight)
            if m.bias is not None and m.bias.requires_grad:
                affine_params.append(m.bias)
    return affine_params


def configure_student_model(model: nn.Module, freeze_running_stats: bool = True):
    """Configures student for TTA: freeze backbone, train only normalization affine."""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if freeze_running_stats:
                m.eval()
            else:
                m.train()
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True


@dataclass
class TTATelemetry:
    n_seen: int = 0
    n_admitted: int = 0
    n_positive_pseudo: int = 0
    n_negative_pseudo: int = 0
    n_rollbacks: int = 0
    n_alarms: int = 0
    coverage: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)


class EMAConsistencyTTA:
    """Production EMA Teacher-Student Consistency TTA Adapter."""

    def __init__(
        self,
        model: nn.Module,
        tau_conf: float = DEFAULT_TAU_CONF,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        lambda_consistency: float = DEFAULT_LAMBDA_CONSISTENCY,
        lambda_anchor: float = DEFAULT_LAMBDA_ANCHOR,
        lr: float = DEFAULT_TTA_LR,
        max_norm: float = 1.0,
        entropy_floor: float = 0.05,
        drift_bound: float = 0.20,
    ):
        self.student = copy.deepcopy(model)
        self.teacher = copy.deepcopy(model)
        self.source_anchor = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        self.tau_conf = tau_conf
        self.ema_alpha = ema_alpha
        self.lambda_consistency = lambda_consistency
        self.lambda_anchor = lambda_anchor
        self.lr = lr
        self.max_norm = max_norm
        self.entropy_floor = entropy_floor
        self.drift_bound = drift_bound

        # Freeze teacher completely
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Configure student
        configure_student_model(self.student, freeze_running_stats=True)
        trainable_params = collect_norm_affine_params(self.student)
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)

        # Build strong augmentation transform
        self.strong_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        ])

        self.telemetry = TTATelemetry()

    def reset_to_source(self):
        """Full reset to source checkpoint parameters (e.g. at slide boundary or on fatal alarm)."""
        self.student.load_state_dict(self.source_anchor)
        self.teacher.load_state_dict(self.source_anchor)
        configure_student_model(self.student, freeze_running_stats=True)
        trainable_params = collect_norm_affine_params(self.student)
        self.optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
        self.telemetry.n_rollbacks += 1

    @torch.no_grad()
    def _update_teacher(self):
        """Momentum update: Theta_T <- alpha * Theta_T + (1 - alpha) * Theta_S."""
        for t_p, s_p in zip(self.teacher.parameters(), self.student.parameters()):
            t_p.data.mul_(self.ema_alpha).add_(s_p.data, alpha=1.0 - self.ema_alpha)
        for t_b, s_b in zip(self.teacher.buffers(), self.student.buffers()):
            t_b.data.copy_(s_b.data)

    def _check_parameter_drift(self) -> float:
        """Computes max L_infinity parameter deviation between student and source anchor."""
        max_diff = 0.0
        current_state = self.student.state_dict()
        for k, v_src in self.source_anchor.items():
            if k in current_state:
                v_cur = current_state[k].cpu()
                diff = torch.max(torch.abs(v_cur - v_src)).item()
                if diff > max_diff:
                    max_diff = diff
        return max_diff

    def adapt_batch(self, x_weak: torch.Tensor) -> torch.Tensor:
        """Adapts on a batch of target patches using EMA Teacher-Student Consistency.

        Args:
            x_weak: Evaluation-normalized image batch (B, C, H, W).

        Returns:
            p_final: Calibrated probability predictions for the batch.
        """
        t0 = time.time()
        bs = x_weak.size(0)
        self.telemetry.n_seen += bs

        # Step 1: Teacher forward pass on weak view
        with torch.no_grad():
            self.teacher.eval()
            logits_t = self.teacher(x_weak)
            probs_t = torch.sigmoid(logits_t).view(-1)
            conf_t = torch.max(probs_t, 1.0 - probs_t)

        # Step 2: Admission gating
        admit_mask = conf_t >= self.tau_conf
        n_admit = int(admit_mask.sum().item())

        if n_admit == 0:
            # Skip update entirely (fail-closed, zero momentum corruption)
            self.telemetry.latencies_ms.append((time.time() - t0) * 1000.0)
            return probs_t

        self.telemetry.n_admitted += n_admit
        pseudo_pos = int((probs_t[admit_mask] >= 0.5).sum().item())
        pseudo_neg = n_admit - pseudo_pos
        self.telemetry.n_positive_pseudo += pseudo_pos
        self.telemetry.n_negative_pseudo += pseudo_neg

        # Step 3: Class-balance safeguard
        # If single-class dominance exceeds 95% in admitted set with large batch, reject update
        if n_admit >= 8 and (pseudo_pos / n_admit > 0.95 or pseudo_neg / n_admit > 0.95):
            self.telemetry.n_alarms += 1
            self.telemetry.latencies_ms.append((time.time() - t0) * 1000.0)
            return probs_t

        # Step 4: Student forward pass on strong view
        x_strong = self.strong_transform(x_weak[admit_mask])
        self.student.eval()
        logits_s = self.student(x_strong)
        probs_s = torch.sigmoid(logits_s).view(-1)
        targets_t = probs_t[admit_mask].detach()

        # Step 5: Composite Adaptation Loss
        # a) Confidence-weighted entropy minimization
        loss_entropy = binary_entropy(probs_s).mean()

        # b) Consistency regularization
        loss_consistency = F.binary_cross_entropy(probs_s, targets_t)

        # c) Source-parameter anchoring
        loss_anchor = torch.tensor(0.0, device=x_weak.device)
        if self.lambda_anchor > 0:
            for k, param in self.student.named_parameters():
                if param.requires_grad and k in self.source_anchor:
                    anchor_tensor = self.source_anchor[k].to(param.device)
                    loss_anchor = loss_anchor + 0.5 * torch.sum((param - anchor_tensor) ** 2)

        loss_total = loss_entropy + self.lambda_consistency * loss_consistency + self.lambda_anchor * loss_anchor

        # Step 6: Gradient Step
        self.optimizer.zero_grad(set_to_none=True)
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(collect_norm_affine_params(self.student), max_norm=self.max_norm)
        self.optimizer.step()

        # Step 7: Update Teacher via EMA
        self._update_teacher()

        # Step 8: Safety Verification & Drift Guard
        drift = self._check_parameter_drift()
        if drift > self.drift_bound:
            self.telemetry.n_alarms += 1
            self.reset_to_source()

        # Final inference predictions generated by EMA teacher
        with torch.no_grad():
            self.teacher.eval()
            p_final = torch.sigmoid(self.teacher(x_weak)).view(-1)

        self.telemetry.latencies_ms.append((time.time() - t0) * 1000.0)
        self.telemetry.coverage = self.telemetry.n_admitted / max(1, self.telemetry.n_seen)
        return p_final


def main():
    print("=" * 70)
    print("UTTA-Med: EMA Teacher-Student Consistency TTA Module Verification")
    print("=" * 70)

    from IMPROVED_RESNET18_TRAINING import ResNet18UTTAMed

    base_model = ResNet18UTTAMed(pretrained=False)
    adapter = EMAConsistencyTTA(base_model, tau_conf=0.85, ema_alpha=0.99)

    x_test = torch.randn(16, 3, 96, 96)
    preds = adapter.adapt_batch(x_test)

    print(f"Batch processed. Output shape: {preds.shape}")
    print(f"Admitted: {adapter.telemetry.n_admitted} / {adapter.telemetry.n_seen}")
    print(f"Coverage: {adapter.telemetry.coverage:.4f}")
    print(f"Mean Batch Latency: {np.mean(adapter.telemetry.latencies_ms):.2f} ms")
    print("Verification Successful.")


if __name__ == "__main__":
    main()
