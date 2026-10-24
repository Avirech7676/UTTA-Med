"""SafeAdapt v2 Gating Policies.

Stage D1: Faithful implementation of patch-level test-time adaptation gating policies:
- Policy A: Frozen Source (Zero updates, deterministic baseline)
- Policy B: Confidence-Only (Deterministic patch-level screening)
- Policy C: MC-Only (Faithful reproduction of v1 UTTA-Med 20-pass MC-Dropout)
- Policy D: Hybrid (Deterministic confidence screening + selective MC-Dropout on ambiguous patches)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GatingDecision:
    """Decision output for a single batch of patches."""

    keep_mask: torch.Tensor          # Boolean tensor [B] indicating which patches to adapt on
    confidences: torch.Tensor        # Float tensor [B] of deterministic confidence values max(p, 1-p)
    uncertainties: torch.Tensor      # Float tensor [B] of predictive variances (or zeros if uncomputed)
    routed_to_mc: torch.Tensor       # Boolean tensor [B] indicating which patches were routed to MC passes
    mc_passes_executed: int          # Number of stochastic forward passes executed for this batch

    @property
    def num_selected(self) -> int:
        return int(self.keep_mask.sum().item())

    @property
    def num_eligible(self) -> int:
        return int(self.keep_mask.numel())

    @property
    def selection_ratio(self) -> float:
        return self.num_selected / max(self.num_eligible, 1)

    @property
    def num_routed_to_mc(self) -> int:
        return int(self.routed_to_mc.sum().item())


class BaseGatingPolicy(ABC):
    """Abstract base class for patch-level adaptation gating."""

    @abstractmethod
    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        """Evaluate gating policy on batch x.

        Notice: Accepts ONLY model and input batch x. Target labels y are strictly prohibited.
        """
        pass


class FrozenSourcePolicy(BaseGatingPolicy):
    """Policy A: Frozen Source.
    
    Zero updates are performed. Keep mask is always all False.
    Zero MC passes executed.
    """

    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        b = x.size(0)
        device = x.device
        with torch.no_grad():
            model.eval()
            out = model(x)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)
        return GatingDecision(
            keep_mask=torch.zeros(b, dtype=torch.bool, device=device),
            confidences=conf,
            uncertainties=torch.zeros(b, dtype=torch.float32, device=device),
            routed_to_mc=torch.zeros(b, dtype=torch.bool, device=device),
            mc_passes_executed=0,
        )


class ConfidenceGatingPolicy(BaseGatingPolicy):
    """Policy B: Deterministic Patch-Level Confidence Gating.
    
    Computes confidence c_i = max(p_i, 1 - p_i).
    Selects patch i if c_i >= tau_conf (or > tau_conf).
    Zero stochastic MC-dropout passes executed.
    """

    def __init__(self, tau_conf: float = 0.90, strictly_greater: bool = True):
        self.tau_conf = float(tau_conf)
        self.strictly_greater = strictly_greater

    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        b = x.size(0)
        device = x.device
        with torch.no_grad():
            model.eval()
            out = model(x)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)
            if self.strictly_greater:
                keep = conf > self.tau_conf
            else:
                keep = conf >= self.tau_conf

        return GatingDecision(
            keep_mask=keep,
            confidences=conf,
            uncertainties=torch.zeros(b, dtype=torch.float32, device=device),
            routed_to_mc=torch.zeros(b, dtype=torch.bool, device=device),
            mc_passes_executed=0,
        )


class MCOnlyGatingPolicy(BaseGatingPolicy):
    """Policy C: Faithful Reproduction of v1 UTTA-Med MC-Dropout Gating.
    
    Computes predictive variance U_i across K stochastic passes with dropout enabled.
    Selects patch i if U_i < tau_unc.
    """

    def __init__(
        self,
        tau_unc: float = 8.875383e-06,
        n_passes: int = 20,
        seed: Optional[int] = None,
    ):
        self.tau_unc = float(tau_unc)
        self.n_passes = int(n_passes)
        self.seed = seed
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        b = x.size(0)
        device = x.device

        # Preserve BatchNorm running statistics in eval mode while enabling Dropout
        model.eval()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        probs = []
        with torch.no_grad():
            for _ in range(self.n_passes):
                out = model(x)
                p = torch.sigmoid(out).view(-1)
                probs.append(p)

        # Restore model to strict eval mode
        model.eval()

        stacked = torch.stack(probs, dim=0)  # [n_passes, B]
        mean_p = stacked.mean(dim=0)
        unc = stacked.var(dim=0, unbiased=False)  # Matches v1: unbiased=False
        conf = torch.maximum(mean_p, 1.0 - mean_p)

        keep = unc < self.tau_unc

        return GatingDecision(
            keep_mask=keep,
            confidences=conf,
            uncertainties=unc,
            routed_to_mc=torch.ones(b, dtype=torch.bool, device=device),
            mc_passes_executed=self.n_passes,
        )


class HybridGatingPolicy(BaseGatingPolicy):
    """Policy D: Efficient Hybrid Confidence/Uncertainty Gating.
    
    Two-stage decision arbitration:
    1. Deterministic screening:
       - Confident patches (c_i >= theta_high): Unambiguous, auto-accepted directly without MC-dropout.
       - Low-confidence noise (c_i < theta_low): Pure noise / artifact, auto-rejected without MC-dropout.
    2. Ambiguous margin (theta_low <= c_i < theta_high):
       - Route ONLY this ambiguous subset to MC-dropout passes!
       - If ambiguous patch has U_i < tau_unc, accept it; otherwise reject.
       
    This preserves epistemic uncertainty protection while eliminating 70-90% of redundant MC passes.
    """

    def __init__(
        self,
        theta_high: float = 0.95,
        theta_low: float = 0.65,
        tau_unc: float = 8.875383e-06,
        n_passes: int = 20,
        seed: Optional[int] = None,
    ):
        self.theta_high = float(theta_high)
        self.theta_low = float(theta_low)
        self.tau_unc = float(tau_unc)
        self.n_passes = int(n_passes)
        self.seed = seed
        self._rng = torch.Generator(device="cpu")
        if seed is not None:
            self._rng.manual_seed(seed)

    def select(self, model: nn.Module, x: torch.Tensor) -> GatingDecision:
        b = x.size(0)
        device = x.device

        # Stage 1: Fast deterministic screening
        with torch.no_grad():
            model.eval()
            out = model(x)
            p = torch.sigmoid(out).view(-1)
            conf = torch.maximum(p, 1.0 - p)

        # Categorize patches
        is_confident = conf >= self.theta_high
        is_low_conf = conf < self.theta_low
        is_ambiguous = (~is_confident) & (~is_low_conf)

        keep_mask = torch.zeros(b, dtype=torch.bool, device=device)
        uncertainties = torch.zeros(b, dtype=torch.float32, device=device)
        routed_to_mc = is_ambiguous.clone()

        # Confident patches are auto-accepted directly
        keep_mask[is_confident] = True

        # Stage 2: Selective MC-dropout on ambiguous subset ONLY
        num_ambiguous = int(is_ambiguous.sum().item())
        mc_passes_done = 0

        if num_ambiguous > 0:
            mc_passes_done = self.n_passes
            amb_indices = torch.nonzero(is_ambiguous, as_tuple=True)[0]
            x_amb = x[amb_indices]

            model.eval()
            for m in model.modules():
                if isinstance(m, nn.Dropout):
                    m.train()

            amb_probs = []
            with torch.no_grad():
                for _ in range(self.n_passes):
                    p_k = torch.sigmoid(model(x_amb)).view(-1)
                    amb_probs.append(p_k)

            model.eval()

            stacked_amb = torch.stack(amb_probs, dim=0)  # [K, num_ambiguous]
            unc_amb = stacked_amb.var(dim=0, unbiased=False)
            uncertainties[amb_indices] = unc_amb

            # Accept ambiguous patches that exhibit low epistemic uncertainty
            amb_keep = unc_amb < self.tau_unc
            keep_mask[amb_indices] = amb_keep

        return GatingDecision(
            keep_mask=keep_mask,
            confidences=conf,
            uncertainties=uncertainties,
            routed_to_mc=routed_to_mc,
            mc_passes_executed=mc_passes_done,
        )
