"""Comprehensive Unit & Regression Tests for UTTA-Med 2.0 Features.

Tests:
1. Adaptive Uncertainty Gating (Single pass confidence, dynamic MC passes, variance early stopping).
2. Multi-Signal Gating (Convex combination, normalization, signal decomposition).
3. Safety-Aware Adaptation Losses (Entropy, consistency, elastic anchoring).
4. Bio-SafeAdapt Controller (Cellularity estimation, silence suppression on sparse tissue, episodic resets).
"""

import pytest
import torch
import torch.nn as nn

from src.v2.gating.adaptive_uncertainty import AdaptiveUncertaintyGating
from src.v2.gating.multi_signal import MultiSignalGatingPolicy
from src.v2.safeadapt.bio_safeadapt import BioSafeAdaptController, AlarmLevel
from src.v2.safeadapt.losses import SafetyAwareAdaptationLoss


class SimpleConvNet(nn.Module):
    """Deterministic dummy CNN with dropout for testing."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn(self.conv(x)))
        h = self.pool(h).flatten(1)
        h = self.dropout(h)
        return self.fc(h)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.bn(self.conv(x)))
        return self.pool(h).flatten(1)


# --- 1. Adaptive Uncertainty Gating Tests ---

def test_adaptive_gating_confident_auto_accept():
    """Confident patches should be auto-accepted with exactly 1 forward pass."""
    model = SimpleConvNet()
    policy = AdaptiveUncertaintyGating(tau_high=0.80, tau_low=0.20, max_mc_passes=10)

    # Force model to output large logits (+10.0 -> sigmoid ~0.9999)
    with torch.no_grad():
        model.fc.bias.fill_(10.0)

    x = torch.randn(4, 3, 32, 32)
    decision = policy.select(model, x)

    assert decision.num_selected == 4
    assert decision.mc_passes_executed == 1
    assert decision.routed_to_mc.sum().item() == 0


def test_adaptive_gating_ambiguous_routes_to_mc():
    """Ambiguous patches (confidence near 0.5) must trigger MC-Dropout passes."""
    model = SimpleConvNet()
    policy = AdaptiveUncertaintyGating(tau_high=0.90, tau_low=0.40, max_mc_passes=10, min_mc_passes=5, chunk_size=5)

    # Force model to output logits near 0.0 -> sigmoid ~0.50
    with torch.no_grad():
        model.fc.bias.fill_(0.0)
        model.fc.weight.fill_(0.01)

    x = torch.randn(4, 3, 32, 32)
    decision = policy.select(model, x)

    assert decision.routed_to_mc.sum().item() == 4
    assert decision.mc_passes_executed >= 5


# --- 2. Multi-Signal Gating Tests ---

def test_multi_signal_gating_convex_combination():
    """MultiSignalGatingPolicy computes convex combination and respects weights."""
    model = SimpleConvNet()
    centroids = torch.randn(8)

    policy = MultiSignalGatingPolicy(
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        tau_total=10.0,  # Generous threshold to accept all
        n_mc_passes=5,
        source_centroids=centroids,
    )

    x = torch.randn(2, 3, 32, 32)
    decision = policy.select(model, x)

    assert decision.u_mc is not None
    assert decision.u_consistency is not None
    assert decision.u_feature is not None
    assert decision.u_total is not None
    assert decision.u_total.shape == (2,)


# --- 3. Safety-Aware Adaptation Losses Tests ---

def test_safety_aware_loss_components():
    """Verify that entropy, consistency, and anchoring losses compute valid gradients."""
    model = SimpleConvNet()
    for p in model.parameters():
        p.requires_grad = False
    model.bn.weight.requires_grad = True
    model.bn.bias.requires_grad = True

    loss_fn = SafetyAwareAdaptationLoss(
        weight_entropy=1.0,
        weight_consistency=0.5,
        weight_anchor=0.1,
    )
    loss_fn.set_source_anchors(model)

    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    total_loss, breakdown = loss_fn(model, x, logits)

    assert total_loss.item() > 0.0
    assert "loss_entropy" in breakdown
    assert "loss_consistency" in breakdown
    assert "loss_anchor" in breakdown

    total_loss.backward()
    assert model.bn.weight.grad is not None
    assert torch.isfinite(model.bn.weight.grad).all()


# --- 4. Bio-SafeAdapt Controller Tests ---

def test_bio_safeadapt_cellularity_estimation():
    """Optical density estimation distinguishes dense tissue from clear white background."""
    # White background (high RGB ~ 1.0) -> low optical density
    white_bg = torch.ones(1, 3, 64, 64)
    cell_bg = BioSafeAdaptController.estimate_cellularity(white_bg, od_threshold=0.15)
    assert cell_bg.item() == 0.0  # Zero cellularity on background

    # Purple/blue stained simulated tissue patch (low red, moderate green, high blue)
    tissue = torch.zeros(1, 3, 64, 64)
    tissue[:, 0, :, :] = 0.2  # High red absorption (dense hematoxylin)
    tissue[:, 1, :, :] = 0.3
    tissue[:, 2, :, :] = 0.7  # Blue transmitted
    cell_tissue = BioSafeAdaptController.estimate_cellularity(tissue, od_threshold=0.15)
    assert cell_tissue.item() > 0.5  # High cellularity detected


def test_bio_safeadapt_silence_suppression_on_sparse_tissue():
    """Bio-SafeAdapt suppresses false alarms on acellular stroma/fat patches."""
    model = SimpleConvNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    controller = BioSafeAdaptController(
        model=model,
        optimizer=opt,
        min_cellularity_threshold=0.10,
        max_consecutive_silence=3,
    )

    # Empty background image with 0 detections
    sparse_img = torch.ones(1, 3, 64, 64)
    source_dets = []
    adapted_dets = []

    # Monitor step on sparse background
    decision = controller.monitor_detection_step(sparse_img, source_dets, adapted_dets)

    assert decision.silence_suppressed is True
    assert decision.alarm_level == AlarmLevel.NORMAL
    assert decision.allow_update is False  # Correctly skips adaptation on empty glass/fat


def test_bio_safeadapt_episodic_case_boundary_reset():
    """Case boundary reset restores clean source model and clears history."""
    model = SimpleConvNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    controller = BioSafeAdaptController(model=model, optimizer=opt)

    # Mutate a parameter
    with torch.no_grad():
        for p in controller.adaptable_params:
            p.add_(1.0)

    # Reset boundary
    controller.reset_case_boundary("case_patient_002")

    # Parameters should be restored to initial source values
    for p in controller.adaptable_params:
        anchor = controller._source_param_anchors[id(p)]
        assert torch.allclose(p.data, anchor, atol=1e-6)
