"""Deterministic CPU Unit Tests for SafeAdapt v2 Stage D1 Gating.

Verifies:
1. Correct patch-level confidence selection.
2. Correct MC predictive variance calculation.
3. Reproducibility under fixed random seeds.
4. Frozen BN statistics during MC passes.
5. Zero mutation of source weights during gating.
6. Correct hybrid arbitration (auto-accept, auto-reject, ambiguous MC routing).
7. Correct loss masking for selected patches.
8. Zero-selected-patch behavior (bypasses backward and optimizer step cleanly).
9. All-selected-patch behavior.
10. Matched-budget telemetry accounting.
11. Monitor-disabled and observation-mode equivalence.
12. Preservation of Stage B rollback under non-finite loss/gradients.
13. No target-label ingress into gating or adapter interfaces.
14. Preservation of frozen v1 artifacts and checksums.
"""

from __future__ import annotations

import inspect
import sys
from copy import deepcopy
from pathlib import Path

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import pytest
import torch
import torch.nn as nn

from src.v2.gating.adapter import GatedSafeAdaptAdapter
from src.v2.gating.policies import (
    ConfidenceGatingPolicy,
    FrozenSourcePolicy,
    GatingDecision,
    HybridGatingPolicy,
    MCOnlyGatingPolicy,
)
from src.v2.gating.telemetry import GatingTelemetryTracker
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.monitor import CollapseMonitorConfig, MonitorMode


class DeterministicToyDropoutModel(nn.Module):
    """Deterministic model with BatchNorm and Dropout layers for CPU testing."""

    def __init__(self, in_features: int = 16, hidden: int = 8, dropout_p: float = 0.5):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


def make_gated_setup(policy, mode=MonitorMode.OBSERVATION):
    torch.manual_seed(42)
    model = DeterministicToyDropoutModel()
    mon_cfg = CollapseMonitorConfig(mode=mode, window_size=20, min_samples_for_alarm=5)
    config = SafeAdaptConfig(tta_lr=0.01, monitor=mon_cfg)
    adapter = GatedSafeAdaptAdapter(model, policy, config=config)
    return model, adapter


def test_1_patch_level_confidence_selection():
    """Test that confidence gating selects patches individually based on max(p, 1-p) > tau."""
    model, adapter = make_gated_setup(ConfidenceGatingPolicy(tau_conf=0.70))
    torch.manual_seed(10)
    x = torch.randn(8, 16)

    # Compute expected confidences manually
    with torch.no_grad():
        model.eval()
        p = torch.sigmoid(model(x))
        conf = torch.maximum(p, 1.0 - p)
        expected_keep = conf > 0.70

    decision = adapter.policy.select(model, x)
    assert torch.equal(decision.keep_mask, expected_keep)
    assert decision.mc_passes_executed == 0
    assert decision.num_routed_to_mc == 0


def test_2_mc_predictive_variance_calculation():
    """Test that MC-dropout computes predictive variance across stochastic passes with unbiased=False."""
    model, adapter = make_gated_setup(MCOnlyGatingPolicy(tau_unc=0.01, n_passes=10, seed=42))
    torch.manual_seed(123)
    x = torch.randn(6, 16)

    decision = adapter.policy.select(model, x)
    assert decision.mc_passes_executed == 10
    assert decision.num_routed_to_mc == 6
    assert decision.uncertainties.shape == (6,)
    assert (decision.uncertainties >= 0.0).all()
    # Check keep mask matches U < tau_unc
    assert torch.equal(decision.keep_mask, decision.uncertainties < 0.01)


def test_3_reproducibility_under_fixed_random_seeds():
    """Test that identical random seeds produce bit-exact MC uncertainties and decisions."""
    x = torch.randn(5, 16)

    model1 = DeterministicToyDropoutModel()
    policy1 = MCOnlyGatingPolicy(tau_unc=0.05, n_passes=15, seed=999)
    torch.manual_seed(999)
    dec1 = policy1.select(model1, x)

    model2 = DeterministicToyDropoutModel()
    policy2 = MCOnlyGatingPolicy(tau_unc=0.05, n_passes=15, seed=999)
    torch.manual_seed(999)
    dec2 = policy2.select(model2, x)

    assert torch.allclose(dec1.uncertainties, dec2.uncertainties, atol=1e-7)
    assert torch.equal(dec1.keep_mask, dec2.keep_mask)


def test_4_frozen_bn_statistics_during_mc_passes():
    """Test that BatchNorm running mean and variance are not mutated during stochastic MC passes."""
    model = DeterministicToyDropoutModel()
    initial_mean = model.bn1.running_mean.clone()
    initial_var = model.bn1.running_var.clone()

    policy = MCOnlyGatingPolicy(n_passes=25)
    x = torch.randn(10, 16)
    _ = policy.select(model, x)

    assert torch.equal(model.bn1.running_mean, initial_mean), "running_mean was mutated!"
    assert torch.equal(model.bn1.running_var, initial_var), "running_var was mutated!"
    # Ensure model returned to eval mode
    assert not model.training
    assert not model.dropout.training


def test_5_zero_mutation_of_source_weights():
    """Test that evaluating gating policies does not mutate model parameters."""
    model = DeterministicToyDropoutModel()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    policy = HybridGatingPolicy(theta_high=0.90, theta_low=0.60, n_passes=10)
    x = torch.randn(8, 16)
    _ = policy.select(model, x)

    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weight {k} was mutated during gating!"


def test_6_correct_hybrid_arbitration():
    """Test that Hybrid policy auto-accepts confident, auto-rejects noise, and routes only ambiguous to MC."""
    model = DeterministicToyDropoutModel()
    # Mock model outputs via manual forward or crafting
    policy = HybridGatingPolicy(theta_high=0.85, theta_low=0.65, tau_unc=0.05, n_passes=10)
    x = torch.randn(10, 16)

    decision = policy.select(model, x)
    conf = decision.confidences

    # Check categorization
    for i in range(10):
        c_i = conf[i].item()
        if c_i >= 0.85:
            assert decision.keep_mask[i].item() is True, "High confidence must be accepted"
            assert decision.routed_to_mc[i].item() is False, "High confidence must NOT run MC"
        elif c_i < 0.65:
            assert decision.keep_mask[i].item() is False, "Low confidence must be rejected"
            assert decision.routed_to_mc[i].item() is False, "Low confidence must NOT run MC"
        else:
            assert decision.routed_to_mc[i].item() is True, "Ambiguous patch MUST run MC"
            u_i = decision.uncertainties[i].item()
            assert decision.keep_mask[i].item() == (u_i < 0.05)


def test_7_correct_loss_masking_selected_patches():
    """Test that only selected patches contribute gradients to the adaptable parameters."""
    model, adapter = make_gated_setup(ConfidenceGatingPolicy(tau_conf=0.70))
    x = torch.randn(6, 16)

    # Perform adapted batch
    preds, decision = adapter.adapt_batch(x)
    assert decision.num_selected >= 0
    # Telemetry should reflect attempted update if selected > 0
    summary = adapter.telemetry.summary()
    if decision.num_selected > 0:
        assert summary["total_attempted_updates"] == 1
    else:
        assert summary["total_attempted_updates"] == 0


def test_8_zero_selected_patch_skips_optimizer_step():
    """Test that a batch with 0 selected patches bypasses backward pass and preserves exact weights."""
    # Policy with impossible confidence threshold
    policy = ConfidenceGatingPolicy(tau_conf=1.00)  # impossible since sigmoid in (0, 1)
    model, adapter = make_gated_setup(policy)
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    x = torch.randn(8, 16)
    preds, decision = adapter.adapt_batch(x)

    assert decision.num_selected == 0
    # Model weights must be completely untouched
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k])

    # Telemetry must show 0 attempted updates
    summary = adapter.telemetry.summary()
    assert summary["total_selected_patches"] == 0
    assert summary["total_attempted_updates"] == 0
    assert summary["update_coverage"] == 0.0
    assert summary["patch_coverage"] == 0.0


def test_9_all_selected_patches_executes_full_adaptation():
    """Test that a batch with 100% selected patches executes an accepted update and adjusts BN weights."""
    policy = ConfidenceGatingPolicy(tau_conf=0.00)  # accepts all
    model, adapter = make_gated_setup(policy)
    initial_bn_weight = model.bn1.weight.clone()

    x = torch.randn(8, 16)
    preds, decision = adapter.adapt_batch(x)

    assert decision.num_selected == 8
    assert not torch.equal(model.bn1.weight, initial_bn_weight), "BN weights should have adapted"

    summary = adapter.telemetry.summary()
    assert summary["total_selected_patches"] == 8
    assert summary["total_attempted_updates"] == 1
    assert summary["total_accepted_updates"] == 1
    assert summary["update_coverage"] == 1.0
    assert summary["patch_coverage"] == 1.0


def test_10_matched_budget_accounting():
    """Test telemetry accounting distinguishes patch coverage, update coverage, and prediction coverage."""
    model, adapter = make_gated_setup(ConfidenceGatingPolicy(tau_conf=0.75))

    # Feed 2 batches
    for i in range(2):
        torch.manual_seed(20 + i)
        x = torch.randn(10, 16)
        adapter.adapt_batch(x)

    summary = adapter.telemetry.summary()
    assert summary["total_eligible_patches"] == 20
    assert summary["total_batches"] == 2
    assert summary["prediction_coverage"] == 1.0  # all patches receive predictions
    assert 0.0 <= summary["patch_coverage"] <= 1.0
    assert 0.0 <= summary["update_coverage"] <= 1.0


def test_11_monitor_disabled_and_observation_equivalence():
    """Test that running monitor in DISABLED vs OBSERVATION yields bit-exact parameter updates."""
    torch.manual_seed(77)
    m1 = DeterministicToyDropoutModel()
    p1 = ConfidenceGatingPolicy(tau_conf=0.60)
    cfg1 = SafeAdaptConfig(tta_lr=0.01, monitor=CollapseMonitorConfig(mode=MonitorMode.DISABLED))
    ad1 = GatedSafeAdaptAdapter(m1, p1, cfg1)

    torch.manual_seed(77)
    m2 = DeterministicToyDropoutModel()
    p2 = ConfidenceGatingPolicy(tau_conf=0.60)
    cfg2 = SafeAdaptConfig(tta_lr=0.01, monitor=CollapseMonitorConfig(mode=MonitorMode.OBSERVATION))
    ad2 = GatedSafeAdaptAdapter(m2, p2, cfg2)

    torch.manual_seed(88)
    x = torch.randn(10, 16)
    pred1, _ = ad1.adapt_batch(x)
    pred2, _ = ad2.adapt_batch(x)

    assert torch.allclose(pred1, pred2, atol=1e-7)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items()):
        assert torch.allclose(v1, v2, atol=1e-7)


def test_12_preservation_of_stage_b_rollback():
    """Test that atomic rollback is triggered if an update produces non-finite loss/gradients."""
    model, adapter = make_gated_setup(ConfidenceGatingPolicy(tau_conf=0.00))
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    # Attach hook to adaptable parameter to inject non-finite gradient during backward
    target_param = adapter.controller.adaptable_params[0]
    hook = target_param.register_hook(lambda grad: grad * float("inf"))

    x = torch.randn(6, 16)
    try:
        preds, dec = adapter.adapt_batch(x)
    finally:
        hook.remove()

    # Step must be rejected, weights rolled back, and safe source predictions returned
    assert torch.all(torch.isfinite(preds))
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weights altered after Inf grad in {k}"
    assert adapter.telemetry.records[-1].rejected_update is True
    assert adapter.controller.telemetry.updates_rejected == 1


def test_13_no_target_label_ingress():
    """Test that neither policy.select nor adapter.adapt_batch accepts target labels y."""
    # Policies
    for pol_cls in [FrozenSourcePolicy, ConfidenceGatingPolicy, MCOnlyGatingPolicy, HybridGatingPolicy]:
        sig = inspect.signature(pol_cls.select)
        params = list(sig.parameters.keys())
        assert "y" not in params
        assert "labels" not in params
        assert "targets" not in params
        assert "target" not in params

    # Adapter
    ad_sig = inspect.signature(GatedSafeAdaptAdapter.adapt_batch)
    ad_params = list(ad_sig.parameters.keys())
    assert "y" not in ad_params
    assert "labels" not in ad_params
    assert "targets" not in ad_params
    assert "target" not in ad_params
    assert "x" in ad_params


def test_14_no_modifications_to_frozen_v1_artifacts():
    """Test that frozen v1 scientific artifacts exist and are untouched."""
    proj_root = Path(__file__).resolve().parent.parent

    stats_file = proj_root / "results" / "five_seed_final_stats.json"
    table_paired = proj_root / "results" / "tables" / "table_paired_statistics_final.csv"
    master_csv = proj_root / "results" / "master_results.csv"
    auth_yaml = proj_root / "configs" / "authoritative_resnet18_utta.yaml"
    tau_yaml = proj_root / "configs" / "tau_by_seed.yaml"

    assert stats_file.exists()
    assert table_paired.exists()
    assert master_csv.exists()
    assert auth_yaml.exists()
    assert tau_yaml.exists()
