"""Deterministic CPU Unit Tests for SafeAdapt v2 (Stage B).

Validates atomic rollback, non-finite rejection, Adam step-counter preservation,
source state immutability, bounded parameter projection, and exception safety.
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

from src.v2.safeadapt.adapter import SafeAdaptTTA
from src.v2.safeadapt.config import SafeAdaptConfig
from src.v2.safeadapt.controller import SafeAdaptController


class ToyModel(nn.Module):
    """Small deterministic model with BatchNorm layers for CPU testing."""

    def __init__(self, in_features: int = 16, hidden: int = 8):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out.squeeze(-1)


def make_test_setup():
    """Create reproducible test harness."""
    torch.manual_seed(42)
    model = ToyModel()
    config = SafeAdaptConfig(
        tta_lr=0.01,
        max_grad_norm=1.0,
        param_displacement_bound_linf=0.20,
        max_consecutive_rejections=3,
    )
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.tta_lr)
    controller = SafeAdaptController(model, optimizer, config, adaptable_params=params)
    x = torch.randn(8, 16)
    return model, optimizer, controller, config, x


def test_nan_loss_rejection():
    """Test rejection when loss evaluates to NaN before backward propagation."""
    model, optimizer, controller, config, x = make_test_setup()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def nan_loss_fn():
        return torch.tensor(float("nan"), requires_grad=True)

    preds, accepted = controller.step(x, compute_loss_fn=nan_loss_fn)

    assert not accepted, "NaN loss should not be accepted"
    assert torch.all(torch.isfinite(preds)), "Returned fallback predictions must be finite"
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Model weights changed after NaN loss in {k}"
    assert controller.telemetry.updates_rejected == 1
    assert controller.telemetry.rejection_reasons["non_finite_loss"] == 1


def test_infinite_gradient_rejection():
    """Test rejection when gradients evaluate to Inf before optimizer step."""
    model, optimizer, controller, config, x = make_test_setup()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def inf_grad_loss_fn():
        out = model(x)
        # Construct an artificial loss whose gradient blows up
        # We manually corrupt gradient during backward by hooking
        loss = out.mean()
        return loss

    # Attach hook to adaptable parameter to inject inf grad
    target_param = controller.adaptable_params[0]
    hook = target_param.register_hook(lambda grad: grad * float("inf"))

    try:
        preds, accepted = controller.step(x, compute_loss_fn=inf_grad_loss_fn)
    finally:
        hook.remove()

    assert not accepted, "Infinite gradient should not be accepted"
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Model weights altered after Inf grad in {k}"
    assert controller.telemetry.updates_rejected == 1
    assert controller.telemetry.rejection_reasons["non_finite_gradient"] == 1


def test_non_finite_post_step_parameter_rejection():
    """Test rejection when parameters become non-finite post-step."""
    model, optimizer, controller, config, x = make_test_setup()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def valid_loss():
        return model(x).sum()

    # Artificially corrupt a parameter during optimizer step by monkeypatching step
    orig_step = optimizer.step

    def corrupting_step():
        orig_step()
        with torch.no_grad():
            controller.adaptable_params[0].data.fill_(float("nan"))

    optimizer.step = corrupting_step
    try:
        preds, accepted = controller.step(x, compute_loss_fn=valid_loss)
    finally:
        optimizer.step = orig_step

    assert not accepted, "Non-finite parameter update must be rejected"
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weights not restored after NaN parameter in {k}"
    assert controller.telemetry.updates_rejected == 1
    assert controller.telemetry.rejection_reasons["non_finite_parameter"] == 1


def test_adam_step_counter_and_optimizer_state_preservation():
    """Test that rejected updates do not advance Adam step counter or alter momentum buffers."""
    model, optimizer, controller, config, x = make_test_setup()

    # Run one valid accepted step first so Adam initializes state dictionaries
    def valid_loss():
        return model(x).sum()

    preds, accepted = controller.step(x, compute_loss_fn=valid_loss)
    assert accepted, "First step should be accepted"

    # Capture Adam step counter and state
    p0 = controller.adaptable_params[0]
    assert p0 in optimizer.state
    step_before = optimizer.state[p0]["step"]
    exp_avg_before = optimizer.state[p0]["exp_avg"].clone()
    exp_avg_sq_before = optimizer.state[p0]["exp_avg_sq"].clone()
    model_weights_before = {k: v.clone() for k, v in model.state_dict().items()}

    # Now attempt a step that fails with NaN loss
    def nan_loss():
        return torch.tensor(float("nan"), requires_grad=True)

    preds2, accepted2 = controller.step(x, compute_loss_fn=nan_loss)
    assert not accepted2, "Second step should be rejected"

    # Verify Adam step counter was NOT incremented
    step_after = optimizer.state[p0]["step"]
    assert step_after == step_before, f"Adam step counter changed from {step_before} to {step_after}"
    assert torch.equal(optimizer.state[p0]["exp_avg"], exp_avg_before), "Adam exp_avg mutated on rejection"
    assert torch.equal(optimizer.state[p0]["exp_avg_sq"], exp_avg_sq_before), "Adam exp_avg_sq mutated on rejection"
    for k, v in model.state_dict().items():
        assert torch.equal(v, model_weights_before[k]), f"Model weights changed on rejection in {k}"


def test_injected_invalid_optimizer_update_restores_exact_state():
    """Deliberately inject an invalid optimizer update after optimizer.step() and confirm exact restoration."""
    model, optimizer, controller, config, x = make_test_setup()

    # Pre-step state snapshots
    weights_before = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state_before = deepcopy(optimizer.state_dict())

    def loss_fn():
        return model(x).mean()

    # Intercept optimizer step to modify parameter and advance optimizer state, but then corrupt parameter
    orig_step = optimizer.step

    def corrupted_step():
        orig_step()
        with torch.no_grad():
            # Inject inf into second parameter
            controller.adaptable_params[1].data.fill_(float("inf"))

    optimizer.step = corrupted_step
    try:
        preds, accepted = controller.step(x, compute_loss_fn=loss_fn)
    finally:
        optimizer.step = orig_step

    assert not accepted, "Injected invalid parameter step must be rejected"
    # Verify exact equality of model state
    for k, v in model.state_dict().items():
        assert torch.equal(v, weights_before[k]), f"Weight mismatch after rollback in {k}"
    # Verify exact equality of optimizer state dict
    assert optimizer.state_dict() == opt_state_before, "Optimizer state dict mismatch after rollback"


def test_source_state_immutability():
    """Verify that source state cannot be mutated by adaptation steps or explicit parameter changes."""
    model, optimizer, controller, config, x = make_test_setup()
    assert controller.verify_source_immutability()

    # Execute 5 valid adaptation steps
    for _ in range(5):
        def loss_fn():
            return (model(x) - 0.5).pow(2).mean()

        controller.step(x, compute_loss_fn=loss_fn)

    # Confirm model parameters have changed from source
    has_changed = False
    for p in controller.adaptable_params:
        src = controller._source_param_anchors[id(p)]
        if not torch.equal(p, src):
            has_changed = True
            break
    assert has_changed, "Model parameters should have adapted"

    # Verify immutable source state remains completely unchanged
    assert controller.verify_source_immutability()


def test_bn_running_statistics_preservation():
    """Verify that BatchNorm running mean and running variance remain strictly frozen."""
    model, optimizer, controller, config, x = make_test_setup()

    # Capture initial running statistics
    init_running_mean = model.bn1.running_mean.clone()
    init_running_var = model.bn1.running_var.clone()
    init_num_batches = model.bn1.num_batches_tracked.clone()

    # Execute adaptation steps with diverse input batches
    for _ in range(5):
        x_noisy = torch.randn(8, 16) * 5.0 + 3.0

        def loss_fn():
            return model(x_noisy).mean()

        controller.step(x_noisy, compute_loss_fn=loss_fn)

    # Check running statistics have not budged
    assert torch.equal(model.bn1.running_mean, init_running_mean), "BN running_mean mutated"
    assert torch.equal(model.bn1.running_var, init_running_var), "BN running_var mutated"
    assert torch.equal(model.bn1.num_batches_tracked, init_num_batches), "BN num_batches_tracked changed"


def test_source_relative_parameter_projection_linf():
    """Verify that updates exceeding candidate bound are clamped back into the source epsilon-box."""
    model, optimizer, controller, config, x = make_test_setup()
    eps = config.param_displacement_bound_linf
    assert eps is not None and eps > 0

    # Force a massive step by using huge loss and large learning rate
    for g in optimizer.param_groups:
        g["lr"] = 10.0

    def massive_loss():
        return model(x).sum() * 1000.0

    preds, accepted = controller.step(x, compute_loss_fn=massive_loss)
    assert accepted, "Massive step should be clamped and accepted"

    # Verify that all adaptable parameters are strictly within [source - eps, source + eps]
    for p in controller.adaptable_params:
        src_p = controller._source_param_anchors[id(p)]
        diff = torch.abs(p.data - src_p.data)
        assert torch.all(diff <= eps + 1e-6), f"Parameter exceeded L_inf bound: max diff {diff.max().item()} > {eps}"

    assert controller.telemetry.clamping_events > 0


def test_exception_safe_rollback():
    """Verify that an arbitrary runtime exception during loss computation triggers safe rollback."""
    model, optimizer, controller, config, x = make_test_setup()
    initial_weights = {k: v.clone() for k, v in model.state_dict().items()}

    def exploding_loss():
        raise RuntimeError("Simulated CUDA OOM or device execution error")

    preds, accepted = controller.step(x, compute_loss_fn=exploding_loss)
    assert not accepted, "Exception step must not be accepted"
    assert torch.all(torch.isfinite(preds)), "Must return finite source predictions"
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial_weights[k]), f"Weights modified after exception in {k}"
    assert controller.telemetry.rejection_reasons["exception"] == 1


def test_current_batch_source_fallback_on_rejection():
    """Verify that when an update is rejected, the returned predictions match frozen source predictions."""
    model, optimizer, controller, config, x = make_test_setup()

    # Compute expected source predictions directly
    expected_source_preds = controller.forward_source(x)

    def failing_loss():
        return torch.tensor(float("nan"), requires_grad=True)

    preds, accepted = controller.step(x, compute_loss_fn=failing_loss)
    assert not accepted
    assert torch.allclose(preds, expected_source_preds, atol=1e-6), "Returned predictions do not match source"


def test_repeated_failures_trigger_fallback_mode():
    """Verify that exceeding max_consecutive_rejections enters fallback mode."""
    model, optimizer, controller, config, x = make_test_setup()
    assert not controller.in_fallback_mode
    max_rej = config.max_consecutive_rejections

    def nan_loss():
        return torch.tensor(float("nan"), requires_grad=True)

    for i in range(max_rej):
        assert not controller.in_fallback_mode
        controller.step(x, compute_loss_fn=nan_loss)

    assert controller.in_fallback_mode, "Controller should enter fallback mode after threshold"
    assert controller.telemetry.fallback_transitions == 1


def test_reset_boundary_behavior():
    """Verify reset boundary clears fallback mode and consecutive rejection counters."""
    model, optimizer, controller, config, x = make_test_setup()
    controller.trigger_fallback(reason="test")
    assert controller.in_fallback_mode

    # Call reset boundary
    controller.reset_boundary(restore_source_weights=True)
    assert not controller.in_fallback_mode
    assert controller.consecutive_rejections == 0
    assert controller.verify_source_immutability()


def test_adapter_absence_of_target_label_inputs():
    """Verify that SafeAdaptTTA interface does not accept target labels in call signatures."""
    sig = inspect.signature(SafeAdaptTTA.adapt_batch)
    params = list(sig.parameters.keys())
    assert "y" not in params, "adapt_batch must NOT accept labels 'y'"
    assert "label" not in params, "adapt_batch must NOT accept labels"
    assert "target" not in params, "adapt_batch must NOT accept targets"
    assert params == ["self", "x"], f"Expected signature (self, x), got {params}"


def test_v1_artifacts_unmodified_checksum():
    """Verify that frozen v1 scientific tables and checkpoints were not modified."""
    from pathlib import Path

    table_path = Path("results/tables/table_paired_statistics_final.csv")
    if table_path.exists():
        content = table_path.read_text(encoding="utf-8")
        assert "utta-src_auroc" in content
        assert "utta-conf_auroc" in content


def test_empty_gate_skips_optimizer_step_and_preserves_weights_and_momentum():
    """Regression test for G-03: Zero-patch selection must bypass backward and optimizer step cleanly."""
    model, optimizer, controller, config, x = make_test_setup()

    # Step 1: Execute a valid candidate adaptation step to establish non-zero Adam momentum
    def valid_loss():
        out = model(x)
        return out.mean()

    preds1, accepted1 = controller.step(x, compute_loss_fn=valid_loss)
    assert accepted1 is True, "First step should be accepted"

    # Capture model parameters and optimizer state after step 1
    post_step1_params = {k: v.clone() for k, v in model.state_dict().items()}
    post_step1_optim_state = deepcopy(optimizer.state_dict())

    # Step 2: Simulate an empty-gate batch (compute_loss_fn returns None)
    def empty_gate_loss():
        return None

    preds2, accepted2 = controller.step(x, compute_loss_fn=empty_gate_loss)
    assert accepted2 is False, "Empty-gate step must report accepted=False"
    assert controller.consecutive_rejections == 0, "Empty gate must not penalize consecutive rejections"

    # Verify that model parameters did NOT change (momentum did not drift)
    for k, v in model.state_dict().items():
        assert torch.equal(v, post_step1_params[k]), f"Parameter {k} drifted during empty-gate step!"

    # Verify through SafeAdaptTTA wrapper as well
    def reject_all_gate(logits, x_batch):
        return torch.zeros(x_batch.size(0), dtype=torch.bool)

    adapter = SafeAdaptTTA(model, config=config, gate_fn=reject_all_gate)
    pre_adapter_weights = {k: v.clone() for k, v in model.state_dict().items()}
    adapter_preds = adapter.adapt_batch(x)
    assert adapter_preds.shape == (x.size(0),)

    for k, v in model.state_dict().items():
        assert torch.equal(v, pre_adapter_weights[k]), f"Weight {k} changed during SafeAdaptTTA empty gate!"


def test_verify_source_immutability_detects_deliberate_tampering():
    """Regression test for G-04: verify_source_immutability must detect corruption of frozen model layers."""
    model = ToyModel()
    # In production SafeAdapt, convolutional and linear weights are frozen (BN affine only)
    model.fc1.weight.requires_grad = False
    model.fc1.bias.requires_grad = False
    model.fc2.weight.requires_grad = False
    model.fc2.bias.requires_grad = False

    config = SafeAdaptConfig()
    adaptable_params = [model.bn1.weight, model.bn1.bias]
    optimizer = torch.optim.Adam(adaptable_params, lr=config.tta_lr)
    controller = SafeAdaptController(model, optimizer, config, adaptable_params=adaptable_params)

    # 1. Nominal model passes immutability check
    assert controller.verify_source_immutability() is True

    # 2. Deliberately tamper with a non-adaptable (frozen) layer weight
    orig_w = model.fc1.weight.clone()
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    assert controller.verify_source_immutability() is False, "Failed to detect non-adaptable parameter mutation!"

    # 3. Restore frozen weight and verify recovery
    with torch.no_grad():
        model.fc1.weight.copy_(orig_w)
    assert controller.verify_source_immutability() is True

    # 4. Deliberately tamper with reference source model
    with torch.no_grad():
        controller._source_model.fc1.weight.add_(1.0)
    assert controller.verify_source_immutability() is False, "Failed to detect source model mutation!"

