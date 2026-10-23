"""Unit tests for Multi-Class Evidential TTA (Avenue 1 Implementation).

Verifies:
1. Subjective Logic belief-vacuity identity: sum(b_c) + u == 1.0.
2. Decoupling of epistemic vacuity from confidence on K=9 vs K=2.
3. Parameter isolation: only BatchNorm affine parameters updated during adaptation.
4. Empty batch skipping: no gradients or updates when gate rejects all samples.
5. Numerical stability of Dirichlet digamma loss under extreme values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.evidential_resnet import (
    NUM_CLASSES,
    build_evidential_resnet18,
    compute_evidential_quantities,
)
from src.tta.evidential import EvidentialTTA, compute_gate_rank_correlations


def test_subjective_logic_identity():
    """Verify Dempster-Shafer / Subjective Logic identity: sum(b_c) + u == 1.0."""
    torch.manual_seed(42)
    B = 16
    K = 9
    logits = torch.randn(B, K)

    ev = compute_evidential_quantities(logits, num_classes=K)
    belief = ev["belief"]   # shape (B, K)
    vacuity = ev["vacuity"] # shape (B,)

    total_belief = torch.sum(belief, dim=-1)
    unity = total_belief + vacuity

    # Must equal 1.0 within floating point precision
    assert torch.allclose(unity, torch.ones_like(unity), atol=1e-5), (
        f"Subjective logic unity violated: {unity}"
    )


def test_evidential_quantities_shapes_and_bounds():
    """Check tensor dimensions and mathematical bounds."""
    B, K = 8, 9
    logits = torch.randn(B, K) * 3.0

    ev = compute_evidential_quantities(logits, activation="softplus", num_classes=K)

    assert ev["evidence"].shape == (B, K)
    assert torch.all(ev["evidence"] >= 0.0), "Evidence must be non-negative"
    assert torch.all(ev["alpha"] >= 1.0), "Dirichlet alpha must be >= 1.0"
    assert torch.all(ev["S"] >= float(K)), "Total Dirichlet strength must be >= K"
    assert torch.all(ev["prob"] >= 0.0) and torch.all(ev["prob"] <= 1.0)
    assert torch.all(ev["vacuity"] > 0.0) and torch.all(ev["vacuity"] <= 1.0)
    assert torch.all(ev["confidence"] >= (1.0 / K))


def test_decoupling_binary_vs_multiclass():
    """Prove that epistemic vacuity decouples from confidence on K=9, breaking the binary trap."""
    np.random.seed(42)
    torch.manual_seed(42)
    N = 100

    # 1. Multi-class K=9: Construct synthetic samples with varied uncertainty profiles
    # (a) Confident samples: one dominant logit
    # (b) In-distribution ambiguous: two conflicting classes (high entropy, low vacuity)
    # (c) Out-of-distribution: flat low logits (high vacuity, high entropy)
    logits_mc = []
    for i in range(N):
        if i % 3 == 0:
            # Confident
            z = np.random.randn(9) * 0.5
            z[np.random.randint(0, 9)] += 5.0
        elif i % 3 == 1:
            # Aleatoric ambiguity (e.g. between class 7 and 8)
            z = np.random.randn(9) * 0.5
            z[7] += 4.0
            z[8] += 4.1
        else:
            # OOD flat artifact
            z = np.ones(9) * -3.0 + np.random.randn(9) * 0.1
        logits_mc.append(z)

    logits_mc_t = torch.tensor(np.array(logits_mc), dtype=torch.float32)
    ev_mc = compute_evidential_quantities(logits_mc_t, num_classes=9)

    corrs_9 = compute_gate_rank_correlations(
        ev_mc["confidence"].numpy(),
        ev_mc["vacuity"].numpy(),
        ev_mc["entropy"].numpy(),
    )

    # 2. Binary K=2 (Single-logit Camelyon17 benchmark setup)
    logits_b = np.random.randn(N) * 3.0
    p_b = 1.0 / (1.0 + np.exp(-logits_b))
    conf_b = np.maximum(p_b, 1.0 - p_b)
    u_binary = (p_b * (1.0 - p_b)) ** 2  # First-order Taylor variance U
    import scipy.stats
    rho_b, _ = scipy.stats.spearmanr(conf_b, u_binary)

    # In binary, confidence and uncertainty are strictly rank-equivalent (|rho| == 1.0)
    assert abs(rho_b) > 0.99, f"Expected binary rank collapse (|rho| > 0.99), got {rho_b}"

    # In multi-class K=9, vacuity decouples from confidence (|rho| drops significantly)
    assert abs(corrs_9["rho_conf_vs_vacuity"]) < 0.80, (
        f"Expected multi-class decoupling (|rho| < 0.80), got {corrs_9['rho_conf_vs_vacuity']}"
    )


def test_evidential_tta_updates_only_bn_affine():
    """Verify that EvidentialTTA adapts strictly BatchNorm affine parameters."""
    torch.manual_seed(42)
    model = build_evidential_resnet18(num_classes=9, pretrained=False)

    # Snapshot weights before adaptation
    conv1_before = model.stem[0].weight.clone()
    fc_before = model.fc.weight.clone()
    bn_weight_before = model.stem[1].weight.clone()

    ctrl = EvidentialTTA(
        model=model,
        lr=1e-3,  # larger lr for test to see clear delta
        tau_vacuity=0.99,  # accept all
        gate_type="vacuity",
    )

    x = torch.randn(4, 3, 224, 224)
    _, info = ctrl.adapt_batch(x)
    assert info["adapted"] is True

    # Conv and FC layers must remain bitwise identical
    assert torch.equal(model.stem[0].weight, conv1_before), "Conv weight was modified!"
    assert torch.equal(model.fc.weight, fc_before), "Linear FC weight was modified!"

    # BN affine weight must have updated
    assert not torch.equal(model.stem[1].weight, bn_weight_before), "BN weight did not update!"


def test_evidential_empty_batch_skip():
    """Verify empty batch skipping when gate rejects all samples."""
    torch.manual_seed(42)
    model = build_evidential_resnet18(num_classes=9, pretrained=False)
    bn_before = model.stem[1].weight.clone()

    ctrl = EvidentialTTA(
        model=model,
        tau_vacuity=-1.0,  # Impossible threshold rejects all
        gate_type="vacuity",
    )

    x = torch.randn(4, 3, 224, 224)
    _, info = ctrl.adapt_batch(x)

    assert info["adapted"] is False
    assert info["n_accepted"] == 0
    assert ctrl.n_batches_skipped == 1
    # Parameters must be 100% unchanged
    assert torch.equal(model.stem[1].weight, bn_before)


def test_dirichlet_loss_numerical_stability():
    """Verify loss does not produce NaN or Inf under extreme logits."""
    B, K = 10, 9
    # Extreme logits: large positive, large negative, zero
    logits = torch.tensor([
        [100.0] * 9,
        [-100.0] * 9,
        [0.0] * 9,
        [50.0, -50.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ] * 2, dtype=torch.float32)

    ev = compute_evidential_quantities(logits, num_classes=K)
    model = build_evidential_resnet18(num_classes=9, pretrained=False)
    ctrl = EvidentialTTA(model=model, loss_type="dirichlet_entropy")

    loss = ctrl.compute_loss(ev["alpha"], ev["S"], ev["prob"])
    assert not torch.any(torch.isnan(loss)), "Loss contains NaN"
    assert not torch.any(torch.isinf(loss)), "Loss contains Inf"
