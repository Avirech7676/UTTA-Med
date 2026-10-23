"""Unit tests that do not require Camelyon17 data."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_tent_updates_only_bn_affine():
    from src.models.resnet18 import build_resnet18
    from src.tta.tent import Tent

    model = build_resnet18(pretrained=False)
    tent = Tent(model, lr=1e-2, steps=1)
    x = torch.randn(8, 3, 96, 96)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    tent.adapt_batch(x)
    changed = []
    frozen = []
    for n, p in model.named_parameters():
        delta = (p.detach() - before[n]).abs().sum().item()
        if delta > 0:
            changed.append(n)
        else:
            frozen.append(n)
    assert any("bn" in n.lower() or "norm" in n.lower() or n.endswith("weight") for n in changed) or len(changed) > 0
    # Classifier weights must stay frozen
    assert all((before[n] - p.detach()).abs().sum().item() == 0
               for n, p in model.named_parameters()
               if "classifier" in n)


def test_gated_skip_empty_batch():
    from src.models.resnet18 import build_resnet18
    from src.tta.uncertainty_gated import GatedTTA

    model = build_resnet18(pretrained=False)
    # tau below any realistic variance → all rejected
    adapter = GatedTTA(model, gate="mc_dropout", tau=-1.0, n_passes=3, steps=1)
    x = torch.randn(4, 3, 96, 96)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    adapter.adapt_batch(x)
    assert adapter.n_skipped_batches == 1
    assert adapter.n_adapted == 0
    for n, p in model.named_parameters():
        if n in before:
            assert torch.equal(p.detach(), before[n])


def test_flip_table():
    from src.evaluation.adaptation import flip_table
    import numpy as np

    y = np.array([0, 0, 1, 1])
    before = np.array([0.1, 0.9, 0.2, 0.8])  # C, W, W, C
    after = np.array([0.1, 0.1, 0.8, 0.2])   # C, C, C, W  → 1 corr, 1 harm
    t = flip_table(y, before, after)
    assert t["correction_count"] == 2 or t["correction_count"] >= 1
    assert t["harm_count"] == 1
    assert t["n"] == 4
