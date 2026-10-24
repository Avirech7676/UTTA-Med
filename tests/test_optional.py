"""Unit tests for remaining optional methods (no Camelyon17 required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_temperature_moves_overconfident():
    from src.calibration.temperature import TemperatureScaler

    # Overconfident wrong-ish scores: true 0/1, probs 0.99/0.01 mixed
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([4.0, 4.0, -4.0, 4.0])  # three overconfident
    sc = TemperatureScaler().fit(logits, y, max_iter=50)
    assert sc.temperature > 1.0


def test_sar_updates_only_bn_and_can_skip():
    from src.models.resnet18 import build_resnet18
    from src.tta.sar import SAR

    model = build_resnet18(pretrained=False)
    sar = SAR(model, lr=1e-2, steps=1, e_margin=0.0)  # reject all
    x = torch.randn(4, 3, 96, 96)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    sar.adapt_batch(x)
    assert sar.n_adapted == 0
    for n, p in model.named_parameters():
        if "classifier" in n:
            assert torch.equal(p.detach(), before[n])


def test_eatac_empty_skip():
    from src.models.resnet18 import build_resnet18
    from src.tta.eata import EATAC

    model = build_resnet18(pretrained=False)
    ad = EATAC(model, lr=1e-2, e_margin=-1.0)  # nothing passes entropy
    x = torch.randn(4, 3, 96, 96)
    ad.adapt_batch(x)
    assert ad.n_adapted == 0
    assert ad.coverage == 0.0


def test_dltta_coverage_one():
    from src.models.resnet18 import build_resnet18
    from src.tta.dltta import DLTTA

    model = build_resnet18(pretrained=False)
    ad = DLTTA(model, lr=1e-3, steps=1)
    x = torch.randn(4, 3, 96, 96)
    out = ad.adapt_batch(x)
    assert out.shape == (4,)
    assert ad.coverage == 1.0


def test_resnet50_forward():
    from src.models.resnet50 import build_resnet50

    m = build_resnet50(pretrained=False)
    y = m(torch.randn(2, 3, 96, 96))
    assert y.shape == (2, 1)


def test_corruptions_preserve_shape():
    from src.data.corruptions import corrupt_tensor

    x = torch.rand(3, 3, 32, 32)
    y = corrupt_tensor(x, "gaussian_noise", 2)
    assert y.shape == x.shape
    y2 = corrupt_tensor(x, "jpeg", 3)
    assert y2.shape == x.shape


def test_build_adapter_optional_names():
    from src.models.resnet18 import build_resnet18
    from src.tta.uncertainty_gated import build_adapter

    m = build_resnet18(pretrained=False)
    for name in ("sar", "eata_c", "dltta"):
        ad = build_adapter(m, name, lr=1e-4)
        assert hasattr(ad, "adapt_batch")
