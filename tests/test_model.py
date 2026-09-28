from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.resnet18 import ResNet18UTTAMed, build_resnet18


def test_forward_and_mc_dropout():
    model = build_resnet18(pretrained=False, dropout_p=0.5)
    assert isinstance(model, ResNet18UTTAMed)
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    assert logits.shape == (4, 1)
    mean_p, unc = model.mc_predict(x, n_passes=5)
    assert mean_p.shape == (4,) and unc.shape == (4,)
    assert torch.all(mean_p >= 0) and torch.all(mean_p <= 1)
    assert torch.all(unc >= 0)
