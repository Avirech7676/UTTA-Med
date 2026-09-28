#!/usr/bin/env python3
"""Quick smoke test of core UTTA-Med components (no dataset download required)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_model():
    from src.models.resnet18 import build_resnet18

    print("[1] Building ResNet-18 ...")
    model = build_resnet18(pretrained=False, dropout_p=0.5)
    x = torch.randn(4, 3, 96, 96)
    logits = model(x)
    assert logits.shape == (4, 1), f"Unexpected logits shape {logits.shape}"
    print(f"    Forward OK — logits shape {tuple(logits.shape)}")

    print("[2] MC-Dropout ...")
    mean_p, unc = model.mc_predict(x, n_passes=5)
    assert mean_p.shape == (4,) and unc.shape == (4,)
    print(f"    MC-Dropout OK — mean_p range [{mean_p.min():.3f}, {mean_p.max():.3f}], "
          f"unc range [{unc.min():.5f}, {unc.max():.5f}]")


def test_tracker():
    from src.training.tracking import ExperimentTracker, make_experiment_id, get_git_commit

    print("[3] Experiment tracker ...")
    eid = make_experiment_id("camelyon17", "resnet18", "source", 42)
    tracker = ExperimentTracker(eid, config={"seed": 42, "method": "source"}, base_dir="experiments")
    tracker.log_metrics({"auroc": 0.0, "f1": 0.0})
    tracker.log_command("python scripts/smoke_test.py")
    run_dir = tracker.finish()
    print(f"    Tracker OK — run dir {run_dir}")
    print(f"    Git commit: {get_git_commit()}")


def test_imports():
    print("[0] Core imports ...")
    import torch
    import torchvision
    import wilds
    import numpy
    import yaml
    print(f"    torch {torch.__version__}, wilds {wilds.__version__}")


if __name__ == "__main__":
    print("=" * 50)
    print("UTTA-Med Smoke Test")
    print("=" * 50)
    test_imports()
    test_model()
    test_tracker()
    print("=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)
