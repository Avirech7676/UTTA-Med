#!/usr/bin/env python3
"""Phase 11 runner pointer.

The actual GPU job is notebooks/kaggle_resnet50_s42.py (Kaggle T4).
Local training of ResNet-50 on full Camelyon17 is not supported in this sandbox.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
print("Phase 11 — ResNet-50 sensitivity (code ready, run on Kaggle)")
print(" notebook:", HERE / "notebooks/kaggle_resnet50_s42.py")
print(" docs:", HERE / "docs/PHASE11_RESNET50.md")
print(" Frozen: bn_freeze_stats, lr=1e-5, N_MC=20, tau=70th pct val U, seed 42")
print(" Output: /kaggle/working/camelyon17_resnet50_full_s42.json")
raise SystemExit(
    "Run notebooks/kaggle_resnet50_s42.py on a GPU machine (Kaggle T4). "
    "Not a local CPU job."
)
