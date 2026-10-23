#!/usr/bin/env python3
"""Evaluate a frozen checkpoint on val / test. Never trains. Never adapts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "id_val"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    from src.data.camelyon17 import get_split, load_camelyon17, make_loader
    from src.models.resnet18 import build_resnet18
    from src.training.train_loop import evaluate

    dataset, _, _ = load_camelyon17(data_root=args.data_root, download=False)
    subset = get_split(dataset, args.split, train=False)
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_resnet18(pretrained=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    metrics = evaluate(model, loader)
    print(json.dumps({"split": args.split, "n": metrics["n"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
