#!/usr/bin/env python3
"""Run TTA on the unlabeled target stream, then evaluate with held-out target labels.

Methods: tent | eata | eata_c | eata_f | sar | dltta | utta | random | confidence | entropy
τ / e_margin must be chosen on val (pass --tau after a val sweep). Never tune on test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--method",
        default="tent",
        choices=[
            "tent", "eata", "eata_c", "eata_f", "sar", "dltta",
            "utta", "random", "confidence", "entropy",
        ],
    )
    p.add_argument("--data-root", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument(
        "--tau",
        type=float,
        default=None,
        help="UTTA: 70th-pct val U (see configs/tau_by_seed.yaml). Random: coverage. NOT 0.05 for MC-Dropout.",
    )
    p.add_argument("--e-margin", type=float, default=0.4, help="EATA/SAR entropy filter")
    p.add_argument("--d-margin", type=float, default=0.4, help="EATA-C cosine diversity")
    p.add_argument("--n-passes", type=int, default=20)
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    return p.parse_args()


def load_model(ckpt_path: str, backbone: str = "resnet18"):
    if backbone == "resnet50":
        from src.models.resnet50 import build_resnet50

        model = build_resnet50(pretrained=False)
    else:
        from src.models.resnet18 import build_resnet18

        model = build_resnet18(pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  method={args.method}  split={args.split}")

    from src.data.camelyon17 import get_split, load_camelyon17, make_loader
    from src.evaluation.adaptation import flip_table
    from src.evaluation.classification import binary_metrics, brier, ece
    from src.tta.uncertainty_gated import build_adapter
    from src.training.train_loop import evaluate

    dataset, _, _ = load_camelyon17(data_root=args.data_root, download=False)
    subset = get_split(dataset, args.split, train=False)
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    source_model = load_model(args.checkpoint, args.backbone).to(device)
    before = evaluate(source_model, loader, device=device)

    if args.tau is None and args.method in ("utta", "utta_med", "mc_dropout"):
        import yaml

        tau_path = ROOT / "configs" / "tau_by_seed.yaml"
        tau_map = yaml.safe_load(tau_path.read_text())["tau"]
        args.tau = float(tau_map[args.seed])
        print(f"loaded frozen tau for seed {args.seed} = {args.tau:.6e} (70th pct val U)")

    model = load_model(args.checkpoint, args.backbone).to(device)
    adapter = build_adapter(
        model,
        args.method,
        lr=args.lr,
        steps=args.steps,
        tau=args.tau,
        e_margin=args.e_margin,
        d_margin=args.d_margin,
        n_passes=args.n_passes,
        seed=args.seed,
    )

    ys, p_before, p_after = [], [], []
    source_model.eval()
    for batch in loader:
        x, y = batch[0].to(device), batch[1].view(-1)
        with torch.no_grad():
            pb = torch.sigmoid(source_model(x)).view(-1).cpu().numpy()
        pa = adapter.adapt_batch(x).detach().cpu().numpy()
        ys.append(y.cpu().numpy())
        p_before.append(pb)
        p_after.append(pa)

    y_true = np.concatenate(ys)
    pb = np.concatenate(p_before)
    pa = np.concatenate(p_after)

    after_metrics = binary_metrics(y_true, pa)
    after_metrics["ece"] = ece(y_true, pa)
    after_metrics["brier"] = brier(y_true, pa)
    harm = flip_table(y_true, pb, pa)
    coverage = float(getattr(adapter, "coverage", 1.0 if args.method in ("tent", "dltta") else 0.0))
    if args.method in ("tent", "dltta"):
        coverage = 1.0

    summary = {
        "method": args.method,
        "split": args.split,
        "seed": args.seed,
        "tau": args.tau,
        "coverage": coverage,
        "n_adapted": int(getattr(adapter, "n_adapted", -1)),
        "n_seen": int(getattr(adapter, "n_seen", len(y_true))),
        "before": before,
        "after": after_metrics,
        "harm": harm,
    }
    out = ROOT / "results" / "metrics" / f"camelyon17_{args.backbone}_{args.method}_s{args.seed}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps({
        "method": args.method,
        "coverage": round(coverage, 4),
        "before_auroc": round(before["auroc"], 4),
        "after_auroc": round(after_metrics["auroc"], 4),
        "after_ece": round(after_metrics["ece"], 4),
        "harm_rate": round(harm["harm_rate"], 4),
        "correction_rate": round(harm["correction_rate"], 4),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
