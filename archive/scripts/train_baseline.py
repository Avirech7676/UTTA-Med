#!/usr/bin/env python3
"""Source-only ResNet-18 baseline.

Train on source hospitals. Early-stop on OOD validation hospital.
Evaluate target hospital ONLY after training is frozen (no-adaptation baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--max-batches", type=int, default=None, help="Smoke-test cap. Do NOT use for paper runs.")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 20 train batches. Not a paper checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_batches = 20
        print("SMOKE MODE — checkpoint is not a source-only baseline. Do not adapt it.")
    set_seed(args.seed)

    from src.data.camelyon17 import get_split, load_camelyon17, make_loader
    from src.models.resnet18 import build_resnet18
    from src.training.tracking import ExperimentTracker, make_experiment_id
    from src.training.train_loop import evaluate, train_source

    cuda = torch.cuda.is_available()
    print(f"device = {'cuda' if cuda else 'cpu'}")
    if not cuda:
        print("NOTE: No CUDA GPU detected. Full 20-epoch training on CPU is ~30h. Use Colab T4.")
        print("      notebooks/01_colab_source_train.ipynb")

    dataset, wilds_root, version_dir = load_camelyon17(data_root=args.data_root, download=False)
    print(f"data root = {wilds_root}")
    print(f"version   = {version_dir}")

    train_set = get_split(dataset, "train", train=True)
    val_set = get_split(dataset, "val", train=False)
    test_set = get_split(dataset, "test", train=False)
    print(f"train={len(train_set)}  val(OOD)={len(val_set)}  test(target)={len(test_set)}")

    train_loader = make_loader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_resnet18(pretrained=args.pretrained, dropout_p=0.5)

    eid = make_experiment_id("camelyon17", "resnet18", "source", args.seed)
    if args.smoke or args.max_batches:
        eid = eid + "_smoke"
    tracker = ExperimentTracker(
        eid,
        config={
            "method": "source-only",
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "data_root": str(wilds_root),
            "smoke": bool(args.smoke or args.max_batches),
            "max_batches": args.max_batches,
        },
        base_dir=str(ROOT / "experiments"),
    )
    ckpt = ROOT / "checkpoints" / f"{eid}.pt"

    result = train_source(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        checkpoint_path=ckpt,
        use_amp=not args.no_amp,
        max_train_batches=args.max_batches,
        max_eval_batches=args.max_batches,
    )

    eval_cap = args.max_batches  # smoke eval is also capped; full run evaluates everything
    val_metrics = evaluate(model, val_loader, max_batches=eval_cap)
    test_metrics = evaluate(model, test_loader, max_batches=eval_cap)
    try:
        id_val = get_split(dataset, "id_val", train=False)
        id_loader = make_loader(id_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        id_metrics = evaluate(model, id_loader, max_batches=eval_cap)
    except Exception:
        id_metrics = None

    summary = {
        "experiment_id": eid,
        "checkpoint": str(ckpt),
        "smoke": bool(args.smoke or args.max_batches),
        "best_val_auroc": result["best_val_auroc"],
        "device": result.get("device"),
        "amp": result.get("amp"),
        "val_ood": val_metrics,
        "test_target": test_metrics,
        "id_val": id_metrics,
        "shift": None,
    }
    if id_metrics is not None:
        summary["shift"] = {
            "auroc_drop": id_metrics["auroc"] - test_metrics["auroc"],
            "auprc_drop": id_metrics.get("auprc", 0) - test_metrics.get("auprc", 0),
            "f1_drop": id_metrics["f1"] - test_metrics["f1"],
            "acc_drop": id_metrics["accuracy"] - test_metrics["accuracy"],
        }

    tracker.log_metrics(summary)
    tracker.finish()

    out = ROOT / "results" / "metrics" / f"{eid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== SOURCE-ONLY RESULTS ===")
    if summary["smoke"]:
        print("WARNING: smoke/max-batches run. Do not report these as the source baseline.")
    print(f"OOD val  AUROC={val_metrics['auroc']:.4f}  F1={val_metrics['f1']:.4f}  ECE={val_metrics['ece']:.4f}")
    print(f"TARGET   AUROC={test_metrics['auroc']:.4f}  F1={test_metrics['f1']:.4f}  ECE={test_metrics['ece']:.4f}")
    if id_metrics is not None:
        print(f"ID val   AUROC={id_metrics['auroc']:.4f}  F1={id_metrics['f1']:.4f}")
        print(f"SHIFT    AUROC drop={summary['shift']['auroc_drop']:.4f}")
    print(f"checkpoint → {ckpt}")
    print(f"metrics    → {out}")


if __name__ == "__main__":
    main()
