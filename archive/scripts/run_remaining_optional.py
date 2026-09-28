#!/usr/bin/env python3
"""Remaining optional GPU job on a frozen ResNet-18 source checkpoint.

Fits temperature on hospital 1, then runs SAR / EATA-C / DLTTA / Camelyon17-C
on hospital 2. Never retunes τ, lr, or BN mode from the 5-seed headline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(ckpt: str, device):
    from src.models.resnet18 import build_resnet18

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_resnet18(pretrained=False)
    state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    model.load_state_dict(state)
    return model.to(device)


@torch.no_grad()
def source_probs(model, x):
    model.eval()
    return torch.sigmoid(model(x)).view(-1)


def adapt_stream(adapter, source, loader, device):
    from src.evaluation.adaptation import flip_table
    from src.evaluation.classification import binary_metrics, brier, ece

    ys, pb, pa = [], [], []
    for batch in tqdm(loader, desc=adapter.__class__.__name__):
        x, y = batch[0].to(device), batch[1].view(-1)
        p0 = source_probs(source, x).cpu().numpy()
        p1 = adapter.adapt_batch(x).detach().cpu().numpy()
        ys.append(y.cpu().numpy())
        pb.append(p0)
        pa.append(p1)
    y = np.concatenate(ys)
    b = np.concatenate(pb)
    a = np.concatenate(pa)
    after = binary_metrics(y, a)
    after["ece"] = ece(y, a)
    after["brier"] = brier(y, a)
    before = binary_metrics(y, b)
    before["ece"] = ece(y, b)
    before["brier"] = brier(y, b)
    return {
        "before": before,
        "after": after,
        "harm": flip_table(y, b, a),
        "coverage": float(getattr(adapter, "coverage", 1.0)),
        "n_reset": int(getattr(adapter, "n_reset", 0)),
        "n_skipped": int(getattr(adapter, "n_skipped", getattr(adapter, "n_skipped_batches", 0))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--e-margin", type=float, default=0.4)
    p.add_argument("--max-c-patches", type=int, default=8000)
    p.add_argument("--out", default="camelyon17_remaining_optional_s42.json")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    from src.calibration.temperature import TemperatureScaler
    from src.data.camelyon17 import get_split, load_camelyon17, make_loader
    from src.data.corruptions import DEFAULT_CORRUPTIONS, corrupt_tensor
    from src.evaluation.classification import binary_metrics, brier, ece
    from src.tta.dltta import DLTTA
    from src.tta.eata import EATAC
    from src.tta.sar import SAR
    from src.training.train_loop import evaluate

    dataset, _, _ = load_camelyon17(download=True)
    val = get_split(dataset, "val", train=False)
    test = get_split(dataset, "test", train=False)
    val_loader = make_loader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = make_loader(test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    source = load_model(args.ckpt, device)
    source.eval()
    print("source val", evaluate(source, val_loader, device=device))
    print("source test", evaluate(source, test_loader, device=device))

    zs, ys = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="temp-val"):
            x, y = batch[0].to(device), batch[1].view(-1)
            zs.append(source(x).view(-1).cpu())
            ys.append(y.float().cpu())
    sc = TemperatureScaler().fit(torch.cat(zs), torch.cat(ys))
    print("fitted T", sc.temperature)

    zt, yt = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].view(-1)
            zt.append(source(x).view(-1).cpu())
            yt.append(y.float().cpu())
    z_test = torch.cat(zt)
    y_test = torch.cat(yt).numpy()
    p_raw = torch.sigmoid(z_test).numpy()
    p_t = sc.transform_prob(z_test).numpy()
    temp_block = {
        "T": sc.temperature,
        "fit_split": "val_ood",
        "raw": {**binary_metrics(y_test, p_raw), "ece": ece(y_test, p_raw), "brier": brier(y_test, p_raw)},
        "scaled": {**binary_metrics(y_test, p_t), "ece": ece(y_test, p_t), "brier": brier(y_test, p_t)},
    }
    print("temp ECE raw", temp_block["raw"]["ece"], "scaled", temp_block["scaled"]["ece"])

    out = {"cuda": torch.cuda.is_available(), "checkpoint": args.ckpt, "temperature": temp_block}

    for name, factory in [
        ("sar", lambda m: SAR(m, lr=args.lr, steps=1, e_margin=args.e_margin)),
        ("eata_c", lambda m: EATAC(m, lr=args.lr, steps=1, e_margin=args.e_margin)),
        ("dltta", lambda m: DLTTA(m, lr=args.lr, steps=1)),
    ]:
        m = load_model(args.ckpt, device)
        src = load_model(args.ckpt, device)
        adapter = factory(m)
        block = adapt_stream(adapter, src, test_loader, device)
        out[f"{name}_test"] = block
        a = block["after"]
        print(
            f"{name:8s} AUROC={a['auroc']:.4f} F1={a['f1']:.4f} ECE={a['ece']:.4f} "
            f"harm={block['harm']['harm_rate']:.4f} cov={block['coverage']:.3f}"
        )

    n_keep = 0
    xs, ys_c = [], []
    for batch in test_loader:
        xs.append(batch[0])
        ys_c.append(batch[1].view(-1))
        n_keep += batch[0].size(0)
        if n_keep >= args.max_c_patches:
            break
    X = torch.cat(xs, 0)[: args.max_c_patches]
    Y = torch.cat(ys_c, 0)[: args.max_c_patches].numpy()
    src = load_model(args.ckpt, device)
    src.eval()
    c_rows = []
    for name in DEFAULT_CORRUPTIONS:
        for sev in (1, 3):
            probs = []
            with torch.no_grad():
                for i in range(0, len(X), args.batch_size):
                    xb = corrupt_tensor(X[i : i + args.batch_size].to(device), name, sev)
                    probs.append(torch.sigmoid(src(xb)).view(-1).cpu().numpy())
            p = np.concatenate(probs)
            m = binary_metrics(Y, p)
            c_rows.append({"corruption": name, "severity": sev, "auroc": m["auroc"], "f1": m["f1"]})
            print("C", name, sev, "AUROC", round(m["auroc"], 4))
    out["camelyon17_c_source"] = c_rows

    path = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    dest = path / args.out
    dest.write_text(json.dumps(out, indent=2, default=str))
    print("wrote", dest)


if __name__ == "__main__":
    main()
