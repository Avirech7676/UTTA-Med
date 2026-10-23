#!/usr/bin/env python3
"""Re-score PCam TTA from an existing BEST.pt. Does not retrain.

Looks for:
  /kaggle/working/pcam_resnet18_s42_BEST.pt
  /kaggle/input/**/pcam_resnet18_s42_BEST.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from kaggle_pcam_camelyon16 import BS, DEVICE, SEED, TF_EV, load_pcam  # noqa: E402
from tta_kit import (  # noqa: E402
    N_MC,
    TAU_PCT,
    ResNetMC,
    entropy_bin,
    harm_bin,
    mc_bin,
    metrics_bin,
    predict_bin,
    seed_all,
    tta_bin,
)

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/tmp/pcam_repair")
OUT.mkdir(parents=True, exist_ok=True)


def find_ckpt():
    hits = list(Path("/kaggle/working").glob("pcam_resnet18_s42_BEST.pt")) if Path("/kaggle/working").exists() else []
    hits += list(Path("/kaggle/input").rglob("pcam_resnet18_s42_BEST.pt")) if Path("/kaggle/input").exists() else []
    hits += list(Path(".").rglob("pcam_resnet18_s42_BEST.pt"))
    if not hits:
        raise SystemExit("No pcam_resnet18_s42_BEST.pt. Add it as a Kaggle dataset or keep the training session.")
    print("ckpt", hits[0])
    return hits[0]


def main():
    seed_all(SEED)
    print("device", DEVICE, "repair-only, no retrain")
    ckpt = torch.load(find_ckpt(), map_location="cpu")
    st = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    _, val, test = load_pcam()
    va = DataLoader(val, batch_size=BS, shuffle=False, num_workers=2)
    te = DataLoader(test, batch_size=BS, shuffle=False, num_workers=2)

    def clone():
        m = ResNetMC("resnet18", 1).to(DEVICE)
        m._arch = "resnet18"
        m.load_state_dict(st)
        return m

    model = clone()
    yt, pt = predict_bin(model, te, DEVICE)
    src = metrics_bin(yt, pt)
    print("SOURCE test", {k: src[k] for k in ("n", "auroc", "f1", "recall", "ece")})

    yv, pbar, u = mc_bin(model, va, DEVICE, n_mc=N_MC)
    tau = float(np.percentile(u, TAU_PCT))
    conf = np.maximum(pbar, 1.0 - pbar)
    ent = entropy_bin(pbar)
    cov = float((u < tau).mean())
    conf_t = float(np.quantile(conf, np.clip(1.0 - cov, 0, 1)))
    ent_t = float(np.quantile(ent, np.clip(cov, 0, 1)))
    from scipy.stats import spearmanr

    sp = float(spearmanr(u, 1.0 - conf).correlation)
    print("tau", tau, "cov", cov, "ent_t", ent_t, "Spearman", sp)
    if not np.isfinite(ent_t):
        raise SystemExit("ent_t not finite")

    rows = {"source_test": src}
    for gate, kw in [
        ("tent", {}),
        ("eata", {}),
        ("random", {"p_keep": cov}),
        ("confidence", {"conf_t": conf_t}),
        ("entropy", {"ent_t": ent_t}),
        ("utta", {"tau": tau}),
    ]:
        m = clone()
        y, _, b, c = tta_bin(m, te, DEVICE, gate, **kw)
        met = metrics_bin(y, b)
        h = harm_bin(yt, pt, b)
        rows[gate] = {**met, "coverage": c, "harm": h}
        print(
            f"{gate:12s} AUROC={met['auroc']:.4f} F1={met['f1']:.4f} "
            f"cov={c:.3f} harm={h['harm_rate']:.4f} nonfinite={met.get('n_nonfinite', 0)}"
        )
        if met.get("n_nonfinite", 0) > 0 or not np.isfinite(met["auroc"]):
            print("  REJECT this row")

    blob = {
        "dataset": "PCam / Camelyon16 patches",
        "seed": SEED,
        "repair": True,
        "tau": tau,
        "spearman_U_vs_1minus_conf": sp,
        "source_test": src,
        "results": rows,
    }
    path = OUT / "pcam_tta_repair_s42.json"
    path.write_text(json.dumps(blob, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
