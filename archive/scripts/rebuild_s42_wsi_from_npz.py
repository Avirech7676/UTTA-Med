"""Rebuild the full seed-42 WSI bootstrap JSON from official npz dumps.

Inner loop is NumPy-only (no sklearn per bootstrap). n_boot=1000, seed=42.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = ROOT / "results" / "predictions"
OUT_A = ROOT / "results" / "statistical" / "wsi_bootstrap" / "camelyon17_wsi_bootstrap_s42.json"
OUT_B = ROOT / "results" / "metrics" / "camelyon17_wsi_bootstrap_s42.json"

COVERAGE = {
    "source": None,
    "tent": 1.0,
    "eata": 0.865,
    "random": 0.701,
    "confidence": 0.568,
    "entropy": 0.568,
    "utta": 0.562,
}

N_BOOT = 1000
RNG = np.random.default_rng(42)


def ece(y, p, n_bins=15):
    y = np.asarray(y).astype(np.float64)
    p = np.clip(np.asarray(p).astype(np.float64), 0, 1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    tot = 0.0
    n = len(y)
    for i in range(n_bins):
        if i < n_bins - 1:
            m = (p >= bins[i]) & (p < bins[i + 1])
        else:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        k = int(m.sum())
        if k == 0:
            continue
        tot += (k / n) * abs(p[m].mean() - y[m].mean())
    return float(tot)


def np_auroc(y, p):
    y = np.asarray(y).astype(np.int8)
    p = np.asarray(p)
    npos = int(y.sum())
    nneg = len(y) - npos
    if npos == 0 or nneg == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(y), dtype=np.float64)
    ranks[order] = np.arange(1, len(y) + 1, dtype=np.float64)
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def np_f1(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def point_metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    pred = (p >= 0.5).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auroc = float(roc_auc_score(y, p))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y, p))
    except ValueError:
        auprc = float("nan")
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(spec),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ece": ece(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "auroc": auroc,
        "auprc": auprc,
    }


def harm_block(y, p, p_before, name):
    if name == "source":
        return None
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= 0.5).astype(int)
    before = (np.asarray(p_before) >= 0.5).astype(int)
    orig_ok = before == y
    orig_bad = ~orig_ok
    harmful = int((orig_ok & (pred != y)).sum())
    correction = int((orig_bad & (pred == y)).sum())
    flips = int((pred != before).sum())
    n_bad = int(orig_bad.sum()) or 1
    return {
        "correction_rate": correction / n_bad,
        "harm_rate": harmful / len(y),
        "flip_rate": flips / len(y),
        "correction": correction,
        "harmful": harmful,
    }


def ci_from_boots(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 10:
        return None
    lo, hi = np.quantile(v, [0.025, 0.975])
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(len(v)),
        "n_slides": 10,
    }


def per_slide_rows(y, p, slides):
    rows = []
    for s in sorted(np.unique(slides)):
        m = slides == s
        ys, ps = y[m].astype(int), p[m]
        pred = (ps >= 0.5).astype(int)
        rows.append(
            {
                "slide": int(s),
                "n": int(m.sum()),
                "pos": int(ys.sum()),
                "auroc": np_auroc(ys, ps),
                "f1": np_f1(ys, pred),
                "acc": float((pred == ys).mean()),
            }
        )
    return rows


def bootstrap(y, p, p_before, slides):
    ids = np.array(sorted(np.unique(slides)))
    groups = {int(s): np.where(slides == s)[0] for s in ids}
    y = y.astype(np.int8, copy=False)
    aurocs = np.empty(N_BOOT)
    f1s = np.empty(N_BOOT)
    harms = np.empty(N_BOOT)
    draws = RNG.choice(ids, size=(N_BOOT, len(ids)), replace=True)
    for b in range(N_BOOT):
        idx = np.concatenate([groups[int(s)] for s in draws[b]])
        ys, ps = y[idx], p[idx]
        pred = (ps >= 0.5).astype(np.int8)
        aurocs[b] = np_auroc(ys, ps)
        f1s[b] = np_f1(ys, pred)
        before = (p_before[idx] >= 0.5).astype(np.int8)
        harms[b] = float(((before == ys) & (pred != ys)).mean())
    return aurocs, f1s, harms


def main():
    methods = {}
    for name in ["source", "tent", "eata", "random", "confidence", "entropy", "utta"]:
        z = np.load(NPZ_DIR / f"s42_{name}.npz")
        y, p, p_before, slide = z["y"], z["p"], z["p_before"], z["slide"]
        pt = point_metrics(y, p)
        aurocs, f1s, harms = bootstrap(y, p, p_before, slide)
        block = {
            "point": pt,
            "coverage": COVERAGE[name],
            "harm": harm_block(y, p, p_before, name),
            "ci_auroc": ci_from_boots(aurocs),
            "ci_f1": ci_from_boots(f1s),
            "ci_ece": None,
            "ci_harm": None if name == "source" else ci_from_boots(harms),
            "per_slide": per_slide_rows(y, p, slide),
        }
        methods[name] = block
        ci = block["ci_auroc"]
        print(
            f"{name:12s} AUROC={pt['auroc']:.4f}  "
            f"CI=[{ci['ci_low']:.3f},{ci['ci_high']:.3f}]  F1={pt['f1']:.4f}"
        )

    out = {
        "seed": 42,
        "checkpoint": "checkpoints/camelyon17_resnet18_source_s42_BEST.pt",
        "epoch": 3,
        "claimed_val_auroc": 0.9792571864971674,
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": 1e-05,
            "n_mc": 20,
            "tau_rule": "70th percentile of val U",
            "e_margin": 0.4,
            "bootstrap": "WSI/slide with replacement",
            "n_boot": N_BOOT,
            "note": "Rebuilt from official Kaggle dumps s42_*.npz. Not a new TTA run.",
        },
        "tau": 8.875383173290174e-06,
        "matched_coverage_val": 0.700263580105432,
        "conf_threshold": 0.9930745959281921,
        "entropy_threshold": 0.04133822023868561,
        "n_test_slides": 10,
        "methods": methods,
    }
    for dest in (OUT_A, OUT_B):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2))
        print("wrote", dest, dest.stat().st_size)


if __name__ == "__main__":
    main()
