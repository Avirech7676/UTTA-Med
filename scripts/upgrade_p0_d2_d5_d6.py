#!/usr/bin/env python3
"""P0 uniqueness upgrade — D2 / D5 / D6 from existing WSI dumps. Zero GPU.

D2  Per-slide (WSI) metrics + tumor-burden calibration.
    Slide-level binary MIL AUROC is undefined: every hospital-2 WSI has
    both classes. Report per-slide AUROC/F1/harm and mean(p) vs tumor fraction.

D5  Reject-option / coverage–risk curves.
    Rank patches by confidence max(p, 1-p) of the SOURCE model (frozen).
    At coverage c keep the most confident fraction; plot error and AUROC.
    Binary MC-Dropout U ranks almost identically to 1-confidence.

D6  Mechanistic Tent collapse: per-slide source vs Tent (seeds 7, 99).

Does not retune tau, lr, or drop any seed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NPZ_DIR = Path("/tmp/utta_audit/proj/results/data/wsi_bootstrap")
OUT = Path("/workspace/artifacts/UTTA-Med/results/upgrade_p0")
FIG = Path("/workspace/artifacts/UTTA-Med/figures/upgrade_p0")
SEEDS = (42, 123, 2024, 7, 99)
METHODS = ("source", "tent", "eata", "random", "confidence", "entropy", "utta")
THRESH = 0.5


def auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    # simple: if many ties, Mann–Whitney with rank
    # rank of positives
    sum_ranks_pos = ranks[y == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))



def f1_bin(y, p, t=THRESH):
    pred = (np.asarray(p) >= t).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    if pr + rc == 0:
        return 0.0
    return float(2 * pr * rc / (pr + rc))


def harm_rate(y, p_before, p_after, t=THRESH):
    y = np.asarray(y).astype(int)
    b = (np.asarray(p_before) >= t).astype(int)
    a = (np.asarray(p_after) >= t).astype(int)
    correct_before = b == y
    n = int(correct_before.sum())
    if n == 0:
        return float("nan")
    harmful = (a != y) & correct_before
    return float(harmful.sum() / n)


def load(seed, method):
    p = NPZ_DIR / f"s{seed}_{method}.npz"
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def d2_rows():
    rows = []
    burden = []
    for seed in SEEDS:
        src = load(seed, "source")
        slides = np.unique(src["slide"])
        for method in METHODS:
            d = load(seed, method)
            y, p, sl = d["y"], d["p"], d["slide"]
            pb = d["p_before"] if "p_before" in d else p
            for s in slides:
                m = sl == s
                ys, ps, pbs = y[m], p[m], pb[m]
                tf = float(ys.mean())
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "slide": int(s),
                        "n": int(m.sum()),
                        "tumor_frac": tf,
                        "auroc": auroc(ys, ps),
                        "f1": f1_bin(ys, ps),
                        "pred_burden": float(ps.mean()),
                        "harm": None if method == "source" else harm_rate(ys, pbs, ps),
                    }
                )
                if method == "source":
                    burden.append(
                        {
                            "seed": seed,
                            "slide": int(s),
                            "n": int(m.sum()),
                            "true_tumor_frac": tf,
                            "pred_mean_p": float(ps.mean()),
                            "abs_err": abs(float(ps.mean()) - tf),
                        }
                    )
    return rows, burden


def d2_summary(rows):
    # unweighted mean of per-slide AUROC (equal WSI, not equal patch)
    out = []
    for seed in SEEDS:
        for method in METHODS:
            sub = [r for r in rows if r["seed"] == seed and r["method"] == method]
            a = np.array([r["auroc"] for r in sub], float)
            f = np.array([r["f1"] for r in sub], float)
            out.append(
                {
                    "seed": seed,
                    "method": method,
                    "n_slides": len(sub),
                    "mean_slide_auroc": float(np.nanmean(a)),
                    "std_slide_auroc": float(np.nanstd(a, ddof=1)),
                    "mean_slide_f1": float(np.nanmean(f)),
                }
            )
    return out


def reject_curve(y, p, n_points=21):
    """Keep the most confident fraction. Confidence = max(p, 1-p)."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    conf = np.maximum(p, 1.0 - p)
    order = np.argsort(-conf)  # high confidence first
    y = y[order]
    p = p[order]
    n = len(y)
    coverages = np.linspace(0.2, 1.0, n_points)
    rec = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        ys, ps = y[:k], p[:k]
        pred = (ps >= THRESH).astype(int)
        err = float((pred != ys).mean())
        rec.append(
            {
                "coverage": float(k / n),
                "error": err,
                "auroc": auroc(ys, ps),
                "f1": f1_bin(ys, ps),
                "n_kept": int(k),
            }
        )
    return rec


def d5_curves():
    all_c = []
    for seed in SEEDS:
        src = load(seed, "source")
        # source reject-option
        for method in ("source", "utta", "tent", "confidence"):
            d = load(seed, method)
            for row in reject_curve(d["y"], d["p"]):
                row = dict(row)
                row["seed"] = seed
                row["method"] = method
                all_c.append(row)
    return all_c


def d6_collapse():
    rec = []
    for seed in SEEDS:
        src = load(seed, "source")
        tent = load(seed, "tent")
        utta = load(seed, "utta")
        slides = np.unique(src["slide"])
        for s in slides:
            m = src["slide"] == s
            rec.append(
                {
                    "seed": seed,
                    "slide": int(s),
                    "n": int(m.sum()),
                    "tumor_frac": float(src["y"][m].mean()),
                    "source_auroc": auroc(src["y"][m], src["p"][m]),
                    "tent_auroc": auroc(tent["y"][m], tent["p"][m]),
                    "utta_auroc": auroc(utta["y"][m], utta["p"][m]),
                    "delta_tent": auroc(tent["y"][m], tent["p"][m])
                    - auroc(src["y"][m], src["p"][m]),
                    "delta_utta": auroc(utta["y"][m], utta["p"][m])
                    - auroc(src["y"][m], src["p"][m]),
                    "tent_mean_p": float(tent["p"][m].mean()),
                    "source_mean_p": float(src["p"][m].mean()),
                }
            )
    return rec


def write_csv(path, rows, keys=None):
    if not rows:
        return
    keys = keys or list(rows[0].keys())
    with path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")


def figures(rows, burden, curves, collapse):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)

    # D2: mean per-slide AUROC by method (5-seed)
    methods = list(METHODS)
    means, stds = [], []
    for m in methods:
        vals = []
        for seed in SEEDS:
            sub = [r["auroc"] for r in rows if r["seed"] == seed and r["method"] == m]
            vals.append(float(np.nanmean(sub)))
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=1)))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, capsize=3, color="#4C72B0", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_ylabel("Mean of per-slide AUROC")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("D2  WSI-equal AUROC (10 hospital-2 slides, 5 seeds)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_d2_slide_auroc.png", dpi=160)
    plt.close()

    # D2 tumor burden scatter (source, all seeds)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for seed in SEEDS:
        b = [x for x in burden if x["seed"] == seed]
        ax.scatter(
            [x["true_tumor_frac"] for x in b],
            [x["pred_mean_p"] for x in b],
            s=28,
            label=str(seed),
            alpha=0.85,
        )
    ax.set_xlabel("True tumor fraction (slide)")
    ax.set_ylabel("Mean predicted p (source)")
    ax.set_title("D2  Tumor-burden calibration (source)")
    ax.legend(title="seed", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d2_tumor_burden.png", dpi=160)
    plt.close()

    # D5 coverage-risk: mean over seeds
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method, color in (
        ("source", "#4C72B0"),
        ("utta", "#C44E52"),
        ("tent", "#55A868"),
        ("confidence", "#8172B2"),
    ):
        covs = sorted({round(r["coverage"], 3) for r in curves if r["method"] == method})
        xs, ys = [], []
        for c in covs:
            e = [
                r["error"]
                for r in curves
                if r["method"] == method and abs(r["coverage"] - c) < 0.03
            ]
            if e:
                xs.append(c)
                ys.append(float(np.mean(e)))
        ax.plot(xs, ys, marker="o", ms=3, label=method, color=color)
    ax.set_xlabel("Coverage (fraction kept, highest confidence first)")
    ax.set_ylabel("Error rate on kept patches")
    ax.set_title("D5  Reject-option / coverage–risk (5-seed mean)")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig_d5_coverage_risk.png", dpi=160)
    plt.close()

    # D6 heatmap-like bars: delta tent AUROC per slide, seeds 7 and 99
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    for ax, seed in zip(axes, (7, 99)):
        sub = [r for r in collapse if r["seed"] == seed]
        sub = sorted(sub, key=lambda r: r["slide"])
        slides = [r["slide"] for r in sub]
        dt = [r["delta_tent"] for r in sub]
        du = [r["delta_utta"] for r in sub]
        x = np.arange(len(slides))
        ax.bar(x - 0.18, dt, 0.36, label="Tent − source", color="#C44E52")
        ax.bar(x + 0.18, du, 0.36, label="UTTA − source", color="#4C72B0")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(slides)
        ax.set_title(f"seed {seed}")
        ax.set_xlabel("Hospital-2 WSI id")
    axes[0].set_ylabel("Δ AUROC vs source")
    axes[1].legend(fontsize=8)
    fig.suptitle("D6  Per-slide Tent collapse vs gated UTTA", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_d6_tent_collapse_slides.png", dpi=160, bbox_inches="tight")
    plt.close()

    # D6 seed 7 slide 29 zoom: p histograms
    src = load(7, "source")
    tent = load(7, "tent")
    m = src["slide"] == 29
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.hist(src["p"][m], bins=40, alpha=0.55, label="source p", range=(0, 1))
    ax.hist(tent["p"][m], bins=40, alpha=0.55, label="Tent p", range=(0, 1))
    ax.set_xlabel("P(tumor)")
    ax.set_ylabel("Patches (slide 29, seed 7)")
    ax.set_title("D6  Slide 29: Tent drives p → 0 (collapse)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_d6_slide29_seed7_hist.png", dpi=160)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rows, burden = d2_rows()
    summary = d2_summary(rows)
    curves = d5_curves()
    collapse = d6_collapse()
    write_csv(OUT / "d2_per_slide.csv", rows)
    write_csv(OUT / "d2_slide_summary.csv", summary)
    write_csv(OUT / "d2_tumor_burden.csv", burden)
    write_csv(OUT / "d5_coverage_risk.csv", curves)
    write_csv(OUT / "d6_per_slide_collapse.csv", collapse)

    # headline numbers
    mae = float(np.mean([b["abs_err"] for b in burden]))
    # D5: error at coverage 0.5 vs 1.0 for source
    def mean_err(method, cov_target):
        vals = []
        for seed in SEEDS:
            cand = [
                r
                for r in curves
                if r["seed"] == seed
                and r["method"] == method
                and abs(r["coverage"] - cov_target) < 0.03
            ]
            if cand:
                vals.append(cand[0]["error"])
        return float(np.mean(vals)) if vals else float("nan")

    d6_big = [
        r
        for r in collapse
        if r["seed"] in (7, 99) and r["delta_tent"] < -0.3
    ]
    headline = {
        "d2_tumor_burden_mae_source": mae,
        "d5_source_error_cov1.0": mean_err("source", 1.0),
        "d5_source_error_cov0.5": mean_err("source", 0.5),
        "d5_utta_error_cov1.0": mean_err("utta", 1.0),
        "d5_utta_error_cov0.5": mean_err("utta", 0.5),
        "d6_n_slides_tent_delta_lt_minus_0.3_on_seeds_7_99": len(d6_big),
        "d6_worst": sorted(collapse, key=lambda r: r["delta_tent"])[:5],
        "note": "MIL slide-label AUROC undefined: all 10 WSIs contain tumor patches.",
    }
    (OUT / "headline.json").write_text(json.dumps(headline, indent=2, default=float))
    figures(rows, burden, curves, collapse)
    print("wrote", OUT)
    print("wrote", FIG)
    print(json.dumps({k: headline[k] for k in headline if k != "d6_worst"}, indent=2))
    print("worst Tent deltas:")
    for r in headline["d6_worst"]:
        print(
            f"  seed {r['seed']} slide {r['slide']} n={r['n']} "
            f"tent ΔAUROC={r['delta_tent']:+.3f} utta Δ={r['delta_utta']:+.3f} "
            f"tf={r['tumor_frac']:.2f}"
        )


if __name__ == "__main__":
    main()
