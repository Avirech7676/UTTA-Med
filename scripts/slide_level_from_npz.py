#!/usr/bin/env python3
"""Phase 1.1 — Slide-level AUROC / F1 and slide-macro summary (zero GPU).

Reads: results/data/wsi_bootstrap/s{seed}_{method}.npz
Writes:
  results/statistical/phase1_slide_level.json
  results/tables/slide_level_5seed.csv
  figures/phase1/fig_slide_macro_auroc.png  (if matplotlib)
"""
from __future__ import annotations

import csv
import json

import numpy as np

from common import (
    METHODS,
    OUT_FIG,
    OUT_STAT,
    OUT_TABLES,
    SEEDS,
    auroc,
    ensure_dirs,
    f1_score,
    load,
    mean_sd,
    try_plot,
)


def main() -> None:
    ensure_dirs()
    rows = []
    for seed in SEEDS:
        src = load(seed, "source")
        uniq = sorted(set(int(s) for s in src["slide"].tolist()))
        for method in METHODS:
            d = load(seed, method)
            for sid in uniq:
                mask = d["slide"] == sid
                y = d["y"][mask]
                p = d["p"][mask]
                rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "slide": sid,
                        "n": int(mask.sum()),
                        "pos": int(y.sum()),
                        "auroc": auroc(y, p),
                        "f1": f1_score(y, p),
                    }
                )

    # Slide-macro: mean over slides, then mean±sd over seeds
    slide_macro = {}
    for method in METHODS:
        per_seed = []
        for seed in SEEDS:
            vals = [
                r["auroc"]
                for r in rows
                if r["seed"] == seed
                and r["method"] == method
                and np.isfinite(r["auroc"])
            ]
            per_seed.append(float(np.mean(vals)) if vals else float("nan"))
        mu, sd = mean_sd(per_seed)
        slide_macro[method] = {
            "mean": mu,
            "sd": sd,
            "per_seed": {str(s): v for s, v in zip(SEEDS, per_seed)},
        }

    # ΔAUROC vs source per slide (Tent damage map)
    delta_rows = []
    for seed in SEEDS:
        src_by_slide = {
            r["slide"]: r["auroc"]
            for r in rows
            if r["seed"] == seed and r["method"] == "source"
        }
        for method in METHODS:
            if method == "source":
                continue
            for r in rows:
                if r["seed"] != seed or r["method"] != method:
                    continue
                base = src_by_slide.get(r["slide"], float("nan"))
                delta_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "slide": r["slide"],
                        "auroc": r["auroc"],
                        "delta_vs_source": (
                            float(r["auroc"] - base)
                            if np.isfinite(r["auroc"]) and np.isfinite(base)
                            else float("nan")
                        ),
                    }
                )

    payload = {
        "protocol": "CPU recompute from official npz; not a new TTA run",
        "n_rows": len(rows),
        "slide_macro_auroc": slide_macro,
        "per_slide": rows,
        "delta_vs_source": delta_rows,
    }
    out_json = OUT_STAT / "phase1_slide_level.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print("wrote", out_json)

    csv_path = OUT_TABLES / "slide_level_5seed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["seed", "method", "slide", "n", "pos", "auroc", "f1"]
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", csv_path)

    print("\nSlide-macro AUROC (mean over slides, then mean±sd over seeds)")
    for m in METHODS:
        v = slide_macro[m]
        print(f"  {m:12} {v['mean']:.3f} ± {v['sd']:.3f}")

    plt = try_plot()
    if plt is not None:
        labels = METHODS
        means = [slide_macro[m]["mean"] for m in labels]
        sds = [slide_macro[m]["sd"] for m in labels]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sds, capsize=3, color="#4C72B0")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Slide-macro AUROC")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Camelyon17 hospital 2 — slide-macro AUROC (n=5 seeds)")
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        fig.tight_layout()
        fig_path = OUT_FIG / "fig_slide_macro_auroc.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("wrote", fig_path)


if __name__ == "__main__":
    main()
