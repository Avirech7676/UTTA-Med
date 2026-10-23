#!/usr/bin/env python3
"""Phase 1.3 — Tent collapse case study (seeds 7 and 99, per-slide damage).

Writes:
  results/statistical/phase1_tent_collapse.json
  results/tables/tent_collapse_per_slide.csv
  figures/phase1/fig_tent_slide_damage.png  (if matplotlib)
"""
from __future__ import annotations

import csv
import json
from typing import Dict, List

import numpy as np

from common import (
    COLLAPSE_SEEDS,
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


def per_slide_metrics(seed: int, method: str) -> List[dict]:
    d = load(seed, method)
    uniq = sorted(set(int(s) for s in d["slide"].tolist()))
    rows = []
    for sid in uniq:
        mask = d["slide"] == sid
        y, p = d["y"][mask], d["p"][mask]
        rows.append(
            {
                "seed": seed,
                "method": method,
                "slide": sid,
                "n": int(mask.sum()),
                "pos": int(y.sum()),
                "auroc": auroc(y, p),
                "f1": f1_score(y, p),
                "mean_p": float(p.mean()),
            }
        )
    return rows


def main() -> None:
    ensure_dirs()

    # Full per-slide for collapse seeds × all methods
    rows: List[dict] = []
    for seed in COLLAPSE_SEEDS:
        src_map: Dict[int, float] = {}
        for method in METHODS:
            for r in per_slide_metrics(seed, method):
                if method == "source":
                    src_map[r["slide"]] = r["auroc"]
                r["delta_vs_source"] = (
                    float(r["auroc"] - src_map[r["slide"]])
                    if method != "source"
                    and r["slide"] in src_map
                    and np.isfinite(r["auroc"])
                    and np.isfinite(src_map[r["slide"]])
                    else 0.0
                    if method == "source"
                    else float("nan")
                )
                rows.append(r)

    # Rank slides by Tent damage (most negative Δ)
    tent_damage = [
        r for r in rows if r["method"] == "tent" and np.isfinite(r["delta_vs_source"])
    ]
    tent_damage_sorted = sorted(tent_damage, key=lambda r: r["delta_vs_source"])

    # Headline: patch-level AUROC for collapse seeds
    patch_summary = {}
    for seed in COLLAPSE_SEEDS:
        patch_summary[str(seed)] = {}
        for method in METHODS:
            d = load(seed, method)
            patch_summary[str(seed)][method] = {
                "auroc": auroc(d["y"], d["p"]),
                "f1": f1_score(d["y"], d["p"]),
            }

    # Compare gated methods on the same seeds
    gated_rescue = {}
    for seed in COLLAPSE_SEEDS:
        gated_rescue[str(seed)] = {
            m: patch_summary[str(seed)][m]
            for m in ["source", "tent", "random", "confidence", "utta", "eata"]
        }

    # Slide 29 focus if present
    slide29 = [r for r in rows if r["slide"] == 29]

    payload = {
        "protocol": "CPU recompute from official npz; not a new TTA run",
        "collapse_seeds": COLLAPSE_SEEDS,
        "patch_level": patch_summary,
        "gated_rescue": gated_rescue,
        "worst_tent_slides": tent_damage_sorted[:10],
        "slide29": slide29,
        "note": (
            "Keep seeds 7 and 99 in the paper. Collapse is the result, not a bug. "
            "Gated methods should not show the same per-slide crater."
        ),
    }
    out_json = OUT_STAT / "phase1_tent_collapse.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print("wrote", out_json)

    csv_path = OUT_TABLES / "tent_collapse_per_slide.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "seed",
            "method",
            "slide",
            "n",
            "pos",
            "auroc",
            "f1",
            "mean_p",
            "delta_vs_source",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", csv_path)

    print("\nPatch-level AUROC on collapse seeds")
    for seed in COLLAPSE_SEEDS:
        print(f"  seed {seed}:")
        for m, v in patch_summary[str(seed)].items():
            print(f"    {m:12} AUROC={v['auroc']:.3f}  F1={v['f1']:.3f}")

    print("\nWorst Tent slides (largest AUROC drop vs source)")
    for r in tent_damage_sorted[:5]:
        print(
            f"  seed={r['seed']} slide={r['slide']}  "
            f"Δ={r['delta_vs_source']:+.3f}  auroc={r['auroc']:.3f}"
        )

    plt = try_plot()
    if plt is not None and tent_damage:
        # Grouped bar: per slide ΔAUROC for Tent vs confidence (seed 7 and 99)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, seed in zip(axes, COLLAPSE_SEEDS):
            tent = {
                r["slide"]: r["delta_vs_source"]
                for r in rows
                if r["seed"] == seed and r["method"] == "tent"
            }
            conf = {
                r["slide"]: r["delta_vs_source"]
                for r in rows
                if r["seed"] == seed and r["method"] == "confidence"
            }
            slides = sorted(tent.keys())
            x = np.arange(len(slides))
            w = 0.35
            ax.bar(
                x - w / 2,
                [tent[s] for s in slides],
                width=w,
                label="Tent",
                color="#C44E52",
            )
            ax.bar(
                x + w / 2,
                [conf[s] for s in slides],
                width=w,
                label="Confidence",
                color="#55A868",
            )
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in slides], rotation=45, ha="right")
            ax.set_title(f"Seed {seed}: ΔAUROC vs source")
            ax.set_xlabel("Slide ID")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("ΔAUROC (method − source)")
        fig.suptitle("Where Tent collapses — per-slide damage")
        fig.tight_layout()
        fig_path = OUT_FIG / "fig_tent_slide_damage.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("wrote", fig_path)


if __name__ == "__main__":
    main()
