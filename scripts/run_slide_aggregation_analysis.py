#!/usr/bin/env python3
"""Comprehensive Slide-Level Aggregation & Tumor Burden Estimation Analysis for UTTA-Med.

Evaluates whole-slide level clinical utility across all 5 seeds and 7 methods:
- Slide-level tumor burden estimation concordance (MAE, RMSE, Pearson r, Spearman rho)
- Slide-macro AUROC and F1 averages across WSIs
- Dissects Tent failure mode in whole-slide clinical burden estimation

Outputs:
  - results/statistical/slide_aggregation_detailed.json
  - results/tables/slide_burden_summary.csv
  - figures/fig_slide_burden_estimation.png
  - paper/ieee/figs/fig_slide_burden_estimation.png
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.slide_aggregation import (
    slide_burden_metrics,
    slide_macro_metrics,
    slide_tumor_burdens,
)

SEEDS = [42, 7, 2024, 123, 99]
METHODS = ["source", "tent", "eata", "random", "confidence", "entropy", "utta"]

NPZ_DIR = ROOT / "results" / "data" / "wsi_bootstrap"
OUT_JSON = ROOT / "results" / "statistical" / "slide_aggregation_detailed.json"
OUT_CSV = ROOT / "results" / "tables" / "slide_burden_summary.csv"
OUT_FIG = ROOT / "figures" / "fig_slide_burden_estimation.png"
PAPER_FIG = ROOT / "paper" / "ieee" / "figs" / "fig_slide_burden_estimation.png"


def load_npz(seed: int, method: str) -> dict:
    npz_path = NPZ_DIR / f"s{seed}_{method}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz dump: {npz_path}")
    data = np.load(npz_path)
    return {
        "y": data["y"].astype(int),
        "p": data["p"].astype(float),
        "slide": data["slide"],
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan")}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
    }


def run_analysis() -> dict:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)

    per_seed: Dict[str, Dict[str, dict]] = {str(s): {} for s in SEEDS}

    for seed in SEEDS:
        for method in METHODS:
            d = load_npz(seed, method)
            burdens = slide_tumor_burdens(d["y"], d["p"], d["slide"])
            b_metrics = slide_burden_metrics(d["y"], d["p"], d["slide"])
            m_metrics = slide_macro_metrics(d["y"], d["p"], d["slide"])

            per_seed[str(seed)][method] = {
                "burden_metrics": b_metrics,
                "macro_metrics": m_metrics,
                "burdens": {str(k): v for k, v in burdens.items()},
            }

    # Aggregate summary across seeds
    summary: Dict[str, dict] = {}
    csv_rows = []

    metrics_to_agg = [
        "mae_binary",
        "rmse_binary",
        "pearson_r_binary",
        "spearman_rho_binary",
        "mae_prob",
        "rmse_prob",
        "pearson_r_prob",
        "spearman_rho_prob",
    ]

    for method in METHODS:
        summary[method] = {}
        for m in metrics_to_agg:
            vals = [per_seed[str(s)][method]["burden_metrics"][m] for s in SEEDS]
            stats = mean_std(vals)
            summary[method][f"{m}_mean"] = stats["mean"]
            summary[method][f"{m}_std"] = stats["std"]

        # Macro patch metrics
        macro_aurocs = [per_seed[str(s)][method]["macro_metrics"]["macro_auroc"] for s in SEEDS]
        macro_f1s = [per_seed[str(s)][method]["macro_metrics"]["macro_f1"] for s in SEEDS]
        st_auroc = mean_std(macro_aurocs)
        st_f1 = mean_std(macro_f1s)
        summary[method]["macro_auroc_mean"] = st_auroc["mean"]
        summary[method]["macro_auroc_std"] = st_auroc["std"]
        summary[method]["macro_f1_mean"] = st_f1["mean"]
        summary[method]["macro_f1_std"] = st_f1["std"]

        csv_rows.append(
            {
                "method": method,
                "macro_auroc": f"{st_auroc['mean']:.4f} ± {st_auroc['std']:.4f}",
                "macro_f1": f"{st_f1['mean']:.4f} ± {st_f1['std']:.4f}",
                "burden_mae": f"{summary[method]['mae_binary_mean']:.4f} ± {summary[method]['mae_binary_std']:.4f}",
                "burden_rmse": f"{summary[method]['rmse_binary_mean']:.4f} ± {summary[method]['rmse_binary_std']:.4f}",
                "burden_pearson_r": f"{summary[method]['pearson_r_binary_mean']:.4f} ± {summary[method]['pearson_r_binary_std']:.4f}",
                "burden_spearman_rho": f"{summary[method]['spearman_rho_binary_mean']:.4f} ± {summary[method]['spearman_rho_binary_std']:.4f}",
            }
        )

    # Write CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "macro_auroc",
                "macro_f1",
                "burden_mae",
                "burden_rmse",
                "burden_pearson_r",
                "burden_spearman_rho",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote summary CSV to {OUT_CSV}")

    # Write JSON
    payload = {
        "metadata": {
            "protocol": "Whole-slide level aggregation and tumor burden estimation across 5 seeds on Camelyon17 test set",
            "slides_count": 10,
            "seeds": SEEDS,
            "methods": METHODS,
        },
        "summary": summary,
        "per_seed": per_seed,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote detailed JSON to {OUT_JSON}")

    # Plot multi-panel figure
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

        # Subplot 1: True vs Predicted Tumor Burden for Collapse Seed 99
        ax1 = axes[0]
        s_seed = "99"
        b_true = [per_seed[s_seed]["source"]["burdens"][str(sid)]["burden_true"] for sid in range(20, 30)]
        b_src = [per_seed[s_seed]["source"]["burdens"][str(sid)]["burden_pred_binary"] for sid in range(20, 30)]
        b_utta = [per_seed[s_seed]["utta"]["burdens"][str(sid)]["burden_pred_binary"] for sid in range(20, 30)]
        b_tent = [per_seed[s_seed]["tent"]["burdens"][str(sid)]["burden_pred_binary"] for sid in range(20, 30)]

        ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Ideal (1:1)")
        ax1.scatter(b_true, b_src, color="#4a5568", label=f"Source (r={per_seed[s_seed]['source']['burden_metrics']['pearson_r_binary']:.2f})", s=45, alpha=0.8)
        ax1.scatter(b_true, b_utta, color="#2c7a7b", label=f"UTTA-Med (r={per_seed[s_seed]['utta']['burden_metrics']['pearson_r_binary']:.2f})", s=45, alpha=0.8)
        ax1.scatter(b_true, b_tent, color="#c53030", marker="x", label=f"Tent (r={per_seed[s_seed]['tent']['burden_metrics']['pearson_r_binary']:.2f})", s=60, lw=2)

        ax1.set_title(f"(A) Slide Tumor Burden (Seed {s_seed})", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Ground Truth Tumor Fraction", fontsize=10)
        ax1.set_ylabel("Predicted Tumor Fraction", fontsize=10)
        ax1.set_xlim(-0.02, 0.95)
        ax1.set_ylim(-0.02, 0.95)
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(frameon=True, fontsize=8.5, loc="upper left")

        # Subplot 2: Tumor Burden Mean Absolute Error (MAE) across Methods (5-seed mean ± std)
        ax2 = axes[1]
        x_pos = np.arange(len(METHODS))
        mae_means = [summary[m]["mae_binary_mean"] for m in METHODS]
        mae_stds = [summary[m]["mae_binary_std"] for m in METHODS]
        colors = ["#4a5568", "#c53030", "#805ad5", "#dd6b20", "#2b6cb0", "#319795", "#2c7a7b"]

        bars = ax2.bar(x_pos, mae_means, yerr=mae_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.upper() if m != "source" else "Source" for m in METHODS], rotation=30, ha="right", fontsize=9)
        ax2.set_title("(B) Slide Burden Estimation Error (MAE)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Absolute Error (MAE)", fontsize=10)
        ax2.set_ylim(0, 0.35)
        ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

        # Subplot 3: Slide-Macro AUROC (5-seed mean ± std)
        ax3 = axes[2]
        macro_means = [summary[m]["macro_auroc_mean"] for m in METHODS]
        macro_stds = [summary[m]["macro_auroc_std"] for m in METHODS]

        ax3.bar(x_pos, macro_means, yerr=macro_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([m.upper() if m != "source" else "Source" for m in METHODS], rotation=30, ha="right", fontsize=9)
        ax3.set_title("(C) Slide-Macro AUROC", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Slide-Macro AUROC", fontsize=10)
        ax3.set_ylim(0.40, 1.0)
        ax3.grid(True, linestyle="--", alpha=0.5, axis="y")

        plt.tight_layout()
        plt.savefig(OUT_FIG, dpi=300)
        plt.close()
        print(f"Saved figure to {OUT_FIG}")

        shutil.copyfile(OUT_FIG, PAPER_FIG)
        print(f"Copied figure to {PAPER_FIG}")
    except Exception as e:
        print(f"Plotting failed: {e}")

    return payload


if __name__ == "__main__":
    run_analysis()
