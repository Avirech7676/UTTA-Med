#!/usr/bin/env python3
"""Comprehensive Coverage-Risk / Reject-Option Analysis for UTTA-Med.

Evaluates two distinct concepts:
1. Selective evaluation / reject option: Rank predictions by predictive uncertainty
   (ascending=True) and evaluate classification & calibration on the accepted subset.
2. Selective adaptation dynamics: Evaluate adaptation behavior (flip table: correction
   rate, harm rate, stability) on the accepted subset across coverage levels.

Outputs:
  - results/statistical/coverage_risk_detailed.json
  - results/tables/coverage_risk_summary.csv
  - figures/fig_coverage_risk_tradeoff.png
  - paper/ieee/figs/fig_coverage_risk_tradeoff.png
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

from src.evaluation.coverage_risk import (
    proxy_uncertainty,
    sweep_coverage_risk,
    sweep_selective_adaptation,
)

SEEDS = [42, 7, 2024, 123, 99]
METHODS = ["source", "confidence", "utta", "tent"]
COVERAGES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

NPZ_DIR = ROOT / "results" / "data" / "wsi_bootstrap"
OUT_JSON = ROOT / "results" / "statistical" / "coverage_risk_detailed.json"
OUT_CSV = ROOT / "results" / "tables" / "coverage_risk_summary.csv"
OUT_FIG = ROOT / "figures" / "fig_coverage_risk_tradeoff.png"
PAPER_FIG = ROOT / "paper" / "ieee" / "figs" / "fig_coverage_risk_tradeoff.png"


def load_npz(seed: int, method: str) -> dict:
    npz_path = NPZ_DIR / f"s{seed}_{method}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz dump: {npz_path}")
    data = np.load(npz_path)
    return {
        "y": data["y"].astype(int),
        "p": data["p"].astype(float),
        "p_before": data["p_before"].astype(float) if "p_before" in data else data["p"].astype(float),
        "slide": data["slide"] if "slide" in data else None,
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

    per_seed_eval: Dict[str, Dict[str, dict]] = {str(s): {} for s in SEEDS}
    per_seed_adapt: Dict[str, Dict[str, dict]] = {str(s): {} for s in SEEDS}

    for seed in SEEDS:
        for method in METHODS:
            d = load_npz(seed, method)
            y = d["y"]
            p = d["p"]
            p_before = d["p_before"]

            # Explicit ranking direction: lower uncertainty = accepted first (ascending=True)
            u = proxy_uncertainty(p)

            # 1. Selective evaluation / reject option
            per_seed_eval[str(seed)][method] = sweep_coverage_risk(
                y_true=y,
                y_prob=p,
                ranking_scores=u,
                coverages=COVERAGES,
                ascending=True,
            )

            # 2. Selective adaptation (before vs after dynamics on accepted subset)
            if method != "source":
                # For adaptation, ranking is determined by source uncertainty
                u_source = proxy_uncertainty(p_before)
                per_seed_adapt[str(seed)][method] = sweep_selective_adaptation(
                    y_true=y,
                    p_before=p_before,
                    p_after=p,
                    ranking_scores=u_source,
                    coverages=COVERAGES,
                    ascending=True,
                )

    # Summarize selective evaluation
    eval_summary: Dict[str, Dict[str, dict]] = {m: {} for m in METHODS}
    metrics_to_agg = [
        "n_accepted",
        "auroc",
        "error_rate",
        "accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "ece",
        "brier",
    ]

    csv_rows = []

    for method in METHODS:
        for cov in COVERAGES:
            cov_key = f"{cov:.2f}"
            eval_summary[method][cov_key] = {"coverage": cov}
            for met in metrics_to_agg:
                vals = [per_seed_eval[str(s)][method][cov_key][met] for s in SEEDS]
                stats = mean_std(vals)
                eval_summary[method][cov_key][f"{met}_mean"] = stats["mean"]
                eval_summary[method][cov_key][f"{met}_std"] = stats["std"]

            s_row = eval_summary[method][cov_key]
            csv_rows.append(
                {
                    "method": method,
                    "coverage": f"{cov:.2f}",
                    "n_accepted": int(round(s_row["n_accepted_mean"])),
                    "auroc": f"{s_row['auroc_mean']:.4f} ± {s_row['auroc_std']:.4f}",
                    "error_rate": f"{s_row['error_rate_mean'] * 100:.2f}% ± {s_row['error_rate_std'] * 100:.2f}%",
                    "accuracy": f"{s_row['accuracy_mean'] * 100:.2f}% ± {s_row['accuracy_std'] * 100:.2f}%",
                    "f1": f"{s_row['f1_mean']:.4f} ± {s_row['f1_std']:.4f}",
                    "ece": f"{s_row['ece_mean']:.4f} ± {s_row['ece_std']:.4f}",
                    "brier": f"{s_row['brier_mean']:.4f} ± {s_row['brier_std']:.4f}",
                }
            )

    # Write summary CSV
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "coverage",
                "n_accepted",
                "auroc",
                "error_rate",
                "accuracy",
                "f1",
                "ece",
                "brier",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote summary CSV to {OUT_CSV}")

    # Write detailed JSON
    payload = {
        "metadata": {
            "protocol": "Selective Evaluation (Reject Option) and Selective Adaptation sweep across 5 seeds on Camelyon17 test set",
            "ranking_direction": "ascending=True with proxy_uncertainty (1 - |2p - 1|)",
            "tie_breaking": "Deterministic secondary sort on original sample index via np.lexsort",
            "coverages": COVERAGES,
            "seeds": SEEDS,
            "methods": METHODS,
        },
        "selective_evaluation": {
            "summary": eval_summary,
            "per_seed": per_seed_eval,
        },
        "selective_adaptation": {
            "per_seed": per_seed_adapt,
        },
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote detailed JSON to {OUT_JSON}")

    # Plot 4-panel figure
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
        axes = axes.flatten()

        color_map = {
            "source": "#4a5568",
            "confidence": "#2b6cb0",
            "utta": "#2c7a7b",
            "tent": "#c53030",
        }
        label_map = {
            "source": "Source (Frozen)",
            "confidence": "Confidence-Gated",
            "utta": "UTTA-Med",
            "tent": "Tent (Unconstrained)",
        }

        # Panel 0: AUROC vs Coverage
        ax = axes[0]
        for m in METHODS:
            xs = [cov * 100 for cov in COVERAGES]
            ys = [eval_summary[m][f"{cov:.2f}"]["auroc_mean"] for cov in COVERAGES]
            es = [eval_summary[m][f"{cov:.2f}"]["auroc_std"] for cov in COVERAGES]
            ax.errorbar(
                xs, ys, yerr=es, marker="o", label=label_map[m], color=color_map[m], capsize=3, lw=1.8
            )
        ax.set_title("(A) AUROC vs Coverage (Reject Option)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coverage (% Patches Kept)", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_ylim(0.40, 1.02)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="lower left")

        # Panel 1: Error Rate vs Coverage
        ax = axes[1]
        for m in METHODS:
            xs = [cov * 100 for cov in COVERAGES]
            ys = [eval_summary[m][f"{cov:.2f}"]["error_rate_mean"] * 100 for cov in COVERAGES]
            es = [eval_summary[m][f"{cov:.2f}"]["error_rate_std"] * 100 for cov in COVERAGES]
            ax.errorbar(
                xs, ys, yerr=es, marker="s", label=label_map[m], color=color_map[m], capsize=3, lw=1.8
            )
        ax.set_title("(B) Error Rate (%) vs Coverage", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coverage (% Patches Kept)", fontsize=10)
        ax.set_ylabel("Error Rate (%)", fontsize=10)
        ax.set_ylim(0.0, 45.0)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

        # Panel 2: F1 Score vs Coverage
        ax = axes[2]
        for m in METHODS:
            xs = [cov * 100 for cov in COVERAGES]
            ys = [eval_summary[m][f"{cov:.2f}"]["f1_mean"] for cov in COVERAGES]
            es = [eval_summary[m][f"{cov:.2f}"]["f1_std"] for cov in COVERAGES]
            ax.errorbar(
                xs, ys, yerr=es, marker="^", label=label_map[m], color=color_map[m], capsize=3, lw=1.8
            )
        ax.set_title("(C) F1-Score vs Coverage", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coverage (% Patches Kept)", fontsize=10)
        ax.set_ylabel("F1 Score", fontsize=10)
        ax.set_ylim(0.40, 1.02)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="lower left")

        # Panel 3: ECE vs Coverage
        ax = axes[3]
        for m in METHODS:
            xs = [cov * 100 for cov in COVERAGES]
            ys = [eval_summary[m][f"{cov:.2f}"]["ece_mean"] for cov in COVERAGES]
            es = [eval_summary[m][f"{cov:.2f}"]["ece_std"] for cov in COVERAGES]
            ax.errorbar(
                xs, ys, yerr=es, marker="d", label=label_map[m], color=color_map[m], capsize=3, lw=1.8
            )
        ax.set_title("(D) Expected Calibration Error (ECE) vs Coverage", fontsize=11, fontweight="bold")
        ax.set_xlabel("Coverage (% Patches Kept)", fontsize=10)
        ax.set_ylabel("ECE", fontsize=10)
        ax.set_ylim(-0.02, 0.45)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=8.5, loc="upper left")

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
