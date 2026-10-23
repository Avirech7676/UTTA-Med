#!/usr/bin/env python3
"""Diagnostic Investigation of Tent Collapse Mechanism on Camelyon17-WILDS.

Performs rigorous empirical characterization of unconstrained entropy minimization failure:
- Prediction distribution shifts (suppression vs inflation)
- Class-specific tumor sensitivity vs non-tumor specificity
- Slide-level damage localization across all 10 Center-2 WSIs
- Entropy and confidence dynamics
- Flip table decomposition (C -> W harm accounting)
- Confidence vs empirical correctness calibration across confidence bins

Important Limitation Note:
Prediction-level failure mode is characterized directly from stored predictions;
parameter-level adaptation dynamics (e.g. BN affine parameter trajectories) were
not recorded during test-time execution.

Outputs:
  - results/statistical/tent_collapse_detailed.json
  - results/tables/tent_collapse_summary.csv
  - figures/fig_tent_collapse_mechanism.png
  - paper/ieee/figs/fig_tent_collapse_mechanism.png
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.classification import binary_metrics
from src.evaluation.tent_collapse import (
    class_specific_stats,
    confidence_correctness_bins,
    entropy_confidence_stats,
    prediction_distribution_stats,
    slide_level_collapse_breakdown,
)

SEEDS = [42, 7, 2024, 123, 99]
COLLAPSE_SEEDS = [7, 99]
METHODS = ["source", "tent", "confidence", "utta"]

NPZ_DIR = ROOT / "results" / "data" / "wsi_bootstrap"
OUT_JSON = ROOT / "results" / "statistical" / "tent_collapse_detailed.json"
OUT_CSV = ROOT / "results" / "tables" / "tent_collapse_summary.csv"
OUT_FIG = ROOT / "figures" / "fig_tent_collapse_mechanism.png"
PAPER_FIG = ROOT / "paper" / "ieee" / "figs" / "fig_tent_collapse_mechanism.png"


def load_npz(seed: int, method: str) -> dict:
    npz_path = NPZ_DIR / f"s{seed}_{method}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing npz dump: {npz_path}")
    data = np.load(npz_path)
    return {
        "y": data["y"].astype(int),
        "p": data["p"].astype(float),
        "p_before": data["p_before"].astype(float) if "p_before" in data else data["p"].astype(float),
        "slide": data["slide"],
    }


def run_analysis() -> dict:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.parent.mkdir(parents=True, exist_ok=True)

    analysis_data: Dict[str, Dict[str, Any]] = {str(s): {} for s in SEEDS}

    for seed in SEEDS:
        for method in METHODS:
            d = load_npz(seed, method)
            y = d["y"]
            p = d["p"]
            p_before = d["p_before"]
            slides = d["slide"]

            # 1. Prediction distribution
            dist_before = prediction_distribution_stats(p_before)
            dist_after = prediction_distribution_stats(p)

            # 2. Class-specific stats
            class_before = class_specific_stats(y, p_before)
            class_after = class_specific_stats(y, p)

            # 3. Entropy & confidence
            ec_before = entropy_confidence_stats(p_before)
            ec_after = entropy_confidence_stats(p)

            # 4. Confidence vs correctness bins
            conf_bins_before = confidence_correctness_bins(y, p_before)
            conf_bins_after = confidence_correctness_bins(y, p)

            # 5. Slide-level localization & flip breakdown
            slide_breakdown = slide_level_collapse_breakdown(y, p_before, p, slides)

            # Overall patch metrics
            metrics_after = binary_metrics(y, p)
            metrics_before = binary_metrics(y, p_before)

            analysis_data[str(seed)][method] = {
                "metrics_before": metrics_before,
                "metrics_after": metrics_after,
                "distribution_before": dist_before,
                "distribution_after": dist_after,
                "class_before": class_before,
                "class_after": class_after,
                "entropy_confidence_before": ec_before,
                "entropy_confidence_after": ec_after,
                "confidence_calibration_before": conf_bins_before,
                "confidence_calibration_after": conf_bins_after,
                "slide_breakdown": slide_breakdown,
            }

    # Build Summary CSV focused on Seed 7, Seed 99, and overall averages
    csv_rows = []
    for seed in SEEDS:
        for method in METHODS:
            m_data = analysis_data[str(seed)][method]
            dist = m_data["distribution_after"]
            cl = m_data["class_after"]
            ec = m_data["entropy_confidence_after"]
            met = m_data["metrics_after"]

            # Total harms across all slides
            total_harm = sum(s["harm_count"] for s in m_data["slide_breakdown"])
            total_corr = sum(s["correction_count"] for s in m_data["slide_breakdown"])

            csv_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "auroc": f"{met['auroc']:.4f}",
                    "f1": f"{met['f1']:.4f}",
                    "accuracy": f"{met['accuracy'] * 100:.2f}%",
                    "tumor_sensitivity": f"{cl['tumor']['sensitivity'] * 100:.2f}%",
                    "non_tumor_specificity": f"{cl['non_tumor']['specificity'] * 100:.2f}%",
                    "mean_p": f"{dist['mean_p']:.4f}",
                    "frac_p_lt_001": f"{dist['frac_p_lt_001'] * 100:.2f}%",
                    "mean_entropy": f"{ec['mean_entropy']:.4f}",
                    "mean_confidence": f"{ec['mean_confidence']:.4f}",
                    "frac_high_conf": f"{ec['frac_high_confidence'] * 100:.2f}%",
                    "total_harm_flips": total_harm,
                    "total_corr_flips": total_corr,
                }
            )

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "method",
                "auroc",
                "f1",
                "accuracy",
                "tumor_sensitivity",
                "non_tumor_specificity",
                "mean_p",
                "frac_p_lt_001",
                "mean_entropy",
                "mean_confidence",
                "frac_high_conf",
                "total_harm_flips",
                "total_corr_flips",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote summary CSV to {OUT_CSV}")

    # Build detailed JSON
    payload = {
        "metadata": {
            "protocol": "Diagnostic Tent Collapse Mechanism Analysis across 5 seeds on Camelyon17 test set",
            "limitation": (
                "Prediction-level failure mode is characterized directly from stored predictions; "
                "exact parameter-level adaptation dynamics (e.g. BN affine parameter trajectories) "
                "were not directly recorded during test-time execution."
            ),
            "seeds": SEEDS,
            "collapse_seeds": COLLAPSE_SEEDS,
            "methods": METHODS,
        },
        "per_seed_analysis": analysis_data,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote detailed JSON to {OUT_JSON}")

    # Plot 4-panel diagnostic figure
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)

        # Panel 1: Prediction Probability Distribution on Collapse Seed 99
        ax1 = axes[0, 0]
        seed_target = "99"
        d_src = load_npz(99, "source")
        d_tent = load_npz(99, "tent")
        d_utta = load_npz(99, "utta")

        bins = np.linspace(0, 1, 30)
        ax1.hist(d_src["p"], bins=bins, alpha=0.5, label="Source", color="#4a5568", density=True)
        ax1.hist(d_utta["p"], bins=bins, alpha=0.5, label="UTTA-Med", color="#2c7a7b", density=True)
        ax1.hist(d_tent["p"], bins=bins, alpha=0.6, label="Tent (Collapsed)", color="#c53030", density=True)

        ax1.set_title("(A) Prediction Probability Shift (Seed 99)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted Probability p", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.set_yscale("log")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(frameon=True, fontsize=9)

        # Panel 2: Slide-Level Delta AUROC vs Tumor Burden (Seed 99)
        ax2 = axes[0, 1]
        s_tent = analysis_data[seed_target]["tent"]["slide_breakdown"]
        s_utta = analysis_data[seed_target]["utta"]["slide_breakdown"]

        burdens = [s["tumor_burden"] * 100 for s in s_tent]
        d_auroc_tent = [s["delta_auroc"] for s in s_tent]
        d_auroc_utta = [s["delta_auroc"] for s in s_utta]
        slide_ids = [s["slide"] for s in s_tent]

        ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax2.scatter(burdens, d_auroc_tent, color="#c53030", s=60, label="Tent ΔAUROC", marker="x", lw=2)
        ax2.scatter(burdens, d_auroc_utta, color="#2c7a7b", s=50, label="UTTA-Med ΔAUROC", marker="o")

        for i, sid in enumerate(slide_ids):
            if sid in [28, 29]:
                ax2.annotate(
                    f"Slide {sid}",
                    (burdens[i], d_auroc_tent[i]),
                    textcoords="offset points",
                    xytext=(-20, -15),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#c53030",
                )

        ax2.set_title("(B) Slide Degradation vs Tumor Burden (Seed 99)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Slide Ground-Truth Tumor Burden (%)", fontsize=10)
        ax2.set_ylabel("ΔAUROC vs Source", fontsize=10)
        ax2.set_ylim(-0.90, 0.15)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(frameon=True, fontsize=9, loc="lower left")

        # Panel 3: Tumor Sensitivity vs Non-Tumor Specificity across Methods (Seed 99)
        ax3 = axes[1, 0]
        methods_plot = ["source", "confidence", "utta", "tent"]
        sens_vals = [
            analysis_data[seed_target][m]["class_after"]["tumor"]["sensitivity"] * 100
            for m in methods_plot
        ]
        spec_vals = [
            analysis_data[seed_target][m]["class_after"]["non_tumor"]["specificity"] * 100
            for m in methods_plot
        ]

        x_ind = np.arange(len(methods_plot))
        width = 0.35

        ax3.bar(x_ind - width / 2, sens_vals, width, label="Tumor Sensitivity", color="#e53e3e", alpha=0.85)
        ax3.bar(x_ind + width / 2, spec_vals, width, label="Non-Tumor Specificity", color="#3182ce", alpha=0.85)

        ax3.set_xticks(x_ind)
        ax3.set_xticklabels(["Source", "Confidence", "UTTA-Med", "Tent"], fontsize=9.5)
        ax3.set_title("(C) Class-Specific Performance (Seed 99)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Percentage (%)", fontsize=10)
        ax3.set_ylim(0, 110)
        ax3.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax3.legend(frameon=True, fontsize=9, loc="lower left")

        # Panel 4: Critical Mechanism Test: Confidence vs Empirical Correctness (Seed 99)
        ax4 = axes[1, 1]
        tent_calib = analysis_data[seed_target]["tent"]["confidence_calibration_after"]
        utta_calib = analysis_data[seed_target]["utta"]["confidence_calibration_after"]
        src_calib = analysis_data[seed_target]["source"]["confidence_calibration_after"]

        bin_centers = [0.55, 0.65, 0.75, 0.85, 0.95]
        acc_tent = [b["accuracy"] * 100 for b in tent_calib]
        acc_utta = [b["accuracy"] * 100 for b in utta_calib]
        acc_src = [b["accuracy"] * 100 for b in src_calib]

        ax4.plot(bin_centers, [b * 100 for b in bin_centers], "k--", alpha=0.4, label="Ideal Calibration")
        ax4.plot(bin_centers, acc_src, marker="s", color="#4a5568", label="Source", lw=1.8)
        ax4.plot(bin_centers, acc_utta, marker="o", color="#2c7a7b", label="UTTA-Med", lw=1.8)
        ax4.plot(bin_centers, acc_tent, marker="^", color="#c53030", label="Tent (Pathological)", lw=1.8)

        ax4.set_title("(D) Confidence vs Empirical Accuracy (Seed 99)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Predictive Confidence Bin Center max(p, 1-p)", fontsize=10)
        ax4.set_ylabel("Empirical Accuracy (%)", fontsize=10)
        ax4.set_ylim(40, 102)
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.legend(frameon=True, fontsize=9, loc="lower left")

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
