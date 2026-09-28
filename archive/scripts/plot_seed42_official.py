#!/usr/bin/env python3
"""Paper figures from frozen seed-42 official metrics (no GPU)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# --- official numbers (Gate B/C/D seed 42) ---
err_bins = [0.0009, 0.0023, 0.0046, 0.0086, 0.0138, 0.0223, 0.0370, 0.0946, 0.2287, 0.4092]
mean_U = [1.47e-11, 7.05e-10, 8.13e-9, 5.22e-8, 2.47e-7, 1.04e-6, 4.46e-6, 2.63e-5, 1.77e-4, 6.88e-4]

tau_pct = [20, 30, 40, 50, 60, 70, 80]
tau_cov = [0.201, 0.299, 0.401, 0.499, 0.601, 0.700, 0.800]
tau_auroc = [0.9846, 0.9850, 0.9851, 0.9858, 0.9862, 0.9867, 0.9858]
tau_f1 = [0.9326, 0.9339, 0.9341, 0.9351, 0.9363, 0.9370, 0.9347]
tau_harm = [0.0034, 0.0036, 0.0039, 0.0035, 0.0035, 0.0033, 0.0033]
src_val_auroc = 0.9793

methods = ["Source", "Tent", "EATA", "Random", "Confidence", "Entropy", "UTTA-Med"]
auroc = [0.9355, 0.9294, 0.9363, 0.9223, 0.9508, 0.9508, 0.9507]
f1 = [0.8350, 0.8248, 0.8447, 0.8051, 0.8708, 0.8708, 0.8697]
ece = [0.0988, 0.1396, 0.1250, 0.1534, 0.0921, 0.0921, 0.0930]
harm = [np.nan, 0.0591, 0.0514, 0.0617, 0.0029, 0.0029, 0.0032]
cov = [np.nan, 1.00, 0.94, 0.70, 0.57, 0.57, 0.56]
colors = ["#4C4C4C", "#D62728", "#FF7F0E", "#8C564B", "#2CA02C", "#17BECF", "#1F77B4"]


def fig_u_error():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(range(1, 11), [e * 100 for e in err_bins], color="#1F77B4", width=0.72)
    ax.set_xlabel("Uncertainty decile (low → high)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("RQ3  ·  MC-Dropout U vs error  (hospital 1, seed 42)")
    ax.set_xticks(range(1, 11))
    ax.text(0.98, 0.92, "Pearson r = 0.42", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_uncertainty_vs_error.png")
    plt.close()


def fig_tau():
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
    axes[0].plot(tau_pct, tau_cov, "o-", color="#1F77B4")
    axes[0].axvline(70, ls="--", color="gray", lw=1)
    axes[0].set_xlabel("τ percentile"); axes[0].set_ylabel("Coverage"); axes[0].set_title("Coverage")
    axes[1].plot(tau_pct, tau_auroc, "o-", color="#2CA02C", label="UTTA-Med")
    axes[1].axhline(src_val_auroc, ls="--", color="gray", label="Source")
    axes[1].axvline(70, ls="--", color="gray", lw=1)
    axes[1].set_xlabel("τ percentile"); axes[1].set_ylabel("AUROC"); axes[1].set_title("OOD val AUROC")
    axes[1].legend(frameon=False, fontsize=8)
    axes[2].plot(tau_pct, [h * 100 for h in tau_harm], "o-", color="#D62728")
    axes[2].axvline(70, ls="--", color="gray", lw=1)
    axes[2].set_xlabel("τ percentile"); axes[2].set_ylabel("Harm rate (%)"); axes[2].set_title("Harm")
    fig.suptitle("τ selected at 70th percentile of val U (seed 42)", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_tau_tradeoff.png", bbox_inches="tight")
    plt.close()


def fig_main():
    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, vals, ylab, ylim in (
        (axes[0], auroc, "AUROC", (0.90, 0.96)),
        (axes[1], f1, "F1", (0.78, 0.89)),
        (axes[2], ece, "ECE (lower better)", (0.06, 0.17)),
    ):
        ax.bar(x, vals, color=colors, width=0.78)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=40, ha="right")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
    fig.suptitle("Camelyon17-WILDS target (hospital 2)  ·  seed 42", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_main_comparison.png", bbox_inches="tight")
    plt.close()


def fig_harm():
    names = methods[1:]
    vals = [h * 100 for h in harm[1:]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(names, vals, color=colors[1:], width=0.72)
    ax.set_ylabel("Harm rate  Correct→Wrong (%)")
    ax.set_title("Selective TTA reduces harmful updates")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(OUT / "fig8_harm_rate.png")
    plt.close()


if __name__ == "__main__":
    fig_u_error()
    fig_tau()
    fig_main()
    fig_harm()
    print("wrote", list(OUT.glob("fig*.png")))
