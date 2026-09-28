#!/usr/bin/env python3
"""Figure 1 — UTTA-Med architecture (no GPU)."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/workspace/artifacts/UTTA-Med/figures/fig1_architecture.png")
OUT.parent.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fc="#1f4e79", tc="white", fs=8.2, lw=0.6):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=fc, edgecolor="#0d2137", linewidth=lw, zorder=2,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="medium", zorder=3,
            wrap=True, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#333"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=11,
        lw=1.15, color=color, zorder=1,
    ))


def main():
    fig, ax = plt.subplots(figsize=(11.2, 6.6), dpi=180)
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(5.6, 6.35, "UTTA-Med  ·  Camelyon17-WILDS  ·  hospital 2 held out",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0d2137")
    ax.text(5.6, 6.05, "Source {0,3,4}  →  tune on hospital 1  →  unlabeled TTA on hospital 2  →  evaluate once",
            ha="center", va="center", fontsize=8, color="#444")

    # row 1
    box(ax, (0.25, 4.85), 2.15, 0.85, "Camelyon17-WILDS\n96×96 patches\nbinary tumor / non-tumor",
        fc="#2c5f8a")
    box(ax, (2.75, 4.85), 2.15, 0.85, "Source train\nhospitals 0, 3, 4\n302k patches  ·  30 WSI",
        fc="#2c5f8a")
    box(ax, (5.25, 4.85), 2.15, 0.85, "ResNet-18  ·  1 logit\nBCE  ·  Dropout 0.5\nsave-best on hosp. 1",
        fc="#1b6b4a")
    box(ax, (7.75, 4.85), 3.15, 0.85, "Frozen source checkpoint\n5 seeds  {42, 123, 2024, 7, 99}\nBN affine only at TTA time",
        fc="#1b6b4a")
    arrow(ax, 2.40, 5.27, 2.75, 5.27)
    arrow(ax, 4.90, 5.27, 5.25, 5.27)
    arrow(ax, 7.40, 5.27, 7.75, 5.27)

    # row 2 — shift
    box(ax, (0.25, 3.55), 3.15, 0.85, "Target hospital 2  (unlabeled)\n85k patches  ·  10 WSI\nshift: AUROC drop 0.070 ± 0.009",
        fc="#8a3b2c")
    box(ax, (3.75, 3.55), 3.3, 0.85, "Ungated TTA  (failure mode)\nTent entropy min. on every sample\ncollapse on 2/5 seeds  ·  harm 0.19",
        fc="#8a3b2c")
    box(ax, (7.40, 3.55), 3.5, 0.85, "OOD val hospital 1 only\nτ / lr / coverage  —  never test labels\nτ = 70th percentile of U",
        fc="#6b4c1b")
    arrow(ax, 1.8, 4.85, 1.8, 4.40)
    arrow(ax, 3.40, 3.97, 3.75, 3.97)
    arrow(ax, 9.3, 4.85, 9.3, 4.40)

    # row 3 — gates
    ax.text(5.6, 3.28, "Hard gate   w(x) = 1[ U(x) < τ ]    ·    skip optimizer step if batch empty",
            ha="center", fontsize=8.3, color="#0d2137", fontweight="medium")
    gates = [
        (0.25, "Random", "#5c5c5c"),
        (3.05, "Confidence", "#3d5a80"),
        (5.85, "Entropy", "#3d5a80"),
        (8.65, "MC-Dropout U\n(UTTA-Med)", "#1b6b4a"),
    ]
    for x, name, c in gates:
        box(ax, (x, 2.25), 2.3, 0.78, name, fc=c, fs=9)
    arrow(ax, 5.4, 3.55, 5.4, 3.12)
    for x, _, _ in gates:
        arrow(ax, 5.4, 3.12, x + 1.15, 3.03)

    # row 4 — adapt / skip
    box(ax, (1.4, 1.05), 3.6, 0.78, "YES  ·  U < τ  ·  Adapt\nentropy min. on accepted samples\nBN γ, β only  ·  stats frozen",
        fc="#1b6b4a")
    box(ax, (6.2, 1.05), 3.6, 0.78, "NO  ·  U ≥ τ  ·  Skip update\nno backward, no optimizer step\nprediction still produced",
        fc="#6b2c2c")
    arrow(ax, 4.2, 2.25, 3.2, 1.83)
    arrow(ax, 7.0, 2.25, 8.0, 1.83)

    # row 5 — eval
    box(ax, (0.25, 0.12), 10.7, 0.72,
        "Hospital 2 evaluated once   ·   AUROC / AUPRC / F1 / ECE / Brier   ·   coverage (all gates)   ·   harm rate   ·   Grad-CAM audit\n"
        "Headline: gating cuts harm 0.187 → 0.007 and raises F1 on 5/5 seeds;  UTTA-Med ≈ confidence;  random still collapses",
        fc="#1f4e79", fs=8.0)
    arrow(ax, 3.2, 1.05, 3.2, 0.84)
    arrow(ax, 8.0, 1.05, 8.0, 0.84)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    # also copy to ieee figs
    ieee = Path("/workspace/artifacts/UTTA-Med/paper/ieee/figs/fig_architecture.png")
    fig.savefig(ieee, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
