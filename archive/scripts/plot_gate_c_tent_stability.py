"""Gate C figure from locked salvage JSON — no GPU.

Shows why bn_freeze_stats + lr=1e-5 is the official Tent, and why
bn_train Tent is a collapse mode, not a baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    Path("/workspace/attachments/camelyon17_resnet18_tent_eata_s42.json"),
    ROOT / "results/metrics/camelyon17_resnet18_tent_eata_s42.json",
]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    for p in CANDIDATES:
        if p.exists():
            return json.loads(p.read_text()), p
    raise FileNotFoundError("tent_eata s42 JSON not found")


def main():
    data, src = load()
    rows = data["val_tent_sweep"]
    src_auroc = data["source_ood"]["auroc"]

    freeze, train = [], []
    for r in rows:
        rec = dict(
            lr=r["lr"],
            auroc=r["after"]["auroc"],
            f1=r["after"]["f1"],
            harm=r["harm"]["harm_rate"],
        )
        (freeze if r["bn_mode"] == "bn_freeze_stats" else train).append(rec)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    def panel(ax, key, ylabel, ylim=None):
        for recs, color, marker, label in (
            (freeze, "#1f4e79", "o", "BN stats frozen (official)"),
            (train, "#c0392b", "s", "BN.train() (batch stats)"),
        ):
            xs = [r["lr"] for r in recs]
            ys = [r[key] for r in recs]
            ax.plot(xs, ys, marker=marker, color=color, lw=2, label=label)
        if key == "auroc":
            ax.axhline(src_auroc, color="#7f8c8d", ls="--", lw=1, label="source (no TTA)")
        ax.set_xscale("log")
        ax.set_xlabel("Tent learning rate")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8)

    panel(axes[0], "auroc", "OOD-val AUROC", (0.25, 1.02))
    panel(axes[1], "harm", "Harm rate", (0, 0.6))
    fig.suptitle("Gate C — Tent is usable only with frozen BN stats and lr = 1e-5", fontsize=11)
    fig.tight_layout()
    out = OUT / "fig_c_tent_stability.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out, "from", src)


if __name__ == "__main__":
    main()
