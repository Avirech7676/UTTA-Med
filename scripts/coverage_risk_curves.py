#!/usr/bin/env python3
"""Phase 1.2 — Coverage–risk / reject-option curves (zero GPU).

Ranks patches by low proxy-U (or high confidence) and keeps the lowest-U fraction.
Reports AUROC and error rate on the kept set.

Note: dumps store post-TTA probabilities, not MC variance. For binary labels,
U_proxy = 1 - |2p - 1| has the same ranking as confidence / entropy, which matches
the paper finding that confidence ≡ entropy on this task.

Writes:
  results/statistical/phase1_coverage_risk.json
  figures/phase1/fig_coverage_risk.png  (if matplotlib)
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from common import (
    OUT_FIG,
    OUT_STAT,
    SEEDS,
    auroc,
    ensure_dirs,
    error_rate,
    load,
    mean_sd,
    proxy_uncertainty,
    try_plot,
)

KEEP_FRACS: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
# Evaluate reject curves on source predictions (deployment reject option)
# and on post-TTA confidence-gated probs for comparison.
EVAL_METHODS = ["source", "confidence", "utta", "tent"]


def curve_for(y: np.ndarray, p: np.ndarray, rank_score: np.ndarray) -> Dict[str, dict]:
    """rank_score: lower = more reliable (keep first)."""
    order = np.argsort(rank_score)  # ascending: keep low-U first
    out = {}
    n = len(y)
    for kf in KEEP_FRACS:
        k = max(1, int(round(n * kf)))
        idx = order[:k]
        out[str(kf)] = {
            "keep_frac": kf,
            "n_kept": int(k),
            "auroc": auroc(y[idx], p[idx]),
            "error_rate": error_rate(y[idx], p[idx]),
        }
    return out


def main() -> None:
    ensure_dirs()
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = {}
        for method in EVAL_METHODS:
            d = load(seed, method)
            y, p = d["y"], d["p"]
            # Rank by source uncertainty when evaluating source reject option;
            # for adapted methods rank by their own predictive confidence.
            U = proxy_uncertainty(p)
            per_seed[str(seed)][method] = curve_for(y, p, U)

    # Aggregate mean±sd over seeds at each keep fraction
    summary = {}
    for method in EVAL_METHODS:
        summary[method] = {}
        for kf in KEEP_FRACS:
            aurocs = [per_seed[str(s)][method][str(kf)]["auroc"] for s in SEEDS]
            errs = [per_seed[str(s)][method][str(kf)]["error_rate"] for s in SEEDS]
            mu_a, sd_a = mean_sd(aurocs)
            mu_e, sd_e = mean_sd(errs)
            summary[method][str(kf)] = {
                "auroc_mean": mu_a,
                "auroc_sd": sd_a,
                "error_mean": mu_e,
                "error_sd": sd_e,
            }

    payload = {
        "protocol": (
            "Reject-option on stored probabilities; U_proxy = 1-|2p-1|. "
            "Not a new TTA run. Binary ranking ≡ confidence."
        ),
        "keep_fracs": KEEP_FRACS,
        "per_seed": per_seed,
        "summary_mean_sd": summary,
    }
    out_path = OUT_STAT / "phase1_coverage_risk.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print("wrote", out_path)

    print("\nSource reject-option AUROC (mean over seeds)")
    for kf in KEEP_FRACS:
        v = summary["source"][str(kf)]
        print(
            f"  keep={kf:.1f}  AUROC {v['auroc_mean']:.3f} ± {v['auroc_sd']:.3f}  "
            f"err {v['error_mean']:.3f} ± {v['error_sd']:.3f}"
        )

    plt = try_plot()
    if plt is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for method in EVAL_METHODS:
            xs = KEEP_FRACS
            ys = [summary[method][str(k)]["auroc_mean"] for k in xs]
            es = [summary[method][str(k)]["auroc_sd"] for k in xs]
            axes[0].errorbar(xs, ys, yerr=es, marker="o", label=method, capsize=2)
            ye = [summary[method][str(k)]["error_mean"] for k in xs]
            ee = [summary[method][str(k)]["error_sd"] for k in xs]
            axes[1].errorbar(xs, ye, yerr=ee, marker="o", label=method, capsize=2)
        axes[0].set_xlabel("Keep fraction (low-U first)")
        axes[0].set_ylabel("AUROC on kept set")
        axes[0].set_title("Coverage–performance")
        axes[0].invert_xaxis()
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0.5, 1.02)
        axes[1].set_xlabel("Keep fraction (low-U first)")
        axes[1].set_ylabel("Error rate on kept set")
        axes[1].set_title("Coverage–risk")
        axes[1].invert_xaxis()
        axes[1].legend(fontsize=8)
        fig.suptitle("Reject option from stored probs (binary U_proxy)")
        fig.tight_layout()
        fig_path = OUT_FIG / "fig_coverage_risk.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print("wrote", fig_path)


if __name__ == "__main__":
    main()
