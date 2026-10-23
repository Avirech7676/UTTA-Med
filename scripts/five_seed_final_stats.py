#!/usr/bin/env python3
"""Merge locked 5-seed results. Seed-level mean±SD, paired Δ, WSI Δ CIs, seeds 7/99."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path("/workspace/artifacts/UTTA-Med")
OUT = ROOT / "results" / "statistical" / "five_seed_final"
OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

HEAD = json.loads((ROOT / "results/tables/headline_5seed.json").read_text())
SEEDS = [42, 123, 2024, 7, 99]
ORDER = ["src", "tent", "eata", "rand", "conf", "utta"]
METHOD_NAME = {
    "src": "Source-only",
    "tent": "Tent",
    "eata": "EATA",
    "rand": "Random-gated",
    "conf": "Confidence-gated",
    "utta": "UTTA-Med",
}
BOOT_DIR = ROOT / "results/statistical/wsi_bootstrap"
NPZ_DIR = Path("/workspace/attachments")
N_BOOT = 1000
T_CRIT = float(stats.t.ppf(0.975, df=len(SEEDS) - 1))


def holm(pvals, names):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        a = min(1.0, pvals[idx] * (m - rank))
        running = max(running, a)
        adj[idx] = running
    return {names[i]: float(adj[i]) for i in range(m)}


def mean_sd(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def paired_delta_stats(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    m, s = mean_sd(d)
    se = s / np.sqrt(len(d))
    lo, hi = m - T_CRIT * se, m + T_CRIT * se
    try:
        w = stats.wilcoxon(d, zero_method="pratt", alternative="two-sided")
        wp, wstat = float(w.pvalue), float(w.statistic)
    except ValueError:
        wp, wstat = 1.0, 0.0
    t = stats.ttest_rel(np.asarray(a, float), np.asarray(b, float))
    d_cohen = m / s if s > 1e-12 else 0.0
    return {
        "mean_delta": m,
        "sd_delta": s,
        "ci95_low": lo,
        "ci95_high": hi,
        "t_stat": float(t.statistic),
        "t_p": float(t.pvalue),
        "wilcoxon_stat": wstat,
        "wilcoxon_p": wp,
        "cohens_d": float(d_cohen),
        "n_pos": int((d > 1e-12).sum()),
        "n_neg": int((d < -1e-12).sum()),
        "n_zero": int((np.abs(d) <= 1e-12).sum()),
        "per_seed": [float(x) for x in d],
    }


def bootstrap_paired_from_groups(y, pa, pb, groups, keys, n_boot, rng, want_f1):
    d_auroc = np.empty(n_boot)
    d_f1 = np.empty(n_boot) if want_f1 else None
    n_keys = len(keys)
    for i in range(n_boot):
        samp = rng.choice(n_keys, size=n_keys, replace=True)
        idx = np.concatenate([groups[keys[j]] for j in samp])
        yi, pai, pbi = y[idx], pa[idx], pb[idx]
        d_auroc[i] = roc_auc_score(yi, pai) - roc_auc_score(yi, pbi)
        if want_f1:
            d_f1[i] = f1_score(yi, pai >= 0.5) - f1_score(yi, pbi >= 0.5)
    out = {
        "delta_auroc": {
            "mean": float(d_auroc.mean()),
            "ci_low": float(np.quantile(d_auroc, 0.025)),
            "ci_high": float(np.quantile(d_auroc, 0.975)),
        },
        "n_boot": n_boot,
        "n_slides": int(n_keys),
        "unit": "pooled patches after WSI resample (seed-42 dumps)",
    }
    if want_f1:
        out["delta_f1"] = {
            "mean": float(d_f1.mean()),
            "ci_low": float(np.quantile(d_f1, 0.025)),
            "ci_high": float(np.quantile(d_f1, 0.975)),
        }
    return out


def bootstrap_paired_perslide(a, b, n_boot, rng, weights=None):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        ds[i] = np.average(a[idx] - b[idx], weights=w[idx])
    return {
        "mean": float(ds.mean()),
        "ci_low": float(np.quantile(ds, 0.025)),
        "ci_high": float(np.quantile(ds, 0.975)),
        "n_boot": n_boot,
        "n_slides": n,
    }


def main():
    rows = {int(k): v for k, v in HEAD["rows"].items()}
    metrics_across = {}
    for m in ORDER:
        for field in ("auroc", "f1", "ece"):
            xs = [rows[s][f"{m}_{field}"] for s in SEEDS]
            mu, sd = mean_sd(xs)
            metrics_across[f"{m}_{field}"] = {"mean": mu, "sd": sd, "values": xs}
        if m != "src":
            for field in ("harm", "cov"):
                xs = [rows[s][f"{m}_{field}"] for s in SEEDS]
                mu, sd = mean_sd(xs)
                metrics_across[f"{m}_{field}"] = {"mean": mu, "sd": sd, "values": xs}

    src_auroc = [rows[s]["src_auroc"] for s in SEEDS]
    utta_auroc = [rows[s]["utta_auroc"] for s in SEEDS]
    conf_auroc = [rows[s]["conf_auroc"] for s in SEEDS]
    tent_auroc = [rows[s]["tent_auroc"] for s in SEEDS]
    eata_auroc = [rows[s]["eata_auroc"] for s in SEEDS]
    rand_auroc = [rows[s]["rand_auroc"] for s in SEEDS]
    src_f1 = [rows[s]["src_f1"] for s in SEEDS]
    utta_f1 = [rows[s]["utta_f1"] for s in SEEDS]
    conf_f1 = [rows[s]["conf_f1"] for s in SEEDS]
    tent_f1 = [rows[s]["tent_f1"] for s in SEEDS]
    eata_f1 = [rows[s]["eata_f1"] for s in SEEDS]
    rand_f1 = [rows[s]["rand_f1"] for s in SEEDS]
    utta_harm = [rows[s]["utta_harm"] for s in SEEDS]
    conf_harm = [rows[s]["conf_harm"] for s in SEEDS]
    tent_harm = [rows[s]["tent_harm"] for s in SEEDS]
    eata_harm = [rows[s]["eata_harm"] for s in SEEDS]
    rand_harm = [rows[s]["rand_harm"] for s in SEEDS]
    utta_cov = [rows[s]["utta_cov"] for s in SEEDS]
    conf_cov = [rows[s]["conf_cov"] for s in SEEDS]
    rand_cov = [rows[s]["rand_cov"] for s in SEEDS]
    eata_cov = [rows[s]["eata_cov"] for s in SEEDS]
    src_sens = [rows[s]["src_sens"] for s in SEEDS]
    drop_auroc = [rows[s]["drop_auroc"] for s in SEEDS]
    u_r = [rows[s]["u_r"] for s in SEEDS]

    comparisons = {
        "utta-src_auroc": paired_delta_stats(utta_auroc, src_auroc),
        "utta-tent_auroc": paired_delta_stats(utta_auroc, tent_auroc),
        "utta-eata_auroc": paired_delta_stats(utta_auroc, eata_auroc),
        "utta-rand_auroc": paired_delta_stats(utta_auroc, rand_auroc),
        "utta-conf_auroc": paired_delta_stats(utta_auroc, conf_auroc),
        "conf-src_auroc": paired_delta_stats(conf_auroc, src_auroc),
        "utta-src_f1": paired_delta_stats(utta_f1, src_f1),
        "utta-tent_f1": paired_delta_stats(utta_f1, tent_f1),
        "utta-eata_f1": paired_delta_stats(utta_f1, eata_f1),
        "utta-rand_f1": paired_delta_stats(utta_f1, rand_f1),
        "utta-conf_f1": paired_delta_stats(utta_f1, conf_f1),
        "conf-src_f1": paired_delta_stats(conf_f1, src_f1),
        "utta-tent_harm": paired_delta_stats(utta_harm, tent_harm),
        "utta-eata_harm": paired_delta_stats(utta_harm, eata_harm),
        "utta-rand_harm": paired_delta_stats(utta_harm, rand_harm),
        "utta-conf_harm": paired_delta_stats(utta_harm, conf_harm),
        "conf-tent_harm": paired_delta_stats(conf_harm, tent_harm),
        "eata-src_auroc": paired_delta_stats(eata_auroc, src_auroc),
        "tent-src_auroc": paired_delta_stats(tent_auroc, src_auroc),
    }
    primary_p = {
        "UTTA vs Source AUROC": comparisons["utta-src_auroc"]["t_p"],
        "UTTA vs Tent AUROC": comparisons["utta-tent_auroc"]["t_p"],
        "UTTA vs EATA AUROC": comparisons["utta-eata_auroc"]["t_p"],
        "UTTA vs Random AUROC": comparisons["utta-rand_auroc"]["t_p"],
        "UTTA vs Confidence AUROC": comparisons["utta-conf_auroc"]["t_p"],
        "UTTA vs Source F1": comparisons["utta-src_f1"]["t_p"],
        "UTTA vs Tent harm": comparisons["utta-tent_harm"]["t_p"],
        "UTTA vs Confidence F1": comparisons["utta-conf_f1"]["t_p"],
    }
    holm_p = holm(list(primary_p.values()), list(primary_p.keys()))

    per_seed_delta = []
    for i, s in enumerate(SEEDS):
        per_seed_delta.append(
            {
                "seed": s,
                "d_auroc_utta_src": utta_auroc[i] - src_auroc[i],
                "d_auroc_utta_conf": utta_auroc[i] - conf_auroc[i],
                "d_auroc_utta_tent": utta_auroc[i] - tent_auroc[i],
                "d_auroc_utta_eata": utta_auroc[i] - eata_auroc[i],
                "d_auroc_utta_rand": utta_auroc[i] - rand_auroc[i],
                "d_f1_utta_src": utta_f1[i] - src_f1[i],
                "d_f1_utta_conf": utta_f1[i] - conf_f1[i],
                "d_f1_utta_tent": utta_f1[i] - tent_f1[i],
                "harm_utta": utta_harm[i],
                "harm_conf": conf_harm[i],
                "harm_tent": tent_harm[i],
                "cov_utta": utta_cov[i],
                "cov_conf": conf_cov[i],
                "src_sens": src_sens[i],
                "src_auroc": src_auroc[i],
                "utta_auroc": utta_auroc[i],
                "conf_auroc": conf_auroc[i],
                "tent_auroc": tent_auroc[i],
            }
        )

    print("seed-level done", flush=True)

    # Seed-42 paired WSI bootstrap (pooled patches)
    zsrc = np.load(NPZ_DIR / "s42_source.npz")
    y, sl = zsrc["y"], zsrc["slide"]
    uniq = np.unique(sl)
    groups = {int(s): np.where(sl == s)[0] for s in uniq}
    keys = np.array(list(groups), dtype=int)
    dumps = {
        "src": zsrc["p"],
        "utta": np.load(NPZ_DIR / "s42_utta.npz")["p"],
        "conf": np.load(NPZ_DIR / "s42_confidence.npz")["p"],
        "tent": np.load(NPZ_DIR / "s42_tent.npz")["p"],
        "eata": np.load(NPZ_DIR / "s42_eata.npz")["p"],
        "rand": np.load(NPZ_DIR / "s42_random.npz")["p"],
    }
    rng = np.random.default_rng(42)
    s42_paired = {}
    pair_list = [
        ("utta", "src", True),
        ("utta", "conf", True),
        ("conf", "src", True),
        ("utta", "tent", False),
        ("utta", "eata", False),
        ("utta", "rand", False),
        ("tent", "src", False),
    ]
    for ma, mb, want_f1 in pair_list:
        print(f"bootstrap s42 {ma}-{mb} f1={want_f1}", flush=True)
        s42_paired[f"{ma}-{mb}"] = bootstrap_paired_from_groups(
            y, dumps[ma], dumps[mb], groups, keys, N_BOOT, rng, want_f1
        )

    def slide_metrics(p):
        out = []
        for s in sorted(np.unique(sl)):
            m = sl == s
            yi, pi = y[m], p[m]
            pred = (pi >= 0.5).astype(int)
            au = float(roc_auc_score(yi, pi)) if len(np.unique(yi)) == 2 else float("nan")
            out.append(
                {
                    "slide": int(s),
                    "n": int(m.sum()),
                    "pos": int(yi.sum()),
                    "auroc": au,
                    "f1": float(f1_score(yi, pred)),
                    "acc": float((pred == yi).mean()),
                }
            )
        return out

    s42_src_sl = slide_metrics(dumps["src"])
    s42_utta_sl = slide_metrics(dumps["utta"])
    s42_tent_sl = slide_metrics(dumps["tent"])
    s42_conf_sl = slide_metrics(dumps["conf"])

    slide_story = {
        42: {
            "slides": [x["slide"] for x in s42_src_sl],
            "n": [x["n"] for x in s42_src_sl],
            "pos": [x["pos"] for x in s42_src_sl],
            "src_auroc": [x["auroc"] for x in s42_src_sl],
            "utta_auroc": [x["auroc"] for x in s42_utta_sl],
            "conf_auroc": [x["auroc"] for x in s42_conf_sl],
            "tent_auroc": [x["auroc"] for x in s42_tent_sl],
            "src_f1": [x["f1"] for x in s42_src_sl],
            "utta_f1": [x["f1"] for x in s42_utta_sl],
            "tent_f1": [x["f1"] for x in s42_tent_sl],
        }
    }

    boot_files = {
        123: BOOT_DIR / "camelyon17_wsi_bootstrap_s123.json",
        2024: BOOT_DIR / "camelyon17_wsi_bootstrap_s2024.json",
        7: BOOT_DIR / "camelyon17_wsi_bootstrap_s7.json",
        99: BOOT_DIR / "camelyon17_wsi_bootstrap_s99.json",
    }
    perslide_paired = {}
    for s, pth in boot_files.items():
        d = json.loads(pth.read_text())
        methods = d["methods"]
        slides = [x["slide"] for x in methods["source"]["per_slide"]]
        ns = [x["n"] for x in methods["source"]["per_slide"]]
        pos = [x["pos"] for x in methods["source"]["per_slide"]]
        src_s = [x["auroc"] for x in methods["source"]["per_slide"]]
        src_f = [x["f1"] for x in methods["source"]["per_slide"]]
        rng_s = np.random.default_rng(s)
        rec = {}
        for mkey, alias in [
            ("utta", "utta"),
            ("confidence", "conf"),
            ("tent", "tent"),
            ("eata", "eata"),
            ("random", "rand"),
        ]:
            ma = [x["auroc"] for x in methods[mkey]["per_slide"]]
            mf = [x["f1"] for x in methods[mkey]["per_slide"]]
            rec[f"{alias}-src_auroc_equal"] = bootstrap_paired_perslide(ma, src_s, N_BOOT, rng_s)
            rec[f"{alias}-src_auroc_nweight"] = bootstrap_paired_perslide(ma, src_s, N_BOOT, rng_s, ns)
            rec[f"{alias}-src_f1_equal"] = bootstrap_paired_perslide(mf, src_f, N_BOOT, rng_s)
            rec[f"{alias}-src_f1_nweight"] = bootstrap_paired_perslide(mf, src_f, N_BOOT, rng_s, ns)
        ua = [x["auroc"] for x in methods["utta"]["per_slide"]]
        ca = [x["auroc"] for x in methods["confidence"]["per_slide"]]
        rec["utta-conf_auroc_equal"] = bootstrap_paired_perslide(ua, ca, N_BOOT, rng_s)
        rec["utta-conf_auroc_nweight"] = bootstrap_paired_perslide(ua, ca, N_BOOT, rng_s, ns)
        perslide_paired[s] = rec
        slide_story[s] = {
            "slides": slides,
            "n": ns,
            "pos": pos,
            "src_auroc": src_s,
            "utta_auroc": ua,
            "conf_auroc": ca,
            "tent_auroc": [x["auroc"] for x in methods["tent"]["per_slide"]],
            "src_f1": src_f,
            "utta_f1": [x["f1"] for x in methods["utta"]["per_slide"]],
            "tent_f1": [x["f1"] for x in methods["tent"]["per_slide"]],
        }

    diag = {}
    for s in SEEDS:
        st = slide_story[s]
        tent_d = np.array(st["tent_auroc"]) - np.array(st["src_auroc"])
        utta_d = np.array(st["utta_auroc"]) - np.array(st["src_auroc"])
        worst = int(np.argmin(tent_d))
        diag[s] = {
            "src_auroc_point": rows[s]["src_auroc"],
            "src_sens": rows[s]["src_sens"],
            "src_f1": rows[s]["src_f1"],
            "ood_val_auroc": rows[s]["ood"],
            "tent_auroc": rows[s]["tent_auroc"],
            "utta_auroc": rows[s]["utta_auroc"],
            "n_slides_tent_drop_gt_0.1": int((tent_d < -0.1).sum()),
            "n_slides_utta_drop_gt_0.02": int((utta_d < -0.02).sum()),
            "n_slides_utta_up_gt_0.02": int((utta_d > 0.02).sum()),
            "worst_tent_slide": st["slides"][worst],
            "worst_tent_d_auroc": float(tent_d[worst]),
            "mean_equal_weight_slide_d_utta": float(utta_d.mean()),
            "per_slide": [
                {
                    "slide": st["slides"][i],
                    "n": st["n"][i],
                    "pos": st["pos"][i],
                    "prev": st["pos"][i] / st["n"][i],
                    "src_auroc": st["src_auroc"][i],
                    "tent_auroc": st["tent_auroc"][i],
                    "utta_auroc": st["utta_auroc"][i],
                    "d_utta": float(utta_d[i]),
                    "d_tent": float(tent_d[i]),
                    "src_f1": st["src_f1"][i],
                    "utta_f1": st["utta_f1"][i],
                    "tent_f1": st["tent_f1"][i],
                }
                for i in range(len(st["slides"]))
            ],
        }

    plt.rcParams.update(
        {"font.size": 9, "figure.dpi": 140, "axes.spines.top": False, "axes.spines.right": False}
    )
    x = np.arange(len(SEEDS))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, [p["d_auroc_utta_src"] for p in per_seed_delta], w, label="UTTA − Source", color="#2a6f97")
    ax.bar(x + w / 2, [p["d_auroc_utta_conf"] for p in per_seed_delta], w, label="UTTA − Confidence", color="#9b2226")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Δ AUROC")
    ax.set_title("Per-seed AUROC deltas (frozen protocol, no retune)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_delta_auroc_per_seed.png")
    fig.savefig(OUT / "fig_delta_auroc_per_seed.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(x - w / 2, [p["d_f1_utta_src"] for p in per_seed_delta], w, label="UTTA − Source", color="#2a6f97")
    ax.bar(x + w / 2, [p["d_f1_utta_conf"] for p in per_seed_delta], w, label="UTTA − Confidence", color="#9b2226")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in SEEDS])
    ax.set_xlabel("Seed")
    ax.set_ylabel("Δ F1")
    ax.set_title("Per-seed F1 deltas — UTTA > Source on 5/5 seeds")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_delta_f1_per_seed.png")
    fig.savefig(OUT / "fig_delta_f1_per_seed.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(SEEDS, tent_harm, "o-", label="Tent", color="#e07a5f")
    ax.plot(SEEDS, rand_harm, "s--", label="Random", color="#9a8c98")
    ax.plot(SEEDS, eata_harm, "d-", label="EATA", color="#3d405b")
    ax.plot(SEEDS, conf_harm, "^-", label="Confidence", color="#81b29a")
    ax.plot(SEEDS, utta_harm, "o-", label="UTTA-Med", color="#2a6f97")
    ax.set_xticks(SEEDS)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Harm rate")
    ax.set_title("Harmful-adaptation rate")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig_harm_per_seed.png")
    fig.savefig(OUT / "fig_harm_per_seed.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=True)
    for ax, s in zip(axes, (7, 99)):
        st = slide_story[s]
        xx = np.arange(len(st["slides"]))
        ax.plot(xx, st["src_auroc"], "o-", label="Source", color="#4a4e69")
        ax.plot(xx, st["tent_auroc"], "s-", label="Tent", color="#e07a5f")
        ax.plot(xx, st["utta_auroc"], "^-", label="UTTA-Med", color="#2a6f97")
        ax.set_xticks(xx)
        ax.set_xticklabels([str(v) for v in st["slides"]], fontsize=7)
        ax.set_title(f"Seed {s} per-slide AUROC")
        ax.set_xlabel("Hospital-2 WSI id")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
    axes[0].set_ylabel("AUROC")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_seed7_99_per_slide.png")
    fig.savefig(OUT / "fig_seed7_99_per_slide.png")
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5))
    labs = [METHOD_NAME[m] for m in ORDER]
    cols = ["#4a4e69", "#e07a5f", "#3d405b", "#9a8c98", "#81b29a", "#2a6f97"]
    for ax, field, title, ylim in [
        (axes[0], "auroc", "AUROC", (0.3, 1.0)),
        (axes[1], "f1", "F1", (0.0, 1.0)),
        (axes[2], "harm", "Harm rate", (0, 0.45)),
    ]:
        mus, sds = [], []
        for m in ORDER:
            if field == "harm" and m == "src":
                mus.append(np.nan)
                sds.append(0.0)
            else:
                mus.append(metrics_across[f"{m}_{field}"]["mean"])
                sds.append(metrics_across[f"{m}_{field}"]["sd"])
        ax.bar(range(len(ORDER)), mus, yerr=sds, color=cols, capsize=3, error_kw={"lw": 0.8})
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(labs, rotation=35, ha="right")
        ax.set_title(title + "  (mean ± SD, n=5)")
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(FIG / "fig_main_mean_sd.png")
    fig.savefig(OUT / "fig_main_mean_sd.png")
    plt.close()

    main_table = []
    for m in ORDER:
        rec = {"method": METHOD_NAME[m]}
        for f in ("auroc", "f1", "ece"):
            rec[f"{f}_mean"] = metrics_across[f"{m}_{f}"]["mean"]
            rec[f"{f}_sd"] = metrics_across[f"{m}_{f}"]["sd"]
        if m != "src":
            rec["harm_mean"] = metrics_across[f"{m}_harm"]["mean"]
            rec["harm_sd"] = metrics_across[f"{m}_harm"]["sd"]
            rec["cov_mean"] = metrics_across[f"{m}_cov"]["mean"]
            rec["cov_sd"] = metrics_across[f"{m}_cov"]["sd"]
        else:
            rec["harm_mean"] = rec["harm_sd"] = rec["cov_mean"] = rec["cov_sd"] = None
        rec["auroc_values"] = metrics_across[f"{m}_auroc"]["values"]
        rec["f1_values"] = metrics_across[f"{m}_f1"]["values"]
        main_table.append(rec)

    payload = {
        "n_seeds": 5,
        "seeds": SEEDS,
        "protocol": {
            "bn_mode": "bn_freeze_stats",
            "tta_lr": 1e-5,
            "n_mc": 20,
            "tau_rule": "70th percentile of val U (per seed, hospital 1 only)",
            "no_retune_on_7_99": True,
            "primary_multi_seed_inference": "paired seed-level mean Δ ± t-interval (df=4)",
            "wsi_delta": "seed 42: pooled-patch WSI resample; other seeds: per-slide paired bootstrap",
        },
        "main_table": main_table,
        "per_seed_delta": per_seed_delta,
        "paired_seed_level": comparisons,
        "holm_adjusted_p": holm_p,
        "seed42_paired_wsi_bootstrap": s42_paired,
        "perslide_paired_bootstrap": perslide_paired,
        "diagnosis_7_99": {str(k): v for k, v in diag.items()},
        "coverage": {
            "utta": mean_sd(utta_cov),
            "conf": mean_sd(conf_cov),
            "rand": mean_sd(rand_cov),
            "eata": mean_sd(eata_cov),
        },
        "harm": {
            "utta": mean_sd(utta_harm),
            "conf": mean_sd(conf_harm),
            "tent": mean_sd(tent_harm),
            "eata": mean_sd(eata_harm),
            "rand": mean_sd(rand_harm),
        },
        "u_error_corr": list(mean_sd(u_r)),
        "shift_auroc": list(mean_sd(drop_auroc)),
    }
    (OUT / "five_seed_final_stats.json").write_text(json.dumps(payload, indent=2, default=str))

    def fmt(mu, sd, nd=3):
        if mu is None:
            return "—"
        return f"{mu:.{nd}f} ± {sd:.{nd}f}"

    with open(OUT / "table1_main_results.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["Method", "AUROC", "F1", "ECE", "Coverage", "Harm rate"])
        for rec in main_table:
            wcsv.writerow(
                [
                    rec["method"],
                    fmt(rec["auroc_mean"], rec["auroc_sd"]),
                    fmt(rec["f1_mean"], rec["f1_sd"]),
                    fmt(rec["ece_mean"], rec["ece_sd"]),
                    fmt(rec["cov_mean"], rec["cov_sd"]) if rec["cov_mean"] is not None else "—",
                    fmt(rec["harm_mean"], rec["harm_sd"]) if rec["harm_mean"] is not None else "—",
                ]
            )

    with open(OUT / "table_per_seed_delta.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(
            [
                "seed",
                "dAUROC UTTA-src",
                "dAUROC UTTA-conf",
                "dAUROC UTTA-tent",
                "dF1 UTTA-src",
                "dF1 UTTA-conf",
                "harm UTTA",
                "harm Tent",
                "cov UTTA",
                "src_sens",
            ]
        )
        for p in per_seed_delta:
            wcsv.writerow(
                [
                    p["seed"],
                    f"{p['d_auroc_utta_src']:+.4f}",
                    f"{p['d_auroc_utta_conf']:+.4f}",
                    f"{p['d_auroc_utta_tent']:+.4f}",
                    f"{p['d_f1_utta_src']:+.4f}",
                    f"{p['d_f1_utta_conf']:+.4f}",
                    f"{p['harm_utta']:.4f}",
                    f"{p['harm_tent']:.4f}",
                    f"{p['cov_utta']:.3f}",
                    f"{p['src_sens']:.3f}",
                ]
            )

    print("\n=== TABLE 1 ===")
    for rec in main_table:
        h = rec["harm_mean"]
        hs = "—" if h is None else f"{h:.3f}±{rec['harm_sd']:.3f}"
        print(
            f"{rec['method']:22s} AUROC {rec['auroc_mean']:.3f}±{rec['auroc_sd']:.3f}  "
            f"F1 {rec['f1_mean']:.3f}±{rec['f1_sd']:.3f}  harm {hs}"
        )
    print("\n=== PAIRED SEED-LEVEL Δ ===")
    for k in [
        "utta-src_auroc",
        "utta-conf_auroc",
        "utta-tent_auroc",
        "utta-eata_auroc",
        "utta-rand_auroc",
        "utta-src_f1",
        "utta-conf_f1",
        "utta-tent_harm",
        "utta-conf_harm",
        "conf-src_f1",
        "tent-src_auroc",
    ]:
        c = comparisons[k]
        print(
            f"{k:22s} Δ={c['mean_delta']:+.4f}  95%tCI [{c['ci95_low']:+.4f},{c['ci95_high']:+.4f}]  "
            f"p_t={c['t_p']:.4f}  p_w={c['wilcoxon_p']:.4f}  d={c['cohens_d']:+.2f}  "
            f"signs +{c['n_pos']}/-{c['n_neg']}"
        )
    print("\nHolm-adjusted:")
    for k, v in holm_p.items():
        print(f"  {k:32s} {v:.4f}")
    print("\n=== SEED 42 PAIRED WSI Δ (pooled) ===")
    for k, v in s42_paired.items():
        line = (
            f"  {k:12s} ΔAUROC {v['delta_auroc']['mean']:+.4f} "
            f"[{v['delta_auroc']['ci_low']:+.4f},{v['delta_auroc']['ci_high']:+.4f}]"
        )
        if "delta_f1" in v:
            line += (
                f"  ΔF1 {v['delta_f1']['mean']:+.4f} "
                f"[{v['delta_f1']['ci_low']:+.4f},{v['delta_f1']['ci_high']:+.4f}]"
            )
        print(line)
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
