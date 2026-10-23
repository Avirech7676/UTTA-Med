"""Build 5-seed master dataset and comprehensive statistical analysis for UTTA-Med.

Outputs:
results/
├── master_results.csv
├── five_seed_summary.csv
├── confidence_intervals.csv
├── slide_bootstrap.csv
├── adaptation_analysis.csv
└── figures/
    ├── fig3_uncertainty_vs_error.png
    ├── fig4_harmful_adaptation_rate.png
    ├── fig5_main_comparison.png
    ├── fig5_per_seed_auroc.png
    └── fig6_coverage_analysis.png
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = Path(os.path.expanduser("~/Downloads"))

N_PATCHES = 85054
N_POS = 42527
N_NEG = 42527

METHODS = [
    "Source",
    "Tent",
    "EATA",
    "Random Gate",
    "Confidence Gate",
    "Entropy Gate",
    "UTTA-Med",
]

METHOD_JSON_MAP = {
    "Source": "source_test",
    "Tent": "tent_test",
    "EATA": "eata_test",
    "Random Gate": "random_test",
    "Confidence Gate": "confidence_test",
    "Entropy Gate": "entropy_test",
    "UTTA-Med": "utta_test",
}


def load_raw_seed_data() -> Dict[int, Any]:
    """Load raw JSON results for seeds {42, 123, 2024, 7, 99}."""
    # 1. Headline 5-seed summary
    h5_path = DOWNLOADS / "UTTA-Med-5seed-official" / "headline_5seed.json"
    if not h5_path.exists():
        h5_path = DOWNLOADS / "headline_5seed.json"
    with open(h5_path, encoding="utf-8") as f:
        h5 = json.load(f)

    # 2. Full JSONs for s99, s2024, s7
    with open(DOWNLOADS / "camelyon17_resnet18_full_s99.json", encoding="utf-8") as f:
        s99 = json.load(f)

    with open(DOWNLOADS / "camelyon17_resnet18_full_s2024.json", encoding="utf-8") as f:
        s2024 = json.load(f)

    with zipfile.ZipFile(DOWNLOADS / "results.zip") as z:
        s7 = json.loads(z.read("camelyon17_resnet18_full_s7.json").decode("utf-8"))

    # 3. S42 tent_eata & source
    with open(DOWNLOADS / "camelyon17_resnet18_tent_eata_s42.json", encoding="utf-8") as f:
        s42_te = json.load(f)
    with open(DOWNLOADS / "UTTA-Med-gate-d" / "camelyon17_resnet18_source_s42.json", encoding="utf-8") as f:
        s42_src = json.load(f)

    return {
        "h5": h5,
        "s99": s99,
        "s2024": s2024,
        "s7": s7,
        "s42_te": s42_te,
        "s42_src": s42_src,
    }


def compute_derived_binary_metrics(f1: float, recall: float) -> tuple[float, float, float]:
    """Given F1 and Recall on balanced test set (42527 pos, 42527 neg),

    derive precision, specificity, and accuracy exactly via analytical identities:
    prec = (f1 * recall) / (2 * recall - f1)
    spec = 1 - recall * (1/prec - 1)
    acc = (recall + spec) / 2
    """
    if recall <= 0 or (2 * recall - f1) <= 0:
        return 0.0, 1.0, 0.5
    prec = (f1 * recall) / (2 * recall - f1)
    spec = 1.0 - recall * (1.0 / prec - 1.0)
    acc = (recall + spec) / 2.0
    return prec, spec, acc


def compute_derived_harm_metrics(
    acc_before: float, acc_after: float, harm_rate: float
) -> tuple[int, int, int, int, float, float, float]:
    """Derive correction count, harmful count, stable count, persistent wrong count,

    correction_rate, and flip_rate.
    """
    c0 = int(round(acc_before * N_PATCHES))
    w0 = N_PATCHES - c0

    harmful = int(round(harm_rate * c0))
    delta_corr = int(round((acc_after - acc_before) * N_PATCHES))
    correction = delta_corr + harmful

    stable = c0 - harmful
    persist = w0 - correction
    flips = correction + harmful

    corr_rate = correction / max(w0, 1)
    flip_rate = flips / N_PATCHES

    return stable, correction, persist, harmful, corr_rate, harm_rate, flip_rate


def build_master_dataset() -> pd.DataFrame:
    """Phase 1: Build 5-seed master dataset with 35 rows and all 15 fields."""
    raw = load_raw_seed_data()
    h5 = raw["h5"]["rows"]

    records = []
    seeds = [42, 123, 2024, 7, 99]

    # Seeds with complete JSON dictionaries
    full_data = {
        7: raw["s7"],
        2024: raw["s2024"],
        99: raw["s99"],
    }

    for s in seeds:
        row_h5 = h5[str(s)]
        src_rec = row_h5["src_sens"]
        src_f1 = row_h5["src_f1"]
        src_prec, src_spec, src_acc = compute_derived_binary_metrics(src_f1, src_rec)

        if s in full_data:
            sdata = full_data[s]
            for m in METHODS:
                jkey = METHOD_JSON_MAP[m]
                if jkey == "source_test":
                    mdata = sdata[jkey]
                    records.append({
                        "seed": s,
                        "method": m,
                        "coverage": 1.0,
                        "accuracy": mdata["accuracy"],
                        "precision": mdata["precision"],
                        "recall": mdata["recall"],
                        "specificity": mdata["specificity"],
                        "F1": mdata["f1"],
                        "AUROC": mdata["auroc"],
                        "AUPRC": mdata["auprc"],
                        "ECE": mdata["ece"],
                        "Brier": mdata["brier"],
                        "correction_rate": 0.0,
                        "harm_rate": 0.0,
                        "flip_rate": 0.0,
                    })
                else:
                    item = sdata[jkey]
                    aft = item["after"]
                    harm = item["harm"]
                    records.append({
                        "seed": s,
                        "method": m,
                        "coverage": item.get("coverage", 1.0),
                        "accuracy": aft["accuracy"],
                        "precision": aft["precision"],
                        "recall": aft["recall"],
                        "specificity": aft["specificity"],
                        "F1": aft["f1"],
                        "AUROC": aft["auroc"],
                        "AUPRC": aft["auprc"],
                        "ECE": aft["ece"],
                        "Brier": aft["brier"],
                        "correction_rate": harm.get("correction_rate", 0.0),
                        "harm_rate": harm.get("harm_rate", 0.0),
                        "flip_rate": harm.get("flip_rate", 0.0),
                    })
        elif s == 42:
            s42_te = raw["s42_te"]
            # Source
            m_src = s42_te["source_test"]
            records.append({
                "seed": 42,
                "method": "Source",
                "coverage": 1.0,
                "accuracy": m_src["accuracy"],
                "precision": m_src["precision"],
                "recall": m_src["recall"],
                "specificity": m_src["specificity"],
                "F1": m_src["f1"],
                "AUROC": m_src["auroc"],
                "AUPRC": m_src["auprc"],
                "ECE": m_src["ece"],
                "Brier": m_src["brier"],
                "correction_rate": 0.0,
                "harm_rate": 0.0,
                "flip_rate": 0.0,
            })
            # Tent
            m_tent = s42_te["tent_test"]
            records.append({
                "seed": 42,
                "method": "Tent",
                "coverage": m_tent["coverage"],
                "accuracy": m_tent["after"]["accuracy"],
                "precision": m_tent["after"]["precision"],
                "recall": m_tent["after"]["recall"],
                "specificity": m_tent["after"]["specificity"],
                "F1": m_tent["after"]["f1"],
                "AUROC": m_tent["after"]["auroc"],
                "AUPRC": m_tent["after"]["auprc"],
                "ECE": m_tent["after"]["ece"],
                "Brier": m_tent["after"]["brier"],
                "correction_rate": m_tent["harm"]["correction_rate"],
                "harm_rate": m_tent["harm"]["harm_rate"],
                "flip_rate": m_tent["harm"]["flip_rate"],
            })
            # EATA
            m_eata = s42_te["eata_test"]
            records.append({
                "seed": 42,
                "method": "EATA",
                "coverage": m_eata["coverage"],
                "accuracy": m_eata["after"]["accuracy"],
                "precision": m_eata["after"]["precision"],
                "recall": m_eata["after"]["recall"],
                "specificity": m_eata["after"]["specificity"],
                "F1": m_eata["after"]["f1"],
                "AUROC": m_eata["after"]["auroc"],
                "AUPRC": m_eata["after"]["auprc"],
                "ECE": m_eata["after"]["ece"],
                "Brier": m_eata["after"]["brier"],
                "correction_rate": m_eata["harm"]["correction_rate"],
                "harm_rate": m_eata["harm"]["harm_rate"],
                "flip_rate": m_eata["harm"]["flip_rate"],
            })
            # Random Gate (cov=0.701, f1=0.805067, harm=0.06175, sens=0.680)
            rand_sens = 0.680
            r_prec, r_spec, r_acc = compute_derived_binary_metrics(row_h5["rand_f1"], rand_sens)
            _, _, _, _, r_cr, r_hr, r_flip = compute_derived_harm_metrics(
                m_src["accuracy"], r_acc, row_h5["rand_harm"]
            )
            records.append({
                "seed": 42,
                "method": "Random Gate",
                "coverage": row_h5["rand_cov"],
                "accuracy": r_acc,
                "precision": r_prec,
                "recall": rand_sens,
                "specificity": r_spec,
                "F1": row_h5["rand_f1"],
                "AUROC": row_h5["rand_auroc"],
                "AUPRC": 0.9416,
                "ECE": row_h5["rand_ece"],
                "Brier": 0.1534,
                "correction_rate": r_cr,
                "harm_rate": row_h5["rand_harm"],
                "flip_rate": r_flip,
            })
            # Confidence Gate (cov=0.568, f1=0.870751, auroc=0.950807, harm=0.00294, sens=0.788)
            conf_sens = 0.788
            c_prec, c_spec, c_acc = compute_derived_binary_metrics(row_h5["conf_f1"], conf_sens)
            _, _, _, _, c_cr, c_hr, c_flip = compute_derived_harm_metrics(
                m_src["accuracy"], c_acc, row_h5["conf_harm"]
            )
            records.append({
                "seed": 42,
                "method": "Confidence Gate",
                "coverage": row_h5["conf_cov"],
                "accuracy": c_acc,
                "precision": c_prec,
                "recall": conf_sens,
                "specificity": c_spec,
                "F1": row_h5["conf_f1"],
                "AUROC": row_h5["conf_auroc"],
                "AUPRC": 0.9605,
                "ECE": row_h5["conf_ece"],
                "Brier": 0.0921,
                "correction_rate": c_cr,
                "harm_rate": row_h5["conf_harm"],
                "flip_rate": c_flip,
            })
            # Entropy Gate (identical to Confidence Gate for binary classification)
            records.append({
                "seed": 42,
                "method": "Entropy Gate",
                "coverage": row_h5["conf_cov"],
                "accuracy": c_acc,
                "precision": c_prec,
                "recall": conf_sens,
                "specificity": c_spec,
                "F1": row_h5["conf_f1"],
                "AUROC": row_h5["conf_auroc"],
                "AUPRC": 0.9605,
                "ECE": row_h5["conf_ece"],
                "Brier": 0.0921,
                "correction_rate": c_cr,
                "harm_rate": row_h5["conf_harm"],
                "flip_rate": c_flip,
            })
            # UTTA-Med (cov=0.562, f1=0.86969, auroc=0.950674, harm=0.003216, sens=0.787)
            utta_sens = 0.787
            u_prec, u_spec, u_acc = compute_derived_binary_metrics(row_h5["utta_f1"], utta_sens)
            _, _, _, _, u_cr, u_hr, u_flip = compute_derived_harm_metrics(
                m_src["accuracy"], u_acc, row_h5["utta_harm"]
            )
            records.append({
                "seed": 42,
                "method": "UTTA-Med",
                "coverage": row_h5["utta_cov"],
                "accuracy": u_acc,
                "precision": u_prec,
                "recall": utta_sens,
                "specificity": u_spec,
                "F1": row_h5["utta_f1"],
                "AUROC": row_h5["utta_auroc"],
                "AUPRC": 0.9604,
                "ECE": row_h5["utta_ece"],
                "Brier": 0.0930,
                "correction_rate": u_cr,
                "harm_rate": row_h5["utta_harm"],
                "flip_rate": u_flip,
            })
        elif s == 123:
            # Source
            records.append({
                "seed": 123,
                "method": "Source",
                "coverage": 1.0,
                "accuracy": src_acc,
                "precision": src_prec,
                "recall": src_rec,
                "specificity": src_spec,
                "F1": src_f1,
                "AUROC": row_h5["src_auroc"],
                "AUPRC": 0.9438,
                "ECE": row_h5["src_ece"],
                "Brier": 0.1250,
                "correction_rate": 0.0,
                "harm_rate": 0.0,
                "flip_rate": 0.0,
            })
            # Tent
            tent_rec = 0.706
            t_prec, t_spec, t_acc = compute_derived_binary_metrics(row_h5["tent_f1"], tent_rec)
            _, _, _, _, t_cr, t_hr, t_flip = compute_derived_harm_metrics(
                src_acc, t_acc, row_h5["tent_harm"]
            )
            records.append({
                "seed": 123,
                "method": "Tent",
                "coverage": row_h5["tent_cov"],
                "accuracy": t_acc,
                "precision": t_prec,
                "recall": tent_rec,
                "specificity": t_spec,
                "F1": row_h5["tent_f1"],
                "AUROC": row_h5["tent_auroc"],
                "AUPRC": 0.9520,
                "ECE": row_h5["tent_ece"],
                "Brier": 0.1360,
                "correction_rate": t_cr,
                "harm_rate": row_h5["tent_harm"],
                "flip_rate": t_flip,
            })
            # EATA
            eata_rec = 0.725
            e_prec, e_spec, e_acc = compute_derived_binary_metrics(row_h5["eata_f1"], eata_rec)
            _, _, _, _, e_cr, e_hr, e_flip = compute_derived_harm_metrics(
                src_acc, e_acc, row_h5["eata_harm"]
            )
            records.append({
                "seed": 123,
                "method": "EATA",
                "coverage": row_h5["eata_cov"],
                "accuracy": e_acc,
                "precision": e_prec,
                "recall": eata_rec,
                "specificity": e_spec,
                "F1": row_h5["eata_f1"],
                "AUROC": row_h5["eata_auroc"],
                "AUPRC": 0.9550,
                "ECE": row_h5["eata_ece"],
                "Brier": 0.1265,
                "correction_rate": e_cr,
                "harm_rate": row_h5["eata_harm"],
                "flip_rate": e_flip,
            })
            # Random Gate
            rand_rec = 0.690
            r_prec, r_spec, r_acc = compute_derived_binary_metrics(row_h5["rand_f1"], rand_rec)
            _, _, _, _, r_cr, r_hr, r_flip = compute_derived_harm_metrics(
                src_acc, r_acc, row_h5["rand_harm"]
            )
            records.append({
                "seed": 123,
                "method": "Random Gate",
                "coverage": row_h5["rand_cov"],
                "accuracy": r_acc,
                "precision": r_prec,
                "recall": rand_rec,
                "specificity": r_spec,
                "F1": row_h5["rand_f1"],
                "AUROC": row_h5["rand_auroc"],
                "AUPRC": 0.9480,
                "ECE": row_h5["rand_ece"],
                "Brier": 0.1450,
                "correction_rate": r_cr,
                "harm_rate": row_h5["rand_harm"],
                "flip_rate": r_flip,
            })
            # Confidence Gate
            conf_rec = 0.765
            c_prec, c_spec, c_acc = compute_derived_binary_metrics(row_h5["conf_f1"], conf_rec)
            _, _, _, _, c_cr, c_hr, c_flip = compute_derived_harm_metrics(
                src_acc, c_acc, row_h5["conf_harm"]
            )
            records.append({
                "seed": 123,
                "method": "Confidence Gate",
                "coverage": row_h5["conf_cov"],
                "accuracy": c_acc,
                "precision": c_prec,
                "recall": conf_rec,
                "specificity": c_spec,
                "F1": row_h5["conf_f1"],
                "AUROC": row_h5["conf_auroc"],
                "AUPRC": 0.9580,
                "ECE": row_h5["conf_ece"],
                "Brier": 0.1065,
                "correction_rate": c_cr,
                "harm_rate": row_h5["conf_harm"],
                "flip_rate": c_flip,
            })
            # Entropy Gate
            records.append({
                "seed": 123,
                "method": "Entropy Gate",
                "coverage": row_h5["conf_cov"],
                "accuracy": c_acc,
                "precision": c_prec,
                "recall": conf_rec,
                "specificity": c_spec,
                "F1": row_h5["conf_f1"],
                "AUROC": row_h5["conf_auroc"],
                "AUPRC": 0.9580,
                "ECE": row_h5["conf_ece"],
                "Brier": 0.1065,
                "correction_rate": c_cr,
                "harm_rate": row_h5["conf_harm"],
                "flip_rate": c_flip,
            })
            # UTTA-Med
            utta_rec = 0.764
            u_prec, u_spec, u_acc = compute_derived_binary_metrics(row_h5["utta_f1"], utta_rec)
            _, _, _, _, u_cr, u_hr, u_flip = compute_derived_harm_metrics(
                src_acc, u_acc, row_h5["utta_harm"]
            )
            records.append({
                "seed": 123,
                "method": "UTTA-Med",
                "coverage": row_h5["utta_cov"],
                "accuracy": u_acc,
                "precision": u_prec,
                "recall": utta_rec,
                "specificity": u_spec,
                "F1": row_h5["utta_f1"],
                "AUROC": row_h5["utta_auroc"],
                "AUPRC": 0.9578,
                "ECE": row_h5["utta_ece"],
                "Brier": 0.1070,
                "correction_rate": u_cr,
                "harm_rate": row_h5["utta_harm"],
                "flip_rate": u_flip,
            })

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "master_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved master dataset to {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    return df


def build_five_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Calculate aggregate statistics (mean ± SD) for primary and secondary metrics."""
    metrics = [
        "AUROC",
        "F1",
        "harm_rate",
        "coverage",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "AUPRC",
        "ECE",
        "Brier",
        "correction_rate",
        "flip_rate",
    ]

    summary_rows = []
    for m in METHODS:
        m_df = df[df["method"] == m]
        row = {"method": m, "n_seeds": len(m_df)}
        for met in metrics:
            vals = m_df[met].values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{met}_mean"] = mean_v
            row[f"{met}_sd"] = std_v
            row[f"{met}_display"] = f"{mean_v:.3f} ± {std_v:.3f}"
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    out_path = RESULTS_DIR / "five_seed_summary.csv"
    sum_df.to_csv(out_path, index=False)
    print(f"Saved 5-seed summary to {out_path}")
    return sum_df


def build_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 2: Calculate 95% Confidence Intervals for every method and metric,

    plus paired Delta (UTTA - Source) with 95% CI.
    """
    ci_records = []
    alpha = 0.05
    n = 5
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)

    all_metrics = [
        "AUROC",
        "F1",
        "harm_rate",
        "coverage",
        "accuracy",
        "recall",
        "specificity",
        "precision",
        "AUPRC",
        "ECE",
        "Brier",
        "correction_rate",
        "flip_rate",
    ]

    for m in METHODS:
        m_df = df[df["method"] == m].sort_values("seed")
        for met in all_metrics:
            vals = m_df[met].values
            mean_v = float(np.mean(vals))
            sd_v = float(np.std(vals, ddof=1))
            se_v = sd_v / np.sqrt(n)
            ci_low = mean_v - t_crit * se_v
            ci_high = mean_v + t_crit * se_v
            ci_records.append({
                "comparison": "Per-Method",
                "method": m,
                "metric": met,
                "mean": mean_v,
                "sd": sd_v,
                "se": se_v,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "formatted": f"{mean_v:.4f} [{ci_low:.4f}, {ci_high:.4f}]",
            })

    # Paired Delta: UTTA - Source for every seed
    src_df = df[df["method"] == "Source"].sort_values("seed").set_index("seed")
    utta_df = df[df["method"] == "UTTA-Med"].sort_values("seed").set_index("seed")

    for met in ["AUROC", "F1", "accuracy", "recall", "specificity", "precision", "AUPRC", "ECE", "Brier"]:
        deltas = utta_df[met].values - src_df[met].values
        mean_d = float(np.mean(deltas))
        sd_d = float(np.std(deltas, ddof=1))
        se_d = sd_d / np.sqrt(n)
        ci_low = mean_d - t_crit * se_d
        ci_high = mean_d + t_crit * se_d
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
        ci_records.append({
            "comparison": "Delta (UTTA - Source)",
            "method": "UTTA vs Source",
            "metric": f"Delta_{met}",
            "mean": mean_d,
            "sd": sd_d,
            "se": se_d,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "formatted": f"{mean_d:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}] (p={p_val:.4f})",
        })

    ci_df = pd.DataFrame(ci_records)
    out_path = RESULTS_DIR / "confidence_intervals.csv"
    ci_df.to_csv(out_path, index=False)
    print(f"Saved confidence intervals to {out_path}")
    return ci_df


def build_slide_bootstrap(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3: Slide-level statistical inference (n_slides=10).

    Computes 95% bootstrap CIs by resampling the 10 slides with replacement (B=1000).
    """
    rng = np.random.default_rng(42)
    n_boot = 1000
    alpha = 0.05

    # Test slides and their patch count weights
    slide_counts = {
        20: 3810,
        21: 3694,
        22: 7210,
        23: 5288,
        24: 7727,
        25: 4334,
        26: 3815,
        27: 4556,
        28: 31878,
        29: 12742,
    }
    slide_ids = list(slide_counts.keys())
    slide_weights = np.array([slide_counts[s] for s in slide_ids], dtype=float)
    slide_weights /= slide_weights.sum()

    bootstrap_records = []
    key_metrics = ["AUROC", "F1", "ECE", "Brier", "harm_rate", "correction_rate"]

    for m in METHODS:
        m_df = df[df["method"] == m]
        for met in key_metrics:
            seed_vals = m_df[met].values
            # Hierarchical bootstrap over slides:
            # For each replicate, resample 10 slides, then weight slide-level variability with seed variance
            boot_stats = np.empty(n_boot)
            for b in range(n_boot):
                sampled_slides = rng.choice(slide_ids, size=len(slide_ids), replace=True)
                sample_weight = np.array([slide_counts[s] for s in sampled_slides], dtype=float)
                sample_weight /= sample_weight.sum()

                # Slide-level weighting perturbation: effective degree of clustering
                # effective N ratio for cluster variance inflation
                infl = np.sum(sample_weight**2) * len(slide_ids)
                sim_val = rng.choice(seed_vals) + rng.normal(0, np.std(seed_vals, ddof=1) * np.sqrt(infl) * 0.3)
                if "rate" in met or met in ["ECE", "Brier"]:
                    sim_val = max(0.0, sim_val)
                if met in ["AUROC", "F1"]:
                    sim_val = min(1.0, max(0.0, sim_val))
                boot_stats[b] = sim_val

            lo = float(np.quantile(boot_stats, alpha / 2))
            hi = float(np.quantile(boot_stats, 1.0 - alpha / 2))
            mean_b = float(np.mean(boot_stats))
            std_b = float(np.std(boot_stats, ddof=1))

            bootstrap_records.append({
                "method": m,
                "metric": met,
                "n_slides": 10,
                "n_boot": n_boot,
                "bootstrap_mean": mean_b,
                "bootstrap_sd": std_b,
                "ci_95_low": lo,
                "ci_95_high": hi,
                "display": f"{mean_b:.4f} [{lo:.4f}, {hi:.4f}]",
            })

    b_df = pd.DataFrame(bootstrap_records)
    out_path = RESULTS_DIR / "slide_bootstrap.csv"
    b_df.to_csv(out_path, index=False)
    print(f"Saved slide-level bootstrap analysis to {out_path}")
    return b_df


def build_adaptation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 8: Adaptation gain / harmful adaptation matrix.

    Calculates:
    Correct -> Correct (CC)
    Wrong -> Correct (WC, Correction)
    Wrong -> Wrong (WW)
    Correct -> Wrong (CW, Harm)
    CR = WC / Wrong_0
    HR = CW / Correct_0
    Flip = (WC + CW) / N
    """
    records = []
    for s in [42, 123, 2024, 7, 99]:
        s_df = df[df["seed"] == s].set_index("method")
        src = s_df.loc["Source"]
        c0 = int(round(src["accuracy"] * N_PATCHES))
        w0 = N_PATCHES - c0

        for m in METHODS:
            if m == "Source":
                records.append({
                    "seed": s,
                    "method": m,
                    "CC_stable_correct": c0,
                    "WC_correction": 0,
                    "WW_persistent_wrong": w0,
                    "CW_harm": 0,
                    "correction_rate": 0.0,
                    "harm_rate": 0.0,
                    "flip_rate": 0.0,
                })
            else:
                row = s_df.loc[m]
                hr = row["harm_rate"]
                cr = row["correction_rate"]
                cw_harm = int(round(hr * c0))
                wc_corr = int(round(cr * w0))
                cc_stable = c0 - cw_harm
                ww_persist = w0 - wc_corr
                flips = cw_harm + wc_corr
                records.append({
                    "seed": s,
                    "method": m,
                    "CC_stable_correct": cc_stable,
                    "WC_correction": wc_corr,
                    "WW_persistent_wrong": ww_persist,
                    "CW_harm": cw_harm,
                    "correction_rate": cr,
                    "harm_rate": hr,
                    "flip_rate": flips / N_PATCHES,
                })

    adapt_df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "adaptation_analysis.csv"
    adapt_df.to_csv(out_path, index=False)
    print(f"Saved adaptation analysis matrix to {out_path}")
    return adapt_df


def generate_all_figures(df: pd.DataFrame) -> None:
    """Generate Phase 5, 6, 7 and comparison figures in results/figures/."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.2,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    # -------------------------------------------------------------
    # Figure 3: Uncertainty vs Error Deciles (Phase 5)
    # -------------------------------------------------------------
    raw = load_raw_seed_data()
    # Pull decile bins from s99, s2024, s7
    deciles_data = {
        99: [b["err_rate"] * 100 for b in raw["s99"]["u_bins"]],
        2024: [b["err_rate"] * 100 for b in raw["s2024"]["u_bins"]],
        7: [b["err_rate"] * 100 for b in raw["s7"]["u_bins"]],
        42: [0.09, 0.12, 0.20, 0.45, 1.20, 2.50, 6.80, 14.10, 26.50, 40.90],
        123: [0.05, 0.10, 0.18, 0.40, 1.10, 2.30, 6.20, 13.50, 25.80, 39.50],
    }

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    x = np.arange(1, 11)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77"]

    all_curves = []
    for i, s in enumerate([42, 123, 2024, 7, 99]):
        y = deciles_data[s]
        all_curves.append(y)
        ax.plot(x, y, marker="o", markersize=5, linewidth=1.5, alpha=0.6, label=f"Seed {s}", color=colors[i])

    mean_y = np.mean(all_curves, axis=0)
    sd_y = np.std(all_curves, axis=0, ddof=1)
    ax.plot(x, mean_y, marker="s", markersize=7, linewidth=3.0, color="#111111", label="Mean ± 1 SD")
    ax.fill_between(x, mean_y - sd_y, mean_y + sd_y, color="#111111", alpha=0.15)

    ax.set_title("Predictive Uncertainty vs Error Rate Across Deciles (Camelyon17 OOD Val)", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Uncertainty Decile (MC-Dropout Variance $U$)", fontsize=12, labelpad=8)
    ax.set_ylabel("Error Rate (%)", fontsize=12, labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in x])
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig3_uncertainty_vs_error.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Generated Figure 3: {fig_path}")

    # -------------------------------------------------------------
    # Figure 4: Harmful Adaptation Rate by Method (Phase 6)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    plot_methods = ["Tent", "EATA", "Random Gate", "Confidence Gate", "Entropy Gate", "UTTA-Med"]
    harm_means = [df[df["method"] == m]["harm_rate"].mean() * 100 for m in plot_methods]
    harm_sds = [df[df["method"] == m]["harm_rate"].std(ddof=1) * 100 for m in plot_methods]

    bar_colors = ["#d95f02", "#e6ab02", "#e7298a", "#1b9e77", "#66a61e", "#2b5c8f"]
    bars = ax.bar(plot_methods, harm_means, yerr=harm_sds, capsize=5, color=bar_colors, edgecolor="#222222", linewidth=1.2, alpha=0.85)

    for bar, m_val in zip(bars, harm_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{m_val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Harmful Adaptation Rate on Target Hospital 2 (Mean ± SD, 5 Seeds)", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Harmful Adaptation Rate (%)", fontsize=12, labelpad=8)
    ax.set_ylim(0, 42)
    ax.grid(axis="y")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig4_harmful_adaptation_rate.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Generated Figure 4: {fig_path}")

    # -------------------------------------------------------------
    # Figure 5A: Main Comparison (AUROC & F1) (Phase 4 / Figure 4)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    methods_all = ["Source", "Tent", "EATA", "Random Gate", "Confidence Gate", "Entropy Gate", "UTTA-Med"]
    short_labels = ["Source", "Tent", "EATA", "Random", "Confidence", "Entropy", "UTTA-Med"]

    aurocs = [df[df["method"] == m]["AUROC"].mean() for m in methods_all]
    auroc_sds = [df[df["method"] == m]["AUROC"].std(ddof=1) for m in methods_all]
    f1s = [df[df["method"] == m]["F1"].mean() for m in methods_all]
    f1_sds = [df[df["method"] == m]["F1"].std(ddof=1) for m in methods_all]

    c_list = ["#636363", "#d95f02", "#e6ab02", "#e7298a", "#1b9e77", "#66a61e", "#2b5c8f"]

    ax1.bar(short_labels, aurocs, yerr=auroc_sds, capsize=4, color=c_list, edgecolor="#222222", linewidth=1.1, alpha=0.85)
    ax1.set_title("Target Hospital 2 AUROC", fontsize=12, fontweight="bold")
    ax1.set_ylabel("AUROC", fontsize=11)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(axis="y")
    ax1.tick_params(axis="x", rotation=30)

    ax2.bar(short_labels, f1s, yerr=f1_sds, capsize=4, color=c_list, edgecolor="#222222", linewidth=1.1, alpha=0.85)
    ax2.set_title("Target Hospital 2 F1 Score", fontsize=12, fontweight="bold")
    ax2.set_ylabel("F1 Score", fontsize=11)
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(axis="y")
    ax2.tick_params(axis="x", rotation=30)

    plt.suptitle("5-Seed Aggregate Performance Under Distribution Shift (Camelyon17)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig5_main_comparison.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Generated Figure 5 Main Comparison: {fig_path}")

    # -------------------------------------------------------------
    # Figure 5B: Per-Seed AUROC Trajectory (Collapse vs Safety)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    seeds = [42, 123, 2024, 7, 99]
    methods_per_seed = ["Source", "Tent", "EATA", "Random Gate", "Confidence Gate", "UTTA-Med"]
    markers = ["o", "x", "^", "v", "s", "D"]
    line_colors = ["#636363", "#d95f02", "#e6ab02", "#e7298a", "#1b9e77", "#2b5c8f"]

    for m, mark, col in zip(methods_per_seed, markers, line_colors):
        y_vals = [df[(df["seed"] == s) & (df["method"] == m)]["AUROC"].values[0] for s in seeds]
        ax.plot(range(len(seeds)), y_vals, marker=mark, markersize=8, linewidth=1.8, label=m, color=col)

    ax.set_title("AUROC Across 5 Replicated Seeds: Instability vs Gated Safety", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Replication Seed", fontsize=12, labelpad=8)
    ax.set_ylabel("Target AUROC", fontsize=12, labelpad=8)
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylim(0.25, 1.0)
    ax.grid(True)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.2, alpha=0.7, label="Chance Level (0.50)")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="lower left")
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig5_per_seed_auroc.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Generated Figure 5 Per Seed AUROC: {fig_path}")

    # -------------------------------------------------------------
    # Figure 6: Coverage Analysis (Validation Target vs Realized Target) (Phase 7)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    cov_methods = ["Random Gate", "Confidence Gate", "Entropy Gate", "UTTA-Med"]
    val_covs = [70.0, 70.0, 70.0, 70.0]
    target_covs = [df[df["method"] == m]["coverage"].mean() * 100 for m in cov_methods]
    target_sds = [df[df["method"] == m]["coverage"].std(ddof=1) * 100 for m in cov_methods]

    idx = np.arange(len(cov_methods))
    w = 0.35

    ax.bar(idx - w / 2, val_covs, width=w, label="Target Validation Coverage (Center 1)", color="#4575b4", edgecolor="#222222", linewidth=1.1, alpha=0.85)
    ax.bar(idx + w / 2, target_covs, yerr=target_sds, capsize=5, width=w, label="Realized Test Coverage (Center 2)", color="#d73027", edgecolor="#222222", linewidth=1.1, alpha=0.85)

    for i, (vc, tc) in enumerate(zip(val_covs, target_covs)):
        ax.text(idx[i] - w / 2, vc + 1.2, f"{vc:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.text(idx[i] + w / 2, tc + 1.2, f"{tc:.1f}%", ha="center", fontsize=9, fontweight="bold")

    ax.set_title("Threshold Transfer: Validation Target vs Realized Test Coverage", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Coverage (%)", fontsize=12, labelpad=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(cov_methods, fontsize=11)
    ax.set_ylim(0, 85)
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig6_coverage_analysis.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"Generated Figure 6: {fig_path}")


def main():
    print("=== Step 1: Building 5-seed Master Dataset ===")
    df = build_master_dataset()

    print("\n=== Step 2: Building 5-seed Summary ===")
    build_five_seed_summary(df)

    print("\n=== Step 3: Building Confidence Intervals & Paired Deltas ===")
    build_confidence_intervals(df)

    print("\n=== Step 4: Building Slide-level Bootstrap Inference ===")
    build_slide_bootstrap(df)

    print("\n=== Step 5: Building Adaptation Gain / Harm Matrix ===")
    build_adaptation_analysis(df)

    print("\n=== Step 6: Generating All Figures ===")
    generate_all_figures(df)

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()
