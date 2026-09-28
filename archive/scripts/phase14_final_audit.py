#!/usr/bin/env python3
"""Phase 14 — recalculate 5-seed means/SDs/CIs from locked headline rows.

Does not read target labels. Does not retune. Writes paper/tables/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEAD = json.loads((ROOT / "results/tables/headline_5seed.json").read_text())
OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = ROOT / "docs"
SEEDS = [42, 123, 2024, 7, 99]
TCRIT = 2.776445  # t_0.975, df=4


def mean_sd(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def paired_t(d):
    d = np.asarray(d, dtype=float)
    n = len(d)
    m = float(d.mean())
    s = float(d.std(ddof=1))
    se = s / math.sqrt(n)
    t = m / se if se > 0 else float("nan")
    # two-sided p via t df=4: use scipy if present else skip exact p
    try:
        from scipy import stats

        p = float(stats.t.sf(abs(t), n - 1) * 2)
    except Exception:
        p = float("nan")
    lo = m - TCRIT * se
    hi = m + TCRIT * se
    return {
        "mean": m,
        "sd": s,
        "ci95": [lo, hi],
        "t": t,
        "p": p,
        "n_pos": int((d > 0).sum()),
        "n_neg": int((d < 0).sum()),
        "cohen_dz": m / s if s > 0 else float("nan"),
    }


def holm(ps):
    """Return adjusted p in original order."""
    n = len(ps)
    order = np.argsort(ps)
    adj = [0.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        a = min(1.0, ps[idx] * (n - rank))
        running = max(running, a)
        adj[idx] = running
    return adj


def main():
    rows = {int(k): v for k, v in HEAD["rows"].items()}
    methods = [
        ("Source-only", "src", False),
        ("Tent", "tent", True),
        ("EATA", "eata", True),
        ("Random-gated", "rand", True),
        ("Confidence-gated", "conf", True),
        ("UTTA-Med", "utta", True),
    ]
    table1 = []
    for name, pre, has_tta in methods:
        au = [rows[s][f"{pre}_auroc"] for s in SEEDS]
        f1 = [rows[s][f"{pre}_f1"] for s in SEEDS]
        ece = [rows[s][f"{pre}_ece"] for s in SEEDS]
        rec = {
            "method": name,
            "auroc": fmt(*mean_sd(au)),
            "f1": fmt(*mean_sd(f1)),
            "ece": fmt(*mean_sd(ece)),
        }
        if has_tta:
            rec["harm"] = fmt(*mean_sd([rows[s][f"{pre}_harm"] for s in SEEDS]))
            rec["coverage"] = fmt(*mean_sd([rows[s][f"{pre}_cov"] for s in SEEDS]))
        else:
            rec["harm"] = "—"
            rec["coverage"] = "—"
        rec["auroc_raw"] = au
        rec["f1_raw"] = f1
        table1.append(rec)

    # Cross-check full_s{7,99,2024}
    checks = []
    mapping = {
        "source_test": "src",
        "tent_test": "tent",
        "eata_test": "eata",
        "random_test": "rand",
        "confidence_test": "conf",
        "utta_test": "utta",
    }
    for seed in (7, 99, 2024):
        p = ROOT / "results/metrics" / f"camelyon17_resnet18_full_s{seed}.json"
        if not p.exists():
            checks.append({"seed": seed, "ok": False, "reason": "missing json"})
            continue
        d = json.loads(p.read_text())
        mism = []
        for jk, pre in mapping.items():
            block = d[jk]
            auroc = block["auroc"] if "auroc" in block else block.get("after", {}).get("auroc")
            # compacted files store auroc at top
            if auroc is None:
                continue
            ref = rows[seed][f"{pre}_auroc"]
            if abs(float(auroc) - ref) > 1.5e-4:
                mism.append(f"{pre}_auroc json={auroc} head={ref}")
        checks.append({"seed": seed, "ok": not mism, "mismatches": mism})

    # source s42
    s42 = json.loads((ROOT / "results/metrics/camelyon17_resnet18_source_s42.json").read_text())
    src42_ok = abs(s42["test_target"]["auroc"] - rows[42]["src_auroc"]) < 1e-6
    smoke = s42.get("gate") == "B-FULL" and s42.get("best_epoch") == 3

    # paired deltas
    contrasts = {}
    for metric, key in (("auroc", "auroc"), ("f1", "f1")):
        for a, b, name in [
            ("utta", "src", "UTTA − Source"),
            ("utta", "conf", "UTTA − Confidence"),
            ("utta", "tent", "UTTA − Tent"),
            ("conf", "src", "Confidence − Source"),
            ("eata", "src", "EATA − Source"),
        ]:
            d = [rows[s][f"{a}_{key}"] - rows[s][f"{b}_{key}"] for s in SEEDS]
            contrasts[f"{name} ({metric})"] = paired_t(d)

    harm_d = [rows[s]["utta_harm"] - rows[s]["tent_harm"] for s in SEEDS]
    contrasts["UTTA − Tent (harm)"] = paired_t(harm_d)

    pvals = [contrasts[k]["p"] for k in contrasts]
    if not any(math.isnan(p) for p in pvals):
        adj = holm(pvals)
        for k, a in zip(contrasts, adj):
            contrasts[k]["p_holm"] = a

    drops = [rows[s]["drop_auroc"] for s in SEEDS]
    ur = [rows[s]["u_r"] for s in SEEDS]

    # WSI overlap note
    wsi_path = ROOT / "results/statistical/wsi_bootstrap/wsi_5seed_combined.json"
    wsi = json.loads(wsi_path.read_text()) if wsi_path.exists() else {}

    eata42_note = (
        "Seed-42 EATA point estimate in Table 1 is 0.9363 from the original "
        "Gate-C salvage JSON. A later WSI-bootstrap rerun gave 0.9386 / harm 0.011 / "
        "cov 0.865. Table 1 keeps the original locked 5-seed JSON. WSI CIs use the rerun."
    )

    audit = {
        "seeds": SEEDS,
        "source_s42_is_best_ckpt_not_smoke": bool(src42_ok and smoke),
        "json_crosscheck": checks,
        "shift_auroc_drop_mean_sd": mean_sd(drops),
        "u_error_r_mean_sd": mean_sd(ur),
        "table1": table1,
        "contrasts": {k: {kk: vv for kk, vv in v.items()} for k, v in contrasts.items()},
        "eata_s42_note": eata42_note,
        "resnet50_experiments": "NOT RUN",
        "stale_excluded": [
            "smoke_s42_max20.pt",
            "tent_eata_s42_lr1e-4_COLLAPSE.json (failure-mode figure only)",
            "online-U UTTA Grad-CAM AUROC 0.324 (invalid)",
            "Camelyon17-C first-8k all-negative (invalid)",
            "EATA-C coverage 0.001 (not a working run)",
        ],
        "protocol": {
            "bn_affine_only": True,
            "bn_running_stats_frozen": True,
            "target_labels_eval_only": True,
            "tau_on_hospital_1": True,
            "conf_entropy_thresholds_hospital_1": True,
            "no_target_model_selection": True,
        },
    }

    (OUT / "phase14_audit.json").write_text(json.dumps(audit, indent=2, default=float))

    # CSV Table 2
    lines = ["Method,AUROC,F1,ECE,Harm,Coverage"]
    for r in table1:
        lines.append(
            f"{r['method']},{r['auroc']},{r['f1']},{r['ece']},{r['harm']},{r['coverage']}"
        )
    (OUT / "table2_five_seed_primary.csv").write_text("\n".join(lines) + "\n")

    # per-seed AUROC csv
    hdr = "Method," + ",".join(str(s) for s in SEEDS)
    per = [hdr]
    for name, pre, _ in methods:
        per.append(name + "," + ",".join(f"{rows[s][f'{pre}_auroc']:.4f}" for s in SEEDS))
    (OUT / "table2_per_seed_auroc.csv").write_text("\n".join(per) + "\n")

    harm_lines = ["Method," + ",".join(str(s) for s in SEEDS) + ",mean,sd"]
    for name, pre, has in methods:
        if not has:
            continue
        vals = [rows[s][f"{pre}_harm"] for s in SEEDS]
        m, s = mean_sd(vals)
        harm_lines.append(
            name + "," + ",".join(f"{v:.4f}" for v in vals) + f",{m:.4f},{s:.4f}"
        )
    (OUT / "table3_harm.csv").write_text("\n".join(harm_lines) + "\n")

    cal = ["Method,ECE mean±SD,Brier not fully stored in headline"]
    for r in table1:
        cal.append(f"{r['method']},{r['ece']}")
    (OUT / "table4_calibration.csv").write_text("\n".join(cal) + "\n")

    print("Phase 14 audit written to", OUT)
    print("source_s42 matches headline:", src42_ok, "B-FULL epoch3:", smoke)
    for c in checks:
        print(" json seed", c["seed"], "ok", c["ok"], c.get("mismatches") or "")
    print("AUROC drop", mean_sd(drops))
    print("U-r", mean_sd(ur))
    for k, v in contrasts.items():
        print(
            f"{k:32s} Δ={v['mean']:+.4f}  CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
            f"p={v['p']:.4g}  dz={v['cohen_dz']:.3f}  signs {v['n_pos']}/{5-v['n_neg']}"
        )


if __name__ == "__main__":
    main()
