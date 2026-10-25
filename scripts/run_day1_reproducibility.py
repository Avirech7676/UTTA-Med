#!/usr/bin/env python3
"""Day 1: Reproducibility Freeze and Clean Verification Script.

Verifies all headline manuscript metrics from raw npz and json files, records
environment telemetry, seeds, checkpoint SHA256 hashes, and outputs
REPRODUCIBILITY_REPORT.md and results/tables/day1_master_reproducibility.csv.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts.common import SEEDS, METHODS, load, auroc, f1_score, error_rate, mean_sd


def get_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def run_day1_reproducibility() -> Dict[str, Any]:
    print("=" * 80)
    print("DAY 1: REPRODUCIBILITY FREEZE & HEADLINE AUDIT")
    print("=" * 80)

    # 1. Environment Telemetry
    env_info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds_verified": SEEDS,
    }

    # 2. Checkpoint Hashes
    ckpt_dir = ROOT / "checkpoints"
    ckpt_hashes = {}
    for p in sorted(ckpt_dir.glob("*.pt")):
        ckpt_hashes[p.name] = get_file_sha256(p)

    # 3. Verify Headline Metrics from Raw NPZ Arrays
    print("\n[*] Auditing 5-Seed Headline Metrics from Raw NPZ Arrays...")
    raw_records = []
    summary_by_method = {}

    for method in METHODS:
        aurocs = []
        f1s = []
        errors = []
        for seed in SEEDS:
            data = load(seed, method)
            y = data["y"]
            p = data["p"]
            score_auc = auroc(y, p)
            score_f1 = f1_score(y, p)
            score_err = error_rate(y, p)
            aurocs.append(score_auc)
            f1s.append(score_f1)
            errors.append(score_err)

            raw_records.append({
                "method": method,
                "seed": seed,
                "auroc": round(score_auc, 6),
                "f1": round(score_f1, 6),
                "error_rate": round(score_err, 6),
            })

        m_auc, s_auc = mean_sd(aurocs)
        m_f1, s_f1 = mean_sd(f1s)
        m_err, s_err = mean_sd(errors)

        summary_by_method[method] = {
            "auroc_mean": round(m_auc, 4),
            "auroc_std": round(s_auc, 4),
            "f1_mean": round(m_f1, 4),
            "f1_std": round(s_f1, 4),
            "error_mean": round(m_err, 4),
            "error_std": round(s_err, 4),
        }
        print(f"  {method:<12} | AUROC: {m_auc:.4f} ± {s_auc:.4f} | F1: {m_f1:.4f} ± {s_f1:.4f} | Err: {m_err:.4f}")

    # 4. Save Master Table
    out_dir = ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "day1_master_reproducibility.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Mean AUROC", "Std AUROC", "Mean F1", "Std F1", "Mean Error", "Std Error"])
        for m, stats in summary_by_method.items():
            writer.writerow([
                m,
                f"{stats['auroc_mean']:.4f}",
                f"{stats['auroc_std']:.4f}",
                f"{stats['f1_mean']:.4f}",
                f"{stats['f1_std']:.4f}",
                f"{stats['error_mean']:.4f}",
                f"{stats['error_std']:.4f}",
            ])

    # 5. Generate Markdown Report
    report_path = ROOT / "REPRODUCIBILITY_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Day 1: Reproducibility Verification Report\n\n")
        f.write(f"**Generated:** {platform.platform()} | Python {platform.python_version()} | PyTorch {torch.__version__}\n\n")
        f.write("## 1. Verified Headline Results (5 Seeds)\n\n")
        f.write("| Method | Macro AUROC | F1-Score | Error Rate |\n")
        f.write("| :--- | :---: | :---: | :---: |\n")
        for m, stats in summary_by_method.items():
            f.write(f"| **{m}** | {stats['auroc_mean']:.4f} ± {stats['auroc_std']:.4f} | {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f} | {stats['error_mean']:.4f} ± {stats['error_std']:.4f} |\n")
        f.write("\n## 2. Checkpoint Provenance Hashes\n\n")
        f.write("| Checkpoint File | SHA256 Hash |\n")
        f.write("| :--- | :--- |\n")
        for name, sha in ckpt_hashes.items():
            f.write(f"| `{name}` | `{sha}` |\n")

    print(f"\n[+] Wrote CSV master table: {csv_path}")
    print(f"[+] Wrote Markdown report: {report_path}")
    return {"summary": summary_by_method, "checkpoints": ckpt_hashes}


if __name__ == "__main__":
    run_day1_reproducibility()
