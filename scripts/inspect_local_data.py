#!/usr/bin/env python3
"""Inspect local Camelyon17 data WITHOUT using test labels for any decision.

Supports:
  - HuggingFace parquet dump (your Desktop folder)
  - official WILDS PNG layout (metadata.csv + patches/)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    data_root = args.data_root or os.environ.get("CAMELYON17_ROOT") or DEFAULT

    from src.data.camelyon17 import inspect_splits, load_camelyon17
    from src.data.paths import find_parquet_data_dir

    print("=" * 64)
    print("UTTA-Med  local Camelyon17 inspection")
    print("=" * 64)
    print(f"data-root : {data_root}")

    parquet_dir = find_parquet_data_dir(Path(data_root))
    if parquet_dir:
        print(f"layout    : HuggingFace parquet")
        print(f"parquet   : {parquet_dir}")
    else:
        print("layout    : looking for WILDS metadata.csv + patches/")

    dataset, root, version_dir = load_camelyon17(data_root=data_root, download=args.download)
    info = inspect_splits(dataset)

    print(f"backend        : {info.get('backend')}")
    print(f"metadata_fields: {info['metadata_fields']}")
    print(f"hospital field : {info['hospital_field']}")
    print(f"slide field    : {info['slide_field']}")
    print(f"\n{'split':10s} {'n':>8s} {'#WSI':>6s} {'hospitals':22s} {'tumor':>8s} {'non':>8s}")
    for name, s in info["splits"].items():
        print(
            f"{name:10s} {s['n_samples']:8d} {s['n_wsis']:6d} "
            f"{str(s['domain_ids']):22s} {s['n_tumor']:8d} {s['n_nontumor']:8d}"
        )
    print("\nWSI intersections (must all be 0):")
    for k, n in info["wsi_intersections"].items():
        flag = "OK" if n == 0 else "LEAKAGE"
        print(f"  {k:16s} {n:5d}  {flag}")
    print(f"\nleakage_free = {info['leakage_free']}")

    yaml_path = ROOT / "configs" / "camelyon17.yaml"
    existing = {}
    if yaml_path.exists():
        existing = yaml.safe_load(yaml_path.read_text()) or {}
    existing["dataset"] = "camelyon17"
    existing["backend"] = info.get("backend")
    existing["root"] = str(root)
    existing["version_dir"] = str(version_dir)
    existing["download"] = False
    existing["metadata_fields"] = info["metadata_fields"]
    existing["splits"] = {}
    role = {
        "train": "SOURCE TRAIN (hospitals 0/3/4 in official WILDS)",
        "val": "OOD VALIDATION hospital 1 — τ / hyperparameter selection only",
        "test": "TARGET hospital 2 — unlabeled adaptation + final evaluation only",
        "id_val": "In-domain validation (source hospitals, held-out patches)",
    }
    for name, s in info["splits"].items():
        existing["splits"][name] = {
            "domain_ids": s["domain_ids"],
            "n_samples": s["n_samples"],
            "n_wsis": s["n_wsis"],
            "hospital_counts": s["hospital_counts"],
            "role": role.get(name, name),
        }
    existing["leakage_free"] = info["leakage_free"]
    existing["notes"] = (
        "Mapping frozen from local files (parquet or WILDS). "
        "Do not hand-edit domain_ids. HF validation parquet is split by center: "
        "center=1 → val (OOD), centers 0/3/4 → id_val."
    )
    yaml_path.write_text(yaml.safe_dump(existing, sort_keys=False))
    print(f"\nWrote verified mapping → {yaml_path}")

    out_json = ROOT / "results" / "data" / "camelyon17_metadata_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"inspection": info, "root": str(root)}, indent=2, default=str))
    print(f"Wrote JSON summary     → {out_json}")
    print("=" * 64)


if __name__ == "__main__":
    main()
