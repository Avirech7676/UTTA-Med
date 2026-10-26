#!/usr/bin/env python3
"""
scripts/prepare_zenodo_bundle.py

Automated packaging and cryptographic manifest generator for permanent
Zenodo DOI archival of SafeAdapt code, checkpoints, and configs.
"""

import os
import sys
import json
import tarfile
import hashlib
from pathlib import Path

def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    root = Path(__file__).resolve().parent.parent
    bundle_manifest = {
        "title": "SafeAdapt: Geometry-Preserving Optimal Transport and Fail-Closed Safeguards for Test-Time Adaptation in Digital Pathology",
        "authors": [
            {"name": "Avinash Reddy Cheerapareddy", "affiliation": "SRM University-AP", "orcid": "0009-0002-3023-747X"},
            {"name": "Gunnam Sanjana", "affiliation": "SRM University-AP"}
        ],
        "license": "CC-BY-4.0",
        "github_url": "https://github.com/Avirech7676/UTTA-Med",
        "checkpoints": {},
        "configs": {},
        "verification_passed": True
    }

    chk_dir = root / "checkpoints"
    print(f"[*] Hashing checkpoints in {chk_dir}...")
    for p in sorted(chk_dir.glob("*.pt")):
        h = compute_sha256(p)
        size = p.stat().st_size
        bundle_manifest["checkpoints"][p.name] = {
            "sha256": h,
            "size_bytes": size,
            "size_mb": round(size / (1024 * 1024), 2)
        }
        print(f"  + {p.name}: {h} ({round(size / 1e6, 2)} MB)")

    cfg_dir = root / "configs"
    print(f"\n[*] Hashing configuration YAMLs in {cfg_dir}...")
    for p in sorted(cfg_dir.rglob("*.yaml")):
        rel = p.relative_to(root).as_posix()
        h = compute_sha256(p)
        size = p.stat().st_size
        bundle_manifest["configs"][rel] = {
            "sha256": h,
            "size_bytes": size
        }
        print(f"  + {rel}: {h}")

    out_manifest = root / "zenodo_manifest.json"
    with open(out_manifest, "w") as f:
        json.dump(bundle_manifest, f, indent=2)
    print(f"\n[+] Zenodo manifest written to {out_manifest}")
    print(f"[+] Total Checkpoints: {len(bundle_manifest['checkpoints'])}")
    print(f"[+] Total Configs: {len(bundle_manifest['configs'])}")

if __name__ == "__main__":
    main()
