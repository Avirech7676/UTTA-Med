#!/usr/bin/env python3
"""Checkpoint verification and automated downloader for SafeAdapt-TTA.

Verifies SHA-256 checksums of model checkpoints required for evaluation
and table reproduction. If checkpoints are missing, downloads them from
the public open-science release mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Tuple

# Official cryptographic hashes for all released checkpoints
CHECKPOINT_CATALOG: Dict[str, Tuple[str, str]] = {
    "camelyon17_resnet18_source_s42_BEST.pt": (
        "f0b85ec1a89707a8ad1807a544280ac2302a572abd0ba84233dff6d4dd35d000",
        "https://zenodo.org/record/ANONYMIZED/files/camelyon17_resnet18_source_s42_BEST.pt",
    ),
    "camelyon17_resnet18_source_s7_BEST.pt": (
        "9c18d65ab0a97527eac12be8ce60b94a87720c357166cc6805364e52d1cf9a1b",
        "https://zenodo.org/record/ANONYMIZED/files/camelyon17_resnet18_source_s7_BEST.pt",
    ),
    "camelyon17_resnet18_source_s123_BEST.pt": (
        "cb57d805869f1176d0c68762bf62f5510fb0492a9012bcf7af6e5c7b60446224",
        "https://zenodo.org/record/ANONYMIZED/files/camelyon17_resnet18_source_s123_BEST.pt",
    ),
    "camelyon17_resnet18_source_s2024_BEST.pt": (
        "d9bed5d9bc05edd7c060610d73a135d6daa56a2a2238a4873b4ea45a2e32c261",
        "https://zenodo.org/record/ANONYMIZED/files/camelyon17_resnet18_source_s2024_BEST.pt",
    ),
    "camelyon17_resnet18_source_s99_BEST.pt": (
        "0b299932abd1841901b2e5bbee5c0735f3c79be9790b7feb9019c3094f42f540",
        "https://zenodo.org/record/ANONYMIZED/files/camelyon17_resnet18_source_s99_BEST.pt",
    ),
    "midog_resnet18_s42_BEST.pt": (
        "409a9c7d3b9d10523221affc477f5a85660009fabd6359999a5b9b95a07c0287",
        "https://zenodo.org/record/ANONYMIZED/files/midog_resnet18_s42_BEST.pt",
    ),
    "nctcrc_resnet18_s42_BEST.pt": (
        "92631223343f5a7770b4da7671b9c7f2f26eee052c00807f263cfb4300b33d8e",
        "https://zenodo.org/record/ANONYMIZED/files/nctcrc_resnet18_s42_BEST.pt",
    ),
    "pathmnist_r18_s42.pt": (
        "ce2cfcf0db4378f56dc3013e079d3095f150e91f0013639bb6c5774655703f73",
        "https://zenodo.org/record/ANONYMIZED/files/pathmnist_r18_s42.pt",
    ),
    "source_centroids_s42.pt": (
        "08340599c894de6ae43a57b40a90b96953934e9a2a3841742cc387c83956c2ec",
        "https://zenodo.org/record/ANONYMIZED/files/source_centroids_s42.pt",
    ),
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_or_download(target_dir: Path, download: bool = False) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    print("=" * 80)
    print("SAFEADAPT-TTA CHECKPOINT PROVENANCE & INTEGRITY AUDITOR")
    print("=" * 80)

    for filename, (expected_hash, url) in CHECKPOINT_CATALOG.items():
        filepath = target_dir / filename
        if not filepath.exists():
            if download and "ANONYMIZED" not in url:
                print(f"[MISSING] Downloading {filename} from {url} ...")
                try:
                    urllib.request.urlretrieve(url, filepath)
                except Exception as e:
                    print(f"  FAILED to download {filename}: {e}")
                    all_ok = False
                    continue
            else:
                print(f"[{'MISSING':7s}] {filename} (not present in {target_dir})")
                all_ok = False
                continue

        actual_hash = compute_sha256(filepath)
        if actual_hash == expected_hash:
            print(f"[{'VERIFIED':7s}] {filename} | SHA256 matches bit-exact")
        else:
            print(f"[{'CORRUPT':7s}] {filename}")
            print(f"  Expected: {expected_hash}")
            print(f"  Actual:   {actual_hash}")
            all_ok = False

    print("=" * 80)
    if all_ok:
        print("ALL CHECKPOINTS VERIFIED SUCCESSFULLY (100% BIT-EXACT MATCH).")
    else:
        print("WARNING: Some checkpoints are missing or corrupt.")
    print("=" * 80)
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify or download SafeAdapt model checkpoints.")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Path to checkpoints directory (default: checkpoints/)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing checkpoints from public mirror if available",
    )
    args = parser.parse_args()

    success = verify_or_download(args.checkpoints_dir, download=args.download)
    sys.exit(0 if success else 1)
