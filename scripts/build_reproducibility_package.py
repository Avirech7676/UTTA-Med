#!/usr/bin/env python3
"""Build clean, self-contained camera-ready reproducibility zip package for UTTA-Med v2."""

import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ZIP = ROOT / "UTTA_MED_V2_REPRODUCIBILITY_PACKAGE.zip"

INCLUDE_DIRS = [
    "src",
    "configs",
    "tests",
    "scripts",
    "paper/tables",
    "paper/figures",
    "paper/ieee",
    "results/tables",
    "results/kaggle_training",
]

INCLUDE_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "INSTALL.md",
    "pytest.ini",
    ".gitignore",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".DS_Store",
    "*.pyc",
    "*.pt",      # exclude large binary weights from zip (provided via Kaggle / download manifest)
    "*.pth",
    "*.h5",
    "*.hdf5",
]

def should_exclude(path_str: str) -> bool:
    for pat in EXCLUDE_PATTERNS:
        if pat.startswith("*"):
            ext = pat[1:]
            if path_str.endswith(ext):
                return True
        elif pat in path_str.split(os.sep) or pat in path_str.split("/"):
            return True
    return False

def build_package():
    print(f"[*] Assembling UTTA-Med v2 Reproducibility Package: {OUTPUT_ZIP.name}")
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()

    total_files = 0
    total_uncompressed_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add individual files
        for rel_file in INCLUDE_FILES:
            f = ROOT / rel_file
            if f.exists():
                arcname = f"UTTA_MED_V2/{rel_file}"
                zf.write(f, arcname)
                total_files += 1
                total_uncompressed_bytes += f.stat().st_size
                print(f"  [+] Added: {rel_file}")

        # 2. Add directories
        for rel_dir in INCLUDE_DIRS:
            d = ROOT / rel_dir
            if not d.exists():
                continue
            for root, dirs, files in os.walk(d):
                for file in files:
                    full_path = Path(root) / file
                    rel_to_root = full_path.relative_to(ROOT)
                    rel_str = str(rel_to_root)

                    if should_exclude(rel_str):
                        continue

                    arcname = f"UTTA_MED_V2/{rel_str.replace(os.sep, '/')}"
                    zf.write(full_path, arcname)
                    total_files += 1
                    total_uncompressed_bytes += full_path.stat().st_size

    zip_size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    raw_size_mb = total_uncompressed_bytes / (1024 * 1024)

    print("\n" + "=" * 70)
    print(f"[+] PACKAGE SUCCESSFULLY CREATED: {OUTPUT_ZIP}")
    print(f"    - Files Included:            {total_files}")
    print(f"    - Raw Uncompressed Size:     {raw_size_mb:.2f} MB")
    print(f"    - Compressed ZIP Size:       {zip_size_mb:.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    build_package()
