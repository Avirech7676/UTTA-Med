#!/usr/bin/env python3
"""Print what is actually inside the Camelyon17 folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("CAMELYON17_ROOT", DEFAULT))
    args = p.parse_args()
    root = Path(args.data_root)

    print("=" * 64)
    print("UTTA-Med data folder diagnostic")
    print("=" * 64)
    print(f"path     : {root}")
    print(f"exists   : {root.exists()}")
    if not root.exists():
        print("Folder does not exist. Check the path / OneDrive sync.")
        return

    data = root / "data"
    parquet = sorted(data.glob("*.parquet")) if data.is_dir() else []
    print(f"data/    : {data.is_dir()}")
    print(f"parquet  : {len(parquet)} files")
    for f in parquet[:8]:
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    if len(parquet) > 8:
        print(f"  ... {len(parquet) - 8} more")

    if parquet:
        print("\nTHIS IS THE HUGGINGFACE PARQUET LAYOUT — correct for UTTA-Med.")
        print("Next:")
        print('  python -m pip install pyarrow')
        print(f'  python scripts\\inspect_local_data.py --data-root "{root}"')
    else:
        print("\nNo train-*.parquet found. Listing top-level:")
        for item in sorted(root.iterdir()):
            print(" ", "<DIR>" if item.is_dir() else "     ", item.name)
    print("=" * 64)


if __name__ == "__main__":
    main()
