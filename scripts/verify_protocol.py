import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import numpy as np
from src.data.camelyon17 import create_camelyon17_datasets
from src.data.paths import find_parquet_data_dir, DEFAULT_DATA_ROOT


def collect_centers(dataset, n=500):
    centers = set()
    limit = min(n, len(dataset))
    # Sample uniformly across shards rather than just the first contiguous rows
    indices = np.linspace(0, len(dataset) - 1, limit, dtype=int)

    for idx in indices:
        _, _, metadata = dataset[int(idx)]
        if isinstance(metadata, dict):
            centers.add(int(metadata["center"]))
        elif hasattr(metadata, "__getitem__"):
            centers.add(int(metadata[0]))
        else:
            centers.add(int(metadata))

    return centers


def main():
    print("=" * 70)
    print("UTTA-Med: Protocol / Leakage Verification")
    print("=" * 70)

    env_root = os.environ.get("CAMELYON17_ROOT")
    start = Path(env_root) if env_root else Path(DEFAULT_DATA_ROOT)
    data_root = find_parquet_data_dir(start)
    if data_root is None:
        print(f"[SKIP] Camelyon17 Parquet data not found at {start}.")
        print("Protocol check skipped (offline mode).")
        return

    print(f"Data root: {data_root}")

    datasets = create_camelyon17_datasets(
        root=data_root,
        train_transform=None,
        eval_transform=None,
    )

    source_train, id_val, ood_val, ood_test = datasets

    print("\nDataset split counts:")
    print(f"  Source train   : {len(source_train):,} (expected: 302,436)")
    print(f"  ID validation  : {len(id_val):,} (expected: 33,560)")
    print(f"  OOD validation : {len(ood_val):,} (expected: 34,904)")
    print(f"  Target test    : {len(ood_test):,} (expected: 85,054)")

    assert len(source_train) == 302436, f"Expected 302,436, got {len(source_train)}"
    assert len(id_val) == 33560, f"Expected 33,560, got {len(id_val)}"
    assert len(ood_val) == 34904, f"Expected 34,904, got {len(ood_val)}"
    assert len(ood_test) == 85054, f"Expected 85,054, got {len(ood_test)}"

    expected = {
        "source_train": {0, 3, 4},
        "id_val": {0, 3, 4},
        "ood_val": {1},
        "ood_test": {2},
    }

    actual = {
        "source_train": collect_centers(source_train),
        "id_val": collect_centers(id_val),
        "ood_val": collect_centers(ood_val),
        "ood_test": collect_centers(ood_test),
    }

    for name in expected:
        print(f"\n{name}")
        print(f"  Expected centers: {sorted(expected[name])}")
        print(f"  Observed centers: {sorted(actual[name])}")

        assert actual[name] == expected[name], (
            f"Center mismatch for {name}: expected {expected[name]}, got {actual[name]}"
        )

    assert actual["ood_val"].isdisjoint(actual["ood_test"]), "OOD val and OOD test overlap!"
    assert actual["source_train"].isdisjoint(actual["ood_val"]), "Source train and OOD val overlap!"
    assert actual["source_train"].isdisjoint(actual["ood_test"]), "Source train and OOD test overlap!"

    print("\n" + "=" * 70)
    print("Status: PROTOCOL VERIFICATION PASSED")
    print("No cross-center leakage detected.")
    print("=" * 70)


if __name__ == "__main__":
    main()
