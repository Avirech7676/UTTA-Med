"""Camelyon17 loader — auto-detects HuggingFace parquet OR official WILDS PNG layout.

Hospital / domain IDs are read from the files — never from papers.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.paths import find_parquet_data_dir, resolve_camelyon17_paths
from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_camelyon17_eval_transform,
    get_camelyon17_train_transform,
)


OFFICIAL_SOURCE_CENTERS = (0, 3, 4)
OFFICIAL_ID_VAL_CENTERS = (0, 3, 4)
OFFICIAL_OOD_VAL_CENTER = 1
OFFICIAL_TARGET_TEST_CENTER = 2


def validate_source_centers(centers) -> None:
    """Validate that source training contains only official source centers {0, 3, 4}.

    Fails closed with ValueError if Center 1 (OOD val) or Center 2 (target test) is included.
    """
    centers_set = set(int(c) for c in centers)
    forbidden = centers_set & {OFFICIAL_OOD_VAL_CENTER, OFFICIAL_TARGET_TEST_CENTER}
    if forbidden:
        raise ValueError(
            f"Protocol violation: Centers {sorted(forbidden)} are strictly forbidden in source training data. "
            f"Center 1 is reserved exclusively for OOD validation and Center 2 for target test evaluation. "
            f"Received source centers: {sorted(centers_set)}."
        )
    if not centers_set.issubset(set(OFFICIAL_SOURCE_CENTERS)):
        raise ValueError(
            f"Invalid source training centers: {sorted(centers_set)}. "
            f"Source training centers must be a subset of official source centers {list(OFFICIAL_SOURCE_CENTERS)}."
        )


def get_transform(train: bool):
    """Delegate to canonical transform definitions in src.data.transforms."""
    return get_camelyon17_train_transform() if train else get_camelyon17_eval_transform()


def load_camelyon17(
    data_root: Optional[str] = None,
    download: bool = False,
    split_scheme: str = "official",
):
    """Load Camelyon17.

    Prefers local HuggingFace parquet.
    Falls back to official wilds PNG layout if metadata.csv + patches/ exist.

    Returns (dataset, root, version_or_data_dir).
    """
    from pathlib import Path
    import os

    explicit = data_root or os.environ.get("CAMELYON17_ROOT")
    if not explicit:
        raise RuntimeError(
            "CAMELYON17_ROOT is not set. "
            "Please set the CAMELYON17_ROOT environment variable "
            "or pass --data-root."
        )
    start = Path(explicit).expanduser()

    parquet_dir = find_parquet_data_dir(start)
    if parquet_dir is not None:
        from src.data.parquet_camelyon17 import ParquetCamelyon17

        print(f"Using HuggingFace parquet backend: {parquet_dir}")
        dataset = ParquetCamelyon17(parquet_dir)
        return dataset, parquet_dir.parent, parquet_dir

    from wilds import get_dataset

    wilds_root, version_dir = resolve_camelyon17_paths(str(start))
    try:
        dataset = get_dataset(
            dataset="camelyon17",
            root_dir=str(wilds_root),
            download=download,
            split_scheme=split_scheme,
        )
    except Exception as e:
        raise FileNotFoundError(
            "Could not load Camelyon17 as parquet or as WILDS PNG layout.\n"
            f"  looked at {start}\n"
            "Need either:\n"
            "  <root>\\data\\train-*.parquet   (HuggingFace dump — your current files)\n"
            "or:\n"
            "  <root>\\camelyon17_v1.0\\metadata.csv + patches\\\n"
            f"Original error: {e}"
        ) from e
    return dataset, wilds_root, version_dir


def get_split(dataset, split: str, train: bool = False):
    subset = dataset.get_subset(split, transform=get_transform(train=train))
    if split == "train":
        if hasattr(subset, "center"):
            validate_source_centers(np.unique(subset.center))
        elif hasattr(subset, "dataset") and hasattr(subset.dataset, "_metadata_array"):
            meta = subset.dataset._metadata_array[subset.indices]
            validate_source_centers(np.unique(meta[:, 0]))
    return subset


def make_loader(
    subset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    import sys

    if sys.platform == "win32":
        num_workers = min(int(num_workers), 2)
    kwargs = dict(
        dataset=subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def _as_numpy(meta) -> np.ndarray:
    if hasattr(meta, "numpy"):
        return meta.numpy()
    return np.asarray(meta)


def inspect_splits(dataset) -> Dict:
    """Summarize official splits and WSI leakage. Does not use labels for decisions."""
    fields = list(getattr(dataset, "metadata_fields", []))
    hosp_field = next((f for f in ("hospital", "domain", "center") if f in fields), fields[0] if fields else None)
    slide_field = next((f for f in ("slide", "wsi", "patient") if f in fields), None)
    y_field = "y" if "y" in fields else None

    hosp_idx = fields.index(hosp_field) if hosp_field in fields else 0
    slide_idx = fields.index(slide_field) if slide_field in fields else None
    y_idx = fields.index(y_field) if y_field in fields else None

    splits: Dict[str, Dict] = {}
    slide_sets: Dict[str, set] = {}

    available = []
    for name in ("train", "val", "test", "id_val"):
        try:
            subset = dataset.get_subset(name, transform=None)
            available.append(name)
        except Exception:
            continue

        meta = _as_numpy(subset.metadata_array)
        n = int(meta.shape[0])
        domain_ids = sorted({int(x) for x in meta[:, hosp_idx].tolist()})
        hospital_counts = {str(k): int(v) for k, v in Counter(int(x) for x in meta[:, hosp_idx].tolist()).items()}

        if slide_idx is not None:
            slides = {int(x) for x in meta[:, slide_idx].tolist()}
        else:
            slides = set()
        slide_sets[name] = slides

        n_tumor = n_nontumor = None
        if y_idx is not None:
            ys = meta[:, y_idx].astype(int)
            n_tumor = int((ys == 1).sum())
            n_nontumor = int((ys == 0).sum())
        elif hasattr(subset, "y_array"):
            ys = _as_numpy(subset.y_array).astype(int).reshape(-1)
            n_tumor = int((ys == 1).sum())
            n_nontumor = int((ys == 0).sum())

        splits[name] = {
            "n_samples": n,
            "n_wsis": len(slides),
            "domain_ids": domain_ids,
            "hospital_counts": hospital_counts,
            "n_tumor": n_tumor,
            "n_nontumor": n_nontumor,
        }

    intersections = {
        "train∩val": len(slide_sets.get("train", set()) & slide_sets.get("val", set())),
        "train∩test": len(slide_sets.get("train", set()) & slide_sets.get("test", set())),
        "val∩test": len(slide_sets.get("val", set()) & slide_sets.get("test", set())),
    }
    leakage_free = all(v == 0 for v in intersections.values()) and "train" in splits and "test" in splits

    return {
        "metadata_fields": fields,
        "hospital_field": hosp_field,
        "slide_field": slide_field,
        "available_splits": available,
        "splits": splits,
        "wsi_intersections": intersections,
        "leakage_free": leakage_free,
        "backend": type(dataset).__name__,
    }


def create_camelyon17_datasets(
    root: Optional[str | Path] = None,
    train_transform=None,
    eval_transform=None,
):
    """Convenience helper returning (source_train, id_val, ood_val, ood_test)."""
    if train_transform is None:
        train_transform = get_camelyon17_train_transform()
    if eval_transform is None:
        eval_transform = get_camelyon17_eval_transform()
    dataset, _, _ = load_camelyon17(data_root=root)
    source_train = dataset.get_subset("train", transform=train_transform)
    id_val = dataset.get_subset("id_val", transform=eval_transform)
    ood_val = dataset.get_subset("val", transform=eval_transform)
    ood_test = dataset.get_subset("test", transform=eval_transform)

    # Defensive check: ensure source_train contains only official source centers
    if hasattr(source_train, "center"):
        validate_source_centers(np.unique(source_train.center))
    elif hasattr(source_train, "metadata_array"):
        meta = _as_numpy(source_train.metadata_array)
        if meta.ndim == 2 and meta.shape[1] >= 1:
            validate_source_centers(np.unique(meta[:, 0]))

    return source_train, id_val, ood_val, ood_test
