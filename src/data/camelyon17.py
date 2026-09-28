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
from torchvision import transforms

from src.data.paths import find_parquet_data_dir, resolve_camelyon17_paths

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((96, 96)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_camelyon17(
    data_root: Optional[str] = None,
    download: bool = False,
    split_scheme: str = "official",
):
    """Load Camelyon17.

    Prefers local HuggingFace parquet (what you have on Desktop).
    Falls back to official wilds PNG layout if metadata.csv + patches/ exist.

    Returns (dataset, root, version_or_data_dir).
    """
    from pathlib import Path
    import os

    explicit = data_root or os.environ.get("CAMELYON17_ROOT")
    start = Path(explicit) if explicit else Path(r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data")

    parquet_dir = find_parquet_data_dir(start)
    if parquet_dir is not None:
        from src.data.parquet_camelyon17 import ParquetCamelyon17

        print(f"Using HuggingFace parquet backend: {parquet_dir}")
        dataset = ParquetCamelyon17(parquet_dir)
        return dataset, parquet_dir.parent, parquet_dir

    from wilds import get_dataset

    wilds_root, version_dir = resolve_camelyon17_paths(data_root)
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
    return dataset.get_subset(split, transform=get_transform(train=train))


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
    dataset, _, _ = load_camelyon17(data_root=root)
    source_train = dataset.get_subset("train", transform=train_transform)
    id_val = dataset.get_subset("id_val", transform=eval_transform)
    ood_val = dataset.get_subset("val", transform=eval_transform)
    ood_test = dataset.get_subset("test", transform=eval_transform)
    return source_train, id_val, ood_val, ood_test
