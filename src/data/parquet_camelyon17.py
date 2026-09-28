"""HuggingFace parquet Camelyon17-WILDS loader.

Matches dataset wltjr1007/Camelyon17-WILDS (and identical parquet dumps):

  data/train-*.parquet
  data/validation-*.parquet
  data/test-*.parquet

Columns: image, label, center, image_id, patient, node, x_coord, y_coord, slide

The HF "validation" split concatenates official WILDS val + id_val.
We split it by `center`: hospital 1 → OOD val; hospitals 0/3/4 → id_val.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


def _glob_split(data_dir: Path, split: str) -> List[Path]:
    if split == "train":
        files = sorted(data_dir.glob("train-*.parquet"))
    elif split in ("validation", "val"):
        files = sorted(data_dir.glob("validation-*.parquet"))
    elif split == "test":
        files = sorted(data_dir.glob("test-*.parquet"))
    else:
        files = []
    if not files:
        raise FileNotFoundError(f"No parquet files for split={split} in {data_dir}")
    return files


def _to_pil(cell) -> Image.Image:
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    if hasattr(cell, "as_py"):
        cell = cell.as_py()
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        path = cell.get("path")
        if path:
            return Image.open(path).convert("RGB")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    raise TypeError(f"Cannot decode image cell of type {type(cell)}")


def _read_meta(files: Sequence[Path]) -> Dict[str, np.ndarray]:
    centers, slides, labels, patients, file_ids, rows = [], [], [], [], [], []
    for fi, path in enumerate(files):
        table = pq.read_table(path, columns=["label", "center", "slide", "patient"])
        n = table.num_rows
        centers.append(np.asarray(table["center"].to_pylist(), dtype=np.int64))
        slides.append(np.asarray(table["slide"].to_pylist(), dtype=np.int64))
        labels.append(np.asarray(table["label"].to_pylist(), dtype=np.int64))
        patients.append(np.asarray(table["patient"].to_pylist(), dtype=np.int64))
        file_ids.append(np.full(n, fi, dtype=np.int32))
        rows.append(np.arange(n, dtype=np.int32))
    return {
        "center": np.concatenate(centers),
        "slide": np.concatenate(slides),
        "y": np.concatenate(labels),
        "patient": np.concatenate(patients),
        "file_id": np.concatenate(file_ids),
        "row": np.concatenate(rows),
    }


@dataclass
class _RowGroupIndex:
    starts: np.ndarray
    sizes: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "_RowGroupIndex":
        pf = pq.ParquetFile(path)
        sizes = np.array(
            [pf.metadata.row_group(i).num_rows for i in range(pf.num_row_groups)],
            dtype=np.int32,
        )
        starts = np.zeros(len(sizes) + 1, dtype=np.int32)
        starts[1:] = np.cumsum(sizes)
        return cls(starts=starts, sizes=sizes)

    def locate(self, row: int) -> tuple[int, int]:
        rg = int(np.searchsorted(self.starts, row, side="right") - 1)
        return rg, int(row - self.starts[rg])


class ParquetSplit(Dataset):
    """One official split. Returns (image, y, metadata) like WILDS."""

    def __init__(
        self,
        files: List[Path],
        meta: Dict[str, np.ndarray],
        mask: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.files = list(files)
        self.transform = transform
        if mask is None:
            mask = np.ones(len(meta["y"]), dtype=bool)
        self.file_id = meta["file_id"][mask]
        self.row = meta["row"][mask]
        self.y = meta["y"][mask]
        self.center = meta["center"][mask]
        self.slide = meta["slide"][mask]
        self.patient = meta["patient"][mask]
        self._metadata_array = np.stack([self.center, self.slide, self.y], axis=1)
        self._rg_index: Dict[int, _RowGroupIndex] = {}
        self._rg_cache: Dict[tuple, object] = {}
        self._cache_order: List[tuple] = []
        self._cache_limit = 4

    def __len__(self) -> int:
        return int(len(self.y))

    @property
    def metadata_array(self):
        return torch.from_numpy(self._metadata_array)

    @property
    def y_array(self):
        return torch.from_numpy(self.y)

    def _row_group_table(self, file_id: int, rg: int):
        key = (file_id, rg)
        if key in self._rg_cache:
            return self._rg_cache[key]
        path = self.files[file_id]
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(rg, columns=["image", "label", "center", "slide"])
        self._rg_cache[key] = table
        self._cache_order.append(key)
        if len(self._cache_order) > self._cache_limit:
            old = self._cache_order.pop(0)
            self._rg_cache.pop(old, None)
        return table

    def __getitem__(self, idx: int):
        fi = int(self.file_id[idx])
        row = int(self.row[idx])
        if fi not in self._rg_index:
            self._rg_index[fi] = _RowGroupIndex.from_file(self.files[fi])
        rg, local = self._rg_index[fi].locate(row)
        table = self._row_group_table(fi, rg)
        img = _to_pil(table["image"][local])
        if self.transform is not None:
            img = self.transform(img)
        y = torch.tensor(int(self.y[idx]), dtype=torch.float32)
        meta = torch.tensor(
            [int(self.center[idx]), int(self.slide[idx]), int(self.y[idx])],
            dtype=torch.long,
        )
        return img, y, meta


class ParquetCamelyon17:
    """WILDS-like handle: get_subset(name), metadata_fields."""

    metadata_fields = ["hospital", "slide", "y"]

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self._files = {
            "train": _glob_split(self.data_dir, "train"),
            "validation": _glob_split(self.data_dir, "validation"),
            "test": _glob_split(self.data_dir, "test"),
        }
        self._meta: Dict[str, Dict[str, np.ndarray]] = {}
        print(f"Indexing parquet metadata in {self.data_dir} ...", flush=True)
        for name, files in self._files.items():
            self._meta[name] = _read_meta(files)
            n = len(self._meta[name]["y"])
            centers = sorted(set(self._meta[name]["center"].tolist()))
            print(f"  {name:12s} n={n:7d}  centers={centers}", flush=True)

    def get_subset(self, split: str, transform=None):
        if split == "train":
            return ParquetSplit(self._files["train"], self._meta["train"], transform=transform)
        if split == "test":
            return ParquetSplit(self._files["test"], self._meta["test"], transform=transform)
        if split in ("val", "validation"):
            meta = self._meta["validation"]
            mask = meta["center"] == 1
            if mask.sum() == 0:
                raise RuntimeError(
                    "OOD validation Center 1 is missing. "
                    "Refusing to use mixed validation data."
                )
            return ParquetSplit(self._files["validation"], meta, mask=mask, transform=transform)
        if split == "id_val":
            meta = self._meta["validation"]
            mask = np.isin(meta["center"], [0, 3, 4])
            if mask.sum() == 0:
                raise KeyError("id_val: no source-hospital rows in validation parquet")
            return ParquetSplit(self._files["validation"], meta, mask=mask, transform=transform)
        raise KeyError(f"Unknown split {split!r}")
