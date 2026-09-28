"""Leakage tests for Camelyon17-WILDS.

Rules (Master Plan v4):
  Train WSI ∩ Validation WSI = ∅
  Train WSI ∩ Test WSI = ∅
  Validation WSI ∩ Test WSI = ∅
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_frozen_yaml_hospital_isolation():
    """Always-on check: official split hospitals are disjoint (no WILDS needed)."""
    cfg = yaml.safe_load((ROOT / "configs" / "camelyon17.yaml").read_text())
    sp = cfg["splits"]
    train = set(sp["train"]["domain_ids"])
    val = set(sp["val"]["domain_ids"])
    test = set(sp["test"]["domain_ids"])
    assert train == {0, 3, 4}
    assert val == {1}
    assert test == {2}
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert cfg["leakage_free"] is True
    assert cfg["protocol"]["head"] == "1-logit BCE"
    assert cfg["protocol"]["dropout"] == 0.5


def _get_slide_ids(subset, metadata_fields):
    meta = subset.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()
    fields = list(metadata_fields) if metadata_fields else []
    slide_idx = None
    for candidate in ["slide", "wsi", "patient"]:
        if candidate in fields:
            slide_idx = fields.index(candidate)
            break
    if slide_idx is None:
        pytest.skip(f"No slide column found in metadata_fields={fields}")
    return set(meta[:, slide_idx].astype(int).tolist())


def _get_hospital_ids(subset, metadata_fields):
    meta = subset.metadata_array
    if hasattr(meta, "numpy"):
        meta = meta.numpy()
    fields = list(metadata_fields) if metadata_fields else []
    hosp_idx = None
    for candidate in ["hospital", "domain", "center"]:
        if candidate in fields:
            hosp_idx = fields.index(candidate)
            break
    if hosp_idx is None:
        hosp_idx = 0
    return set(meta[:, hosp_idx].astype(int).tolist())


@pytest.fixture(scope="module")
def camelyon17():
    try:
        from wilds import get_dataset
    except ImportError:
        pytest.skip("wilds not installed")
    try:
        ds = get_dataset(dataset="camelyon17", download=False)
    except Exception as e:
        pytest.skip(f"Camelyon17 data not available locally: {e}")
    return ds


def test_wsi_no_overlap(camelyon17):
    fields = getattr(camelyon17, "metadata_fields", [])
    train = camelyon17.get_subset("train")
    val = camelyon17.get_subset("val")
    test = camelyon17.get_subset("test")
    train_slides = _get_slide_ids(train, fields)
    val_slides = _get_slide_ids(val, fields)
    test_slides = _get_slide_ids(test, fields)
    assert train_slides.isdisjoint(val_slides)
    assert train_slides.isdisjoint(test_slides)
    assert val_slides.isdisjoint(test_slides)


def test_hospital_separation(camelyon17):
    fields = getattr(camelyon17, "metadata_fields", [])
    for split in ["train", "val", "test"]:
        subset = camelyon17.get_subset(split)
        hospitals = _get_hospital_ids(subset, fields)
        assert len(hospitals) >= 1
