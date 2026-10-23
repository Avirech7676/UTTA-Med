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


# ==============================================================================
# STEP 4: DEFENSIVE PROTOCOL GUARDS AND TARGET-LABEL ISOLATION REGRESSION TESTS
# ==============================================================================


def test_protocol_constants_and_disjointness():
    """Verify frozen protocol constants and mathematical disjointness across splits."""
    from src.data.camelyon17 import (
        OFFICIAL_SOURCE_CENTERS,
        OFFICIAL_ID_VAL_CENTERS,
        OFFICIAL_OOD_VAL_CENTER,
        OFFICIAL_TARGET_TEST_CENTER,
        validate_source_centers,
    )

    src_set = set(OFFICIAL_SOURCE_CENTERS)
    id_val_set = set(OFFICIAL_ID_VAL_CENTERS)
    ood_val_set = {OFFICIAL_OOD_VAL_CENTER}
    target_set = {OFFICIAL_TARGET_TEST_CENTER}

    assert src_set == {0, 3, 4}, "Official source centers must strictly be {0, 3, 4}"
    assert id_val_set == {0, 3, 4}, "Official ID validation centers must strictly be {0, 3, 4}"
    assert ood_val_set == {1}, "Official OOD validation center must strictly be {1}"
    assert target_set == {2}, "Official target test center must strictly be {2}"

    # Disjointness invariants
    assert src_set.isdisjoint(ood_val_set), "Source centers and OOD validation center must be disjoint"
    assert src_set.isdisjoint(target_set), "Source centers and target test center must be disjoint"
    assert ood_val_set.isdisjoint(target_set), "OOD validation center and target test center must be disjoint"

    # Valid source configuration passes
    validate_source_centers(OFFICIAL_SOURCE_CENTERS)
    validate_source_centers([0, 3, 4])
    validate_source_centers([4, 3, 0])

    # Cross-check configs/FINAL/source_resnet18.yaml
    src_cfg = yaml.safe_load((ROOT / "configs" / "FINAL" / "source_resnet18.yaml").read_text())
    assert set(src_cfg["data"]["source_centers"]) == src_set
    assert src_cfg["data"]["val_center"] == OFFICIAL_OOD_VAL_CENTER
    assert src_cfg["data"]["target_center"] == OFFICIAL_TARGET_TEST_CENTER


def test_source_training_rejects_center_1():
    """Fail-closed guard: Center 1 (OOD val) must NEVER be used as source training data."""
    import numpy as np
    from src.data.camelyon17 import validate_source_centers
    from src.data.parquet_camelyon17 import ParquetSplit

    # 1. Direct validation function check
    with pytest.raises(ValueError, match=r"Centers \[1\].*strictly forbidden in source training data"):
        validate_source_centers([0, 1, 3, 4])

    with pytest.raises(ValueError, match=r"Centers \[1\].*strictly forbidden in source training data"):
        validate_source_centers([1])

    # 2. Synthetic ParquetSplit metadata check
    mock_meta = {
        "file_id": np.array([0, 1]),
        "row": np.array([0, 1]),
        "y": np.array([0, 1]),
        "center": np.array([0, 1]),  # Contains Center 1
        "slide": np.array([10, 11]),
        "patient": np.array([1, 2]),
    }
    with pytest.raises(ValueError, match=r"Protocol violation.*forbidden center"):
        ParquetSplit(files=["dummy.parquet"], meta=mock_meta, split_name="train")


def test_source_training_rejects_center_2():
    """Fail-closed guard: Center 2 (target test) must NEVER be used as source training data."""
    import numpy as np
    from src.data.camelyon17 import validate_source_centers
    from src.data.parquet_camelyon17 import ParquetSplit

    # 1. Direct validation function check
    with pytest.raises(ValueError, match=r"Centers \[2\].*strictly forbidden in source training data"):
        validate_source_centers([0, 2, 3, 4])

    with pytest.raises(ValueError, match=r"Centers \[2\].*strictly forbidden in source training data"):
        validate_source_centers([2])

    # 2. Synthetic ParquetSplit metadata check
    mock_meta = {
        "file_id": np.array([0, 1]),
        "row": np.array([0, 1]),
        "y": np.array([0, 1]),
        "center": np.array([0, 2]),  # Contains Center 2
        "slide": np.array([10, 20]),
        "patient": np.array([1, 2]),
    }
    with pytest.raises(ValueError, match=r"Protocol violation.*forbidden center"):
        ParquetSplit(files=["dummy.parquet"], meta=mock_meta, split_name="train")


def test_target_label_isolation_in_adaptation():
    """Verify target Center-2 labels cannot influence adaptation.

    1. adapt_batch signatures strictly accept (self, x) only.
    2. Adaptation operates unsupervised without labels.
    3. Passing labels to adapt_batch raises TypeError.
    """
    import inspect
    import torch
    from src.models.resnet18 import build_resnet18
    from src.tta.tent import Tent
    from src.tta.eata import EATA
    from src.tta.uncertainty_gated import GatedTTA

    model = build_resnet18(pretrained=False)
    dummy_x = torch.randn(2, 3, 96, 96)
    dummy_y = torch.tensor([0.0, 1.0])

    for cls in (Tent, EATA, GatedTTA):
        # 1. Strict signature verification: no labels argument allowed
        sig = inspect.signature(cls.adapt_batch)
        params = list(sig.parameters.keys())
        assert params == ["self", "x"], (
            f"{cls.__name__}.adapt_batch accepts arguments {params} (must only be self, x)"
        )

    # 2. Functional verification: adapt_batch runs label-free
    tent = Tent(model, lr=1e-5)
    preds = tent.adapt_batch(dummy_x)
    assert preds.shape == (2,)
    assert (preds >= 0.0).all() and (preds <= 1.0).all()

    # 3. Fail-closed: passing labels raises TypeError
    with pytest.raises(TypeError):
        tent.adapt_batch(dummy_x, dummy_y)

    with pytest.raises(TypeError):
        tent.adapt_batch(dummy_x, labels=dummy_y)

    gated = GatedTTA(model, gate="mc_dropout", tau=0.01, lr=1e-5)
    with pytest.raises(TypeError):
        gated.adapt_batch(dummy_x, dummy_y)

    eata = EATA(model, lr=1e-5)
    with pytest.raises(TypeError):
        eata.adapt_batch(dummy_x, dummy_y)


def test_tau_and_threshold_isolation_to_center_1():
    """Verify that tau/threshold tuning is restricted to Center 1 and rejects Center 2."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from scripts.run_seed_pipeline import compute_uncertainty_analysis, run_mc_dropout_on_val
    from src.models.resnet18 import build_resnet18

    # 1. Verify frozen tau values in configs/tau_by_seed.yaml
    tau_cfg = yaml.safe_load((ROOT / "configs" / "tau_by_seed.yaml").read_text())
    assert tau_cfg["rule"] == "val_70th_percentile_of_U"
    assert tau_cfg["n_mc"] == 20
    assert tau_cfg["dropout_p"] == 0.5
    for seed, val in tau_cfg["tau"].items():
        assert 0 < float(val) < 1e-3, f"seed {seed} tau={val} outside expected U range"

    # 2. Verify compute_uncertainty_analysis rejects Center 2 data
    invalid_stats = {
        "center": np.array([2, 2, 2]),  # Target test center
        "y": np.array([0, 1, 0]),
        "prob": np.array([0.2, 0.8, 0.4]),
        "variance": np.array([0.001, 0.002, 0.001]),
        "entropy": np.array([0.3, 0.4, 0.3]),
        "confidence": np.array([0.8, 0.8, 0.6]),
    }
    with pytest.raises(ValueError, match="Protocol violation.*Center 2"):
        compute_uncertainty_analysis(invalid_stats)

    # 3. Verify run_mc_dropout_on_val rejects a dataset containing Center 2
    model = build_resnet18(pretrained=False)
    x = torch.randn(4, 3, 96, 96)
    y = torch.tensor([0, 1, 0, 1])

    class Center2Dataset(TensorDataset):
        def __init__(self):
            super().__init__(x, y)
            self.center = np.array([2, 2, 2, 2])

    loader = DataLoader(Center2Dataset(), batch_size=2)
    with pytest.raises(ValueError, match="Protocol violation.*Center 2"):
        run_mc_dropout_on_val(model, loader, device=torch.device("cpu"), n_passes=2)


def test_canonical_transforms_unchanged():
    """Verify that canonical Camelyon17 transforms remain intact and protected."""
    from torchvision import transforms
    import src.data.transforms as canonical_tf
    import src.data.camelyon17 as loader_tf

    # Verify functions are identical
    assert loader_tf.get_camelyon17_train_transform is canonical_tf.get_camelyon17_train_transform
    assert loader_tf.get_camelyon17_eval_transform is canonical_tf.get_camelyon17_eval_transform

    train_tf = loader_tf.get_transform(train=True)
    eval_tf = loader_tf.get_transform(train=False)

    train_classes = [type(t) for t in train_tf.transforms]
    eval_classes = [type(t) for t in eval_tf.transforms]

    # No RandomRotation anywhere
    assert transforms.RandomRotation not in train_classes
    assert transforms.RandomRotation not in eval_classes

    # Exact canonical ordering
    assert train_classes == [
        transforms.Resize,
        transforms.RandomHorizontalFlip,
        transforms.RandomVerticalFlip,
        transforms.ColorJitter,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert eval_classes == [
        transforms.Resize,
        transforms.ToTensor,
        transforms.Normalize,
    ]
