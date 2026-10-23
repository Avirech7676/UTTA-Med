"""Tests for path resolution, environment variable handling, and fail-closed behavior."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_missing_camelyon17_root_fails_clearly(monkeypatch):
    """When CAMELYON17_ROOT is unset and no explicit path is given, fail clearly."""
    monkeypatch.delenv("CAMELYON17_ROOT", raising=False)

    from src.data.paths import resolve_parquet_root, find_version_dir
    from src.data.camelyon17 import load_camelyon17

    # 1. resolve_parquet_root() must fail closed with informative RuntimeError
    with pytest.raises(RuntimeError, match="CAMELYON17_ROOT is not set"):
        resolve_parquet_root()

    # 2. load_camelyon17() must fail closed with informative RuntimeError
    with pytest.raises(RuntimeError, match="CAMELYON17_ROOT is not set"):
        load_camelyon17(data_root=None)

    # 3. find_version_dir() must fail closed
    with pytest.raises(RuntimeError, match="CAMELYON17_ROOT is not set"):
        find_version_dir(data_root=None)


def test_no_personal_fallback_used(monkeypatch):
    """Ensure no hardcoded developer paths exist in paths module."""
    monkeypatch.delenv("CAMELYON17_ROOT", raising=False)
    import src.data.paths as paths_mod

    # DEFAULT_DATA_ROOT must not exist
    assert not hasattr(paths_mod, "DEFAULT_DATA_ROOT")

    # Inspection of candidate roots must fail without fallback
    with pytest.raises(RuntimeError, match="CAMELYON17_ROOT is not set"):
        paths_mod._candidate_roots(None)


def test_resolve_parquet_root_valid(tmp_path, monkeypatch):
    """resolve_parquet_root() correctly resolves a valid Parquet directory."""
    data_dir = tmp_path / "mock_camelyon17" / "data"
    data_dir.mkdir(parents=True)
    dummy_parquet = data_dir / "train-00000-of-00014.parquet"
    dummy_parquet.write_text("mock parquet content")

    from src.data.paths import resolve_parquet_root

    # Test explicit root (parent folder containing data/)
    resolved = resolve_parquet_root(tmp_path / "mock_camelyon17")
    assert resolved == data_dir.resolve()

    # Test explicit direct data directory
    resolved_direct = resolve_parquet_root(data_dir)
    assert resolved_direct == data_dir.resolve()

    # Test via CAMELYON17_ROOT
    monkeypatch.setenv("CAMELYON17_ROOT", str(tmp_path / "mock_camelyon17"))
    resolved_env = resolve_parquet_root()
    assert resolved_env == data_dir.resolve()


def test_resolve_parquet_root_fails_closed_invalid(tmp_path, monkeypatch):
    """resolve_parquet_root() fails closed with RuntimeError when Parquet files missing."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    from src.data.paths import resolve_parquet_root

    # Explicit invalid path with no parquet
    with pytest.raises(RuntimeError, match="Unable to locate Camelyon17 Parquet data"):
        resolve_parquet_root(empty_dir)

    # Non-existent path
    with pytest.raises(RuntimeError, match="Unable to locate Camelyon17 Parquet data"):
        resolve_parquet_root(tmp_path / "non_existent_path")

    # Invalid path set in CAMELYON17_ROOT
    monkeypatch.setenv("CAMELYON17_ROOT", str(empty_dir))
    with pytest.raises(RuntimeError, match="Unable to locate Camelyon17 Parquet data"):
        resolve_parquet_root()


def test_camelyon17_root_respected(tmp_path, monkeypatch):
    """Verify CAMELYON17_ROOT environment variable is respected."""
    custom_root = tmp_path / "custom_data"
    custom_root.mkdir()
    (custom_root / "train-00000.parquet").write_text("mock")

    monkeypatch.setenv("CAMELYON17_ROOT", str(custom_root))

    from src.data.paths import resolve_parquet_root

    resolved = resolve_parquet_root()
    assert resolved == custom_root.resolve()
