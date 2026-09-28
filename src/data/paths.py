"""Resolve the local Camelyon17-WILDS folder on the research machine.

Never hard-code hospital IDs. This module only finds files on disk.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

DEFAULT_DATA_ROOT = r"C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"
VERSION_DIR_NAME = "camelyon17_v1.0"


def find_parquet_data_dir(root: Path) -> Optional[Path]:
    """Return the folder that contains train-*.parquet, or None."""
    root = Path(root)
    candidates = [root / "data", root]
    if root.is_dir():
        try:
            for child in root.iterdir():
                if child.is_dir():
                    candidates.append(child)
                    candidates.append(child / "data")
        except OSError:
            pass
    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_dir() and any(cand.glob("train-*.parquet")):
                return cand.resolve()
        except OSError:
            continue
    return None



def _candidate_roots(explicit: Optional[str]) -> list[Path]:
    roots: list[Path] = []
    if explicit:
        roots.append(Path(explicit).expanduser())
    env = os.environ.get("CAMELYON17_ROOT")
    if env:
        roots.append(Path(env).expanduser())
    roots.append(Path(DEFAULT_DATA_ROOT))
    extra: list[Path] = []
    for r in list(roots):
        extra.extend(
            [
                r / VERSION_DIR_NAME,
                r / "camelyon17",
                r / "data",
                r / "data" / VERSION_DIR_NAME,
                r / "WILDS",
                r / "wilds",
            ]
        )
    roots.extend(extra)
    seen = set()
    out = []
    for p in roots:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _looks_like_version_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "metadata.csv").is_file()


def _search_metadata(start: Path, max_depth: int = 4) -> Optional[Path]:
    """Find metadata.csv under start, limited depth (WILDS version folder)."""
    if not start.is_dir():
        return None
    if (start / "metadata.csv").is_file():
        return start
    start = start.resolve()
    for dirpath, dirnames, filenames in os.walk(start):
        rel = Path(dirpath).relative_to(start)
        depth = 0 if str(rel) == "." else len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        if "metadata.csv" in filenames:
            return Path(dirpath)
    return None


def _list_top(path: Path, n: int = 20) -> str:
    if not path.exists():
        return f"  (path does not exist: {path})"
    if not path.is_dir():
        return f"  (not a directory: {path})"
    try:
        items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as e:
        return f"  (cannot list: {e})"
    if not items:
        return "  (folder is empty — if this is OneDrive, right-click → Always keep on this device)"
    lines = []
    for item in items[:n]:
        kind = "<DIR> " if item.is_dir() else "      "
        lines.append(f"  {kind}{item.name}")
    if len(items) > n:
        lines.append(f"  ... {len(items) - n} more")
    return "\n".join(lines)


def find_version_dir(data_root: Optional[str] = None) -> Path:
    """Return the folder that contains metadata.csv (typically camelyon17_v1.0)."""
    tried = []
    for cand in _candidate_roots(data_root):
        tried.append(cand)
        if _looks_like_version_dir(cand):
            return cand
        nested = cand / VERSION_DIR_NAME
        tried.append(nested)
        if _looks_like_version_dir(nested):
            return nested
        found = _search_metadata(cand, max_depth=4)
        if found is not None:
            return found

    primary = Path(data_root).expanduser() if data_root else Path(DEFAULT_DATA_ROOT)
    listing = _list_top(primary)
    tried_txt = "\n".join(f"  - {p}" for p in tried[:10])
    raise FileNotFoundError(
        "Could not find Camelyon17-WILDS metadata.csv.\n\n"
        f"Looked in:\n{tried_txt}\n\n"
        f"Contents of {primary}:\n{listing}\n\n"
        "WILDS needs this layout:\n"
        f"  {primary}\\camelyon17_v1.0\\metadata.csv\n"
        f"  {primary}\\camelyon17_v1.0\\patches\\\n\n"
        "If you instead have large .tif whole-slide images, that is the original\n"
        "Camelyon17 challenge data — not the WILDS patch dataset this project uses.\n"
        "Run:  python scripts\\diagnose_data.py --data-root \"" + str(primary) + "\"\n"
        "If the folder is on OneDrive and looks empty, right-click it → Always keep on this device."
    )


def resolve_camelyon17_paths(data_root: Optional[str] = None) -> Tuple[Path, Path]:
    """Return (wilds_root_dir, version_dir)."""
    version_dir = find_version_dir(data_root)
    if version_dir.name.startswith("camelyon17_v"):
        wilds_root = version_dir.parent
    else:
        wilds_root = version_dir.parent if (version_dir / "metadata.csv").is_file() else version_dir
        # WILDS get_dataset(root_dir=X) looks for X/camelyon17_v1.0/
        if version_dir.name.startswith("camelyon17"):
            wilds_root = version_dir.parent
        else:
            wilds_root = version_dir
    return wilds_root.resolve(), version_dir.resolve()


def describe_layout(version_dir: Path) -> Dict:
    meta = version_dir / "metadata.csv"
    patches = version_dir / "patches"
    release = None
    for name in ("RELEASE_v1.0.txt", "RELEASE.txt"):
        if (version_dir / name).is_file():
            release = str(version_dir / name)
            break
    n_patch_subdirs = 0
    if patches.is_dir():
        try:
            n_patch_subdirs = sum(1 for p in patches.iterdir() if p.is_dir())
        except OSError:
            n_patch_subdirs = -1
    return {
        "version_dir": str(version_dir),
        "metadata_csv": str(meta) if meta.is_file() else None,
        "metadata_bytes": meta.stat().st_size if meta.is_file() else 0,
        "patches_dir": str(patches) if patches.is_dir() else None,
        "n_patch_subdirs": n_patch_subdirs,
        "release_file": release,
        "has_metadata": meta.is_file(),
        "has_patches": patches.is_dir(),
    }
