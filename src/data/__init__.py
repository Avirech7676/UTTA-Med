"""Camelyon17 / path helpers. Keep this package import-light."""

__all__ = [
    "get_camelyon17_root",
    "load_camelyon17",
    "create_camelyon17_datasets",
]

def __getattr__(name: str):
    if name in ("get_camelyon17_root",):
        from src.data.paths import get_camelyon17_root
        return get_camelyon17_root
    if name in ("load_camelyon17", "create_camelyon17_datasets"):
        from src.data.camelyon17 import load_camelyon17, create_camelyon17_datasets
        return load_camelyon17 if name == "load_camelyon17" else create_camelyon17_datasets
    raise AttributeError(name)
