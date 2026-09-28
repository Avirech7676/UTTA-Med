#!/usr/bin/env python3
"""Alias for inspect_local_data.py (kept so older docs still work)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("inspect_local_data.py")), run_name="__main__")
