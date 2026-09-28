"""Central reproducibility and seeding utilities for UTTA-Med.

Ensures deterministic random states across PyTorch, NumPy, Python standard random,
and CUDA (when available) across all benchmark methods (Source, Tent, EATA, UTTA-Med).
"""

from __future__ import annotations

import os
import random
from typing import List, Optional

import numpy as np
import torch

# Mandatory frozen 5-seed matrix from Master Plan v4 (§0.6)
SEEDS: List[int] = [42, 123, 2024, 7, 99]
EXTENDED_SEEDS: List[int] = [11, 22, 33, 77, 888]


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def seed_worker(worker_id: int) -> None:
    """Worker init function for PyTorch DataLoader workers to guarantee reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
