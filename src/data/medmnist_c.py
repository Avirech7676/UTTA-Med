"""PathMNIST / MedMNIST-C optional robustness loader.

Primary paper remains Camelyon17-WILDS. This module is **optional Figure 9**.
If `medmnist` is not installed, `load_pathmnist` raises a clear error.

Corruptions use `src.data.corruptions` (MedMNIST-C analogue) so we do not
depend on the unpublished MedMNIST-C wheel.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class PathMNISTWrap(Dataset):
    """RGB 28×28 PathMNIST, resized to 96×96 to reuse ResNet stems."""

    def __init__(self, split: str = "test", resize: int = 96):
        try:
            import medmnist
        except ImportError as e:
            raise ImportError(
                "PathMNIST requires `pip install medmnist`. "
                "This dataset is optional (Master Plan §53)."
            ) from e
        self.ds = medmnist.PathMNIST(split=split, download=True)
        self.resize = resize
        # PathMNIST is 9-class. Binary proxy: tumor-associated vs rest is not
        # defined by WILDS; we keep multiclass labels for a separate head.
        self.n_classes = 9

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int]:
        from PIL import Image

        img, y = self.ds[i]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        img = img.resize((self.resize, self.resize))
        arr = np.asarray(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        label = int(np.asarray(y).reshape(-1)[0])
        return x, label
