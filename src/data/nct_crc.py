"""NCT-CRC-HE Multi-Class Histopathology Dataset Loader.

Supports NCT-CRC-HE-100K (source training & in-domain validation)
and CRC-VAL-HE-7K (out-of-domain external validation).

Tissue classes (K=9):
0: ADI   (Adipose tissue)
1: BACK  (Background)
2: DEB   (Debris)
3: LYM   (Lymphocytes)
4: MUC   (Mucus)
5: MUS   (Smooth muscle)
6: NORM  (Normal colon mucosa)
7: STR   (Cancer-associated stroma)
8: TUM   (Colorectal adenocarcinoma epithelium)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NCT_CLASSES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM",
]
NUM_CLASSES = len(NCT_CLASSES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(NCT_CLASSES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(NCT_CLASSES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_nct_transforms(train: bool = False, image_size: int = 224) -> transforms.Compose:
    """Standard transformations for NCT-CRC-HE."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class MockNCTCRCDataset(Dataset):
    """Deterministic synthetic dataset for offline CPU testing & CI."""

    def __init__(
        self,
        num_samples: int = 180,
        num_classes: int = 9,
        image_size: int = 224,
        seed: int = 42,
        shift_severity: float = 0.0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.shift_severity = shift_severity

        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)
        # Class distribution balance check
        self.samples = []
        gen = torch.Generator().manual_seed(seed)
        self.tensors = torch.randn(num_samples, 3, image_size, image_size, generator=gen) * 0.1
        for i in range(num_samples):
            lbl = int(self.labels[i])
            self.tensors[i, lbl % 3, :, :] += float(lbl) * 0.2
            if self.shift_severity > 0:
                self.tensors[i] += float(self.shift_severity) * 0.5
            self.samples.append((f"mock_tile_{i}.png", lbl))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tensors[idx], int(self.labels[idx])


class NCTCRCDataset(Dataset):
    """Filesystem dataset for NCT-CRC-HE image directory."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform or get_nct_transforms(train=False)

        self.samples: List[Tuple[Path, int]] = []
        if self.root_dir.exists():
            for cls_name, cls_idx in CLASS_TO_IDX.items():
                cls_dir = self.root_dir / cls_name
                if cls_dir.is_dir():
                    for img_p in cls_dir.glob("*.tif*"):
                        self.samples.append((img_p, cls_idx))
                    for img_p in cls_dir.glob("*.png*"):
                        self.samples.append((img_p, cls_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        tensor_img = self.transform(img)
        return tensor_img, label


def build_nct_dataloader(
    dataset_path: Optional[str] = None,
    batch_size: int = 64,
    shuffle: bool = False,
    is_target_eval: bool = False,
    mock_samples: int = 180,
    seed: int = 42,
) -> DataLoader:
    """Construct DataLoader for NCT-CRC-HE (real or mock fallback)."""
    if dataset_path and os.path.exists(dataset_path):
        ds = NCTCRCDataset(dataset_path, transform=get_nct_transforms(train=not is_target_eval))
    else:
        shift = 0.5 if is_target_eval else 0.0
        ds = MockNCTCRCDataset(
            num_samples=mock_samples,
            seed=seed,
            shift_severity=shift,
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
