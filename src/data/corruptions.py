"""ImageNet-C / MedMNIST-C style corruptions for robustness (Figure 9).

Applied at test time only. Never used for τ, training, or adaptation hyperparams.
Works on uint8 HWC arrays or float tensors in [0, 1] of shape [B, 3, H, W].
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


SEVERITIES = (1, 2, 3, 4, 5)

# Subset used for the optional Figure-9 sweep (cheap, no extra deps).
DEFAULT_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "brightness", "contrast", "jpeg")


def _to_uint8_hwc(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 3 and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
    return a


def _from_uint8_hwc(a: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    t = torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1)
    return t.to(device=like.device, dtype=like.dtype)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    sigma = [0.04, 0.06, 0.08, 0.12, 0.18][severity - 1]
    noise = np.random.normal(0, sigma, img.shape) * 255
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    radius = [0.5, 1.0, 1.5, 2.5, 4.0][severity - 1]
    return np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=radius)))


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [1.1, 1.2, 1.4, 1.6, 1.9][severity - 1]
    return np.array(ImageEnhance.Brightness(Image.fromarray(img)).enhance(factor))


def contrast(img: np.ndarray, severity: int) -> np.ndarray:
    factor = [1.2, 1.5, 1.8, 2.2, 2.8][severity - 1]
    return np.array(ImageEnhance.Contrast(Image.fromarray(img)).enhance(factor))


def jpeg(img: np.ndarray, severity: int) -> np.ndarray:
    quality = [70, 50, 35, 20, 10][severity - 1]
    import io

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


FN: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "defocus_blur": defocus_blur,
    "brightness": brightness,
    "contrast": contrast,
    "jpeg": jpeg,
}


def corrupt_tensor(x: torch.Tensor, name: str, severity: int) -> torch.Tensor:
    """x: [B, 3, H, W] in [0, 1] or [0, 255]. Returns same shape/device."""
    fn = FN[name]
    outs = []
    for i in range(x.size(0)):
        arr = _to_uint8_hwc(x[i].detach().cpu().numpy())
        outs.append(_from_uint8_hwc(fn(arr, severity), x[i]))
    return torch.stack(outs, dim=0)


def available_corruptions() -> List[str]:
    return list(FN.keys())
