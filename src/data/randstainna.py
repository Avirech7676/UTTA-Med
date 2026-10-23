"""RandStainNA: Random Stain Normalization and Augmentation for Histopathology.

Implements optical density (OD) decomposition and stain matrix perturbation
for cross-scanner and cross-hospital stain invariance in digital pathology.

Mathematical Formulation:
1. RGB to Optical Density (OD):
   OD = -log10((I + 1) / 256)
2. Stain Matrix Decomposition:
   OD = V @ C, where V in R^{3x2} (Hematoxylin & Eosin vectors) and C in R^{2xN}.
3. Concentration & Color Perturbation:
   C_aug = C * diag(alpha_H, alpha_E), alpha ~ U(1 - delta_c, 1 + delta_c)
   V_aug = V + epsilon, epsilon ~ N(0, sigma_v^2)
4. Reconstructed RGB:
   I_aug = 255 * 10^(-V_aug @ C_aug), clipped to [0, 255].
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# Reference canonical stain vectors (Macenko et al.) for H&E
DEFAULT_HE_STAIN_MATRIX = np.array([
    [0.650, 0.070],  # Red channel for [Hematoxylin, Eosin]
    [0.704, 0.990],  # Green channel
    [0.286, 0.105],  # Blue channel
], dtype=np.float32)


class RandStainNA:
    """Random Stain Normalization and Augmentation transform for PIL images and tensors."""

    def __init__(
        self,
        p: float = 0.5,
        delta_concentration: float = 0.20,
        delta_stain: float = 0.05,
        stain_matrix: Optional[np.ndarray] = None,
    ):
        """
        Args:
            p: Probability of applying stain augmentation.
            delta_concentration: Range for concentration scaling [1 - delta, 1 + delta].
            delta_stain: Standard deviation of additive Gaussian noise to stain vectors.
            stain_matrix: 3x2 reference stain matrix. Defaults to canonical Macenko H&E.
        """
        self.p = p
        self.delta_c = delta_concentration
        self.delta_s = delta_stain
        self.stain_matrix = stain_matrix if stain_matrix is not None else DEFAULT_HE_STAIN_MATRIX

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Union[Image.Image, np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p:
            return img

        is_pil = isinstance(img, Image.Image)
        is_torch = isinstance(img, torch.Tensor)

        if is_pil:
            arr = np.array(img, dtype=np.float32)
        elif is_torch:
            arr = img.permute(1, 2, 0).cpu().numpy().astype(np.float32) if img.dim() == 3 else img.numpy()
        else:
            arr = np.array(img, dtype=np.float32)

        # Convert RGB to Optical Density (OD)
        arr_clipped = np.clip(arr, 1.0, 255.0)
        od = -np.log10(arr_clipped / 256.0)  # Shape (H, W, 3)
        h, w, c = od.shape
        od_flat = od.reshape(-1, 3).T  # Shape (3, N)

        # Estimate concentrations via pseudo-inverse: C = (V^T V)^(-1) V^T OD
        v = self.stain_matrix.copy()
        try:
            v_pinv = np.linalg.pinv(v)
            concentrations = np.dot(v_pinv, od_flat)  # Shape (2, N)
            concentrations = np.maximum(concentrations, 0.0)
        except Exception:
            return img

        # Random concentration scaling (alpha_H, alpha_E)
        alpha = np.random.uniform(1.0 - self.delta_c, 1.0 + self.delta_c, size=(2, 1)).astype(np.float32)
        c_aug = concentrations * alpha

        # Random stain matrix vector perturbation
        v_noise = np.random.normal(0.0, self.delta_s, size=v.shape).astype(np.float32)
        v_aug = np.maximum(v + v_noise, 0.01)
        # Normalize columns to unit length
        v_aug = v_aug / (np.linalg.norm(v_aug, axis=0, keepdims=True) + 1e-8)

        # Reconstruct Optical Density and RGB
        od_aug = np.dot(v_aug, c_aug).T.reshape(h, w, c)
        rgb_aug = 256.0 * np.power(10.0, -od_aug) - 1.0
        rgb_aug = np.clip(rgb_aug, 0.0, 255.0).astype(np.uint8)

        if is_pil:
            return Image.fromarray(rgb_aug)
        elif is_torch:
            t = torch.from_numpy(rgb_aug).permute(2, 0, 1).float() / 255.0
            return t.to(img.device) if hasattr(img, "device") else t
        return rgb_aug
