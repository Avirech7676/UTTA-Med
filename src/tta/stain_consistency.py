"""Beer-Lambert Stain Consistency Augmentation for Histopathology TTA.

Implements stochastic optical density (OD) stain perturbation in Beer-Lambert
absorbance space, enforcing model invariance against scanner chromatic aberration
and stain protocol drift without destroying morphological tissue geometry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeerLambertStainAugmenter(nn.Module):
    """Applies stochastic stain concentration jitter in Optical Density space."""

    def __init__(
        self,
        stain_jitter_factor: float = 0.15,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.stain_jitter_factor = stain_jitter_factor
        self.eps = eps

        # Standard normalized optical density stain vectors (Ruifrok & Johnston 2001)
        # Hematoxylin (H) and Eosin (E)
        he_matrix = torch.tensor(
            [
                [0.650, 0.704, 0.286],  # Hematoxylin
                [0.072, 0.990, 0.105],  # Eosin
            ],
            dtype=torch.float32,
        )
        # Normalize rows to unit vectors
        he_matrix = F.normalize(he_matrix, p=2, dim=-1)
        self.register_buffer("he_matrix", he_matrix)

    def rgb_to_od(self, rgb: torch.Tensor) -> torch.Tensor:
        """Converts normalized RGB tensor [0, 1] to Optical Density space."""
        # rgb shape: (B, 3, H, W)
        rgb_clamped = torch.clamp(rgb, min=self.eps, max=1.0)
        od = -torch.log10(rgb_clamped)
        return od

    def od_to_rgb(self, od: torch.Tensor) -> torch.Tensor:
        """Converts Optical Density back to normalized RGB tensor [0, 1]."""
        rgb = 10.0 ** (-od)
        return torch.clamp(rgb, min=0.0, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies stochastic stain scaling to batch x: (B, 3, H, W)."""
        if not self.training and not x.requires_grad:
            # Deterministic or evaluation pass jitter
            pass

        B, C, H, W = x.shape
        od = self.rgb_to_od(x)

        # Generate stochastic multiplier alpha in [1 - delta, 1 + delta] per sample
        device = x.device
        delta = self.stain_jitter_factor
        alpha = (
            torch.empty(B, 3, 1, 1, device=device).uniform_(1.0 - delta, 1.0 + delta)
        )

        # Apply stain scaling in OD space
        od_perturbed = od * alpha
        rgb_perturbed = self.od_to_rgb(od_perturbed)

        return rgb_perturbed
