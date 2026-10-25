"""Cellularity-Mitosis Ratio (CMR) Gating for Histopathology Detection.

Implements biophysical optical density deconvolution (Beer-Lambert law) to compute
nuclear cellularity fraction rho_cell(x) and Cellularity-Mitosis Ratio (CMR).
Decouples physiological absence of mitoses in acellular stroma from detector collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CMRGateConfig:
    """Configuration for Cellularity-Mitosis Ratio gating."""
    tau_cellularity: float = 0.15      # Minimum nuclear area fraction to qualify as cellular tissue
    tau_hematoxylin: float = 0.18     # Optical density threshold for hematoxylin nuclear stain
    epsilon: float = 1e-6              # Numerical stability epsilon for Beer-Lambert log10
    min_patch_cellularity: float = 0.05 # Rejection threshold for glass/background


class CellularityMitosisRatioGate(nn.Module):
    """Computes nuclear cellularity via Optical Density deconvolution and gates detection."""

    def __init__(self, config: Optional[CMRGateConfig] = None) -> None:
        super().__init__()
        self.config = config or CMRGateConfig()

        # Ruifrok & Johnston (2001) normalized Hematoxylin optical density vector
        # [R, G, B] absorption coefficients for Hematoxylin
        h_vec = torch.tensor([0.650, 0.704, 0.286], dtype=torch.float32)
        h_vec = F.normalize(h_vec, p=2, dim=-1)
        self.register_buffer("h_vector", h_vec)

    def rgb_to_od(self, rgb: torch.Tensor) -> torch.Tensor:
        """Convert normalized RGB tensor [0, 1] to Optical Density absorbance space.
        
        Args:
            rgb: Tensor of shape (B, 3, H, W) or (3, H, W) with values in [0, 1].
        Returns:
            OD tensor of same shape.
        """
        clamped = torch.clamp(rgb, min=self.config.epsilon, max=1.0)
        return -torch.log10(clamped)

    def compute_nuclear_optical_density(self, rgb: torch.Tensor) -> torch.Tensor:
        """Deconvolve hematoxylin optical density map.
        
        Args:
            rgb: Tensor of shape (B, 3, H, W) or (3, H, W).
        Returns:
            Hematoxylin OD map: (B, H, W) or (H, W).
        """
        is_batched = (rgb.dim() == 4)
        if not is_batched:
            rgb = rgb.unsqueeze(0)

        od = self.rgb_to_od(rgb)  # (B, 3, H, W)
        # Project OD onto unit Hematoxylin vector: sum over channel dim
        h_vec = self.h_vector.view(1, 3, 1, 1).to(rgb.device)
        h_od = torch.sum(od * h_vec, dim=1)  # (B, H, W)

        if not is_batched:
            return h_od.squeeze(0)
        return h_od

    def compute_cellularity_fraction(self, rgb: torch.Tensor) -> torch.Tensor:
        """Compute nuclear cellular area fraction rho_cell(x) in [0.0, 1.0].
        
        Args:
            rgb: Tensor of shape (B, 3, H, W) or (3, H, W).
        Returns:
            rho_cell: Tensor of shape (B,) or scalar tensor.
        """
        h_od = self.compute_nuclear_optical_density(rgb)  # (B, H, W) or (H, W)
        is_batched = (h_od.dim() == 3)
        if not is_batched:
            h_od = h_od.unsqueeze(0)

        # Nuclear mask: pixels with Hematoxylin optical density above threshold
        nuclear_mask = (h_od >= self.config.tau_hematoxylin).float()
        # Area fraction
        rho_cell = torch.mean(nuclear_mask, dim=(-2, -1))  # (B,)

        if not is_batched:
            return rho_cell.squeeze(0)
        return rho_cell

    def compute_cmr(
        self,
        n_mitoses: torch.Tensor | int,
        cellularity: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Compute Cellularity-Mitosis Ratio (CMR).
        
        CMR = n_mitoses / max(rho_cell, epsilon)
        """
        eps = self.config.epsilon
        if isinstance(n_mitoses, int) and isinstance(cellularity, (int, float)):
            return float(n_mitoses) / max(float(cellularity), eps)
        
        if not isinstance(n_mitoses, torch.Tensor):
            n_mitoses = torch.tensor(n_mitoses, dtype=torch.float32)
        if not isinstance(cellularity, torch.Tensor):
            cellularity = torch.tensor(cellularity, dtype=torch.float32)

        return n_mitoses.float() / torch.clamp(cellularity.float(), min=eps)

    def is_cellular(self, rgb: torch.Tensor) -> torch.Tensor:
        """Determine whether tissue patch meets minimum cellularity threshold.
        
        Returns boolean tensor indicating rho_cell >= tau_cellularity.
        """
        rho = self.compute_cellularity_fraction(rgb)
        return rho >= self.config.tau_cellularity

    def should_advance_silence(
        self,
        n_pred: int,
        n_source: Optional[int],
        rho_cell: float,
    ) -> Tuple[bool, str]:
        """Determine if silence counter should advance.
        
        Conditioning criteria:
        Advances iff:
        (i) n_pred == 0 (adapted detector saw nothing),
        (ii) rho_cell >= tau_cellularity (tissue is confirmed hypercellular tumor),
        (iii) n_source > 0 (frozen source detector saw candidate lesion).
        """
        if n_pred > 0:
            return False, "DETECTION_PRESENT"

        if rho_cell < self.config.tau_cellularity:
            return False, "BENIGN_ACELLULAR_STROMA_SKIPPED"

        if n_source is not None and n_source == 0:
            return False, "CONCORDANT_NEGATIVE_INVARIANCE"

        return True, "TRUE_SILENCE_ALARM_ADVANCED"
