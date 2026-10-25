"""Unit tests for CMR Gating and SafeAdapt Detection v2."""

import pytest
import torch
import torch.nn as nn
from src.v2.gating.cmr_gate import CellularityMitosisRatioGate, CMRGateConfig
from src.v2.safeadapt.detection_v2 import SafeAdaptDetectionAdapterV2, SafeAdaptDetectionConfig


class MockDetector(nn.Module):
    """Mock detector returning proposal predictions."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return {
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.85]),
            "labels": torch.tensor([1]),
        }

    def predict(self, x):
        return self.forward(x)


def test_cmr_optical_density_calculation():
    gate = CellularityMitosisRatioGate()
    # Synthetic RGB patch (1, 3, 64, 64)
    rgb = torch.ones(1, 3, 64, 64) * 0.5
    h_od = gate.compute_nuclear_optical_density(rgb)
    assert h_od.shape == (1, 64, 64)
    assert (h_od >= 0.0).all()

    rho = gate.compute_cellularity_fraction(rgb)
    assert rho.shape == (1,)
    assert 0.0 <= rho.item() <= 1.0


def test_cmr_silence_advance_logic():
    gate = CellularityMitosisRatioGate(CMRGateConfig(tau_cellularity=0.15))

    # Case 1: Detection is present -> never advance silence
    adv, reason = gate.should_advance_silence(n_pred=2, n_source=1, rho_cell=0.25)
    assert not adv
    assert reason == "DETECTION_PRESENT"

    # Case 2: Acellular stroma (rho_cell < 0.15) and n_pred == 0 -> skip silence
    adv, reason = gate.should_advance_silence(n_pred=0, n_source=0, rho_cell=0.04)
    assert not adv
    assert reason == "BENIGN_ACELLULAR_STROMA_SKIPPED"

    # Case 3: Concordant negative (both 0) on cellular tissue -> skip silence
    adv, reason = gate.should_advance_silence(n_pred=0, n_source=0, rho_cell=0.30)
    assert not adv
    assert reason == "CONCORDANT_NEGATIVE_INVARIANCE"

    # Case 4: True silence (n_pred == 0, cellular tissue, but source detector predicted candidate) -> advance!
    adv, reason = gate.should_advance_silence(n_pred=0, n_source=2, rho_cell=0.35)
    assert adv
    assert reason == "TRUE_SILENCE_ALARM_ADVANCED"


def test_detection_v2_adaptation_and_recoil():
    detector = MockDetector()
    adapter = SafeAdaptDetectionAdapterV2(detector, config=SafeAdaptDetectionConfig(lr=1e-4))

    # Test patch (1, 3, 64, 64) with high cellularity
    patch = torch.ones(1, 3, 64, 64) * 0.2
    res = adapter.adapt_patch(patch)
    assert "adapted" in res
    assert "predictions" in res
    assert "cellularity" in res
