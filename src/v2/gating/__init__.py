"""SafeAdapt v2 Gating Package.

Provides modular, faithful, and compute-efficient test-time adaptation gating:
- FrozenSourcePolicy (Policy A)
- ConfidenceGatingPolicy (Policy B)
- MCOnlyGatingPolicy (Policy C)
- HybridGatingPolicy (Policy D)
- GatedSafeAdaptAdapter
- GatingTelemetryTracker
"""

from src.v2.gating.adapter import GatedSafeAdaptAdapter
from src.v2.gating.cmr_gate import CellularityMitosisRatioGate, CMRGateConfig
from src.v2.gating.policies import (
    BaseGatingPolicy,
    ConfidenceGatingPolicy,
    FrozenSourcePolicy,
    GatingDecision,
    HybridGatingPolicy,
    MCOnlyGatingPolicy,
)
from src.v2.gating.telemetry import BatchGatingRecord, GatingTelemetryTracker

__all__ = [
    "BaseGatingPolicy",
    "FrozenSourcePolicy",
    "ConfidenceGatingPolicy",
    "MCOnlyGatingPolicy",
    "HybridGatingPolicy",
    "GatingDecision",
    "GatedSafeAdaptAdapter",
    "GatingTelemetryTracker",
    "BatchGatingRecord",
    "CellularityMitosisRatioGate",
    "CMRGateConfig",
]

