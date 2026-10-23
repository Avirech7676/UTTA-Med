"""SafeAdapt v2 Gating Package.

Provides modular, faithful, and compute-efficient test-time adaptation gating:
- FrozenSourcePolicy (Policy A)
- ConfidenceGatingPolicy (Policy B)
- MCOnlyGatingPolicy (Policy C)
- HybridGatingPolicy (Policy D)
- AdaptiveUncertaintyGating (UTTA-Med 2.0 Adaptive Inference)
- MultiSignalGatingPolicy (UTTA-Med 2.0 Multi-Signal Gating)
- GatedSafeAdaptAdapter
- GatingTelemetryTracker
"""

from src.v2.gating.adapter import GatedSafeAdaptAdapter
from src.v2.gating.adaptive_uncertainty import AdaptiveGatingTelemetry, AdaptiveUncertaintyGating
from src.v2.gating.multi_signal import MultiSignalDecision, MultiSignalGatingPolicy
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
    "AdaptiveUncertaintyGating",
    "AdaptiveGatingTelemetry",
    "MultiSignalGatingPolicy",
    "MultiSignalDecision",
    "GatingDecision",
    "GatedSafeAdaptAdapter",
    "GatingTelemetryTracker",
    "BatchGatingRecord",
]
