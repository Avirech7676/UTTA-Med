"""SafeAdapt v2 Structured Telemetry Module.

Records adaptation events, numerical rejections, parameter displacement statistics,
and fallback transitions without logging raw patient image data or identifying metadata.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TelemetryEvent:
    """Individual adaptation step event."""

    batch_idx: int
    event_type: str  # ACCEPTED, REJECTED, CLAMPED, ROLLBACK, FALLBACK, RESET
    reason: Optional[str] = None
    loss: Optional[float] = None
    grad_norm: Optional[float] = None
    max_displacement_linf: Optional[float] = None
    num_clamped_tensors: int = 0
    in_fallback: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to privacy-preserving summary dict."""
        return {
            "batch_idx": self.batch_idx,
            "event_type": self.event_type,
            "reason": self.reason,
            "loss": round(self.loss, 6) if self.loss is not None else None,
            "grad_norm": round(self.grad_norm, 6) if self.grad_norm is not None else None,
            "max_displacement_linf": round(self.max_displacement_linf, 6)
            if self.max_displacement_linf is not None
            else None,
            "num_clamped_tensors": self.num_clamped_tensors,
            "in_fallback": self.in_fallback,
            "timestamp": self.timestamp,
        }

    def to_fhir_observation(self) -> Dict[str, Any]:
        """Convert telemetry event to an HL7 FHIR R4 Observation resource."""
        return {
            "resourceType": "Observation",
            "status": "final",
            "code": {
                "coding": [
                    {
                        "system": "http://safeadapt.digitalpathology.org/codes",
                        "code": f"TTA-{self.event_type}",
                        "display": f"SafeAdapt Test-Time Adaptation {self.event_type} Event",
                    }
                ]
            },
            "effectiveDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "component": [
                {
                    "code": {"coding": [{"code": "batch-idx", "display": "Target Batch Index"}]},
                    "valueInteger": self.batch_idx,
                },
                {
                    "code": {"coding": [{"code": "fallback-status", "display": "Latching Fallback Active"}]},
                    "valueBoolean": self.in_fallback,
                },
                {
                    "code": {"coding": [{"code": "linf-displacement", "display": "Max Parameter Displacement L_inf"}]},
                    "valueQuantity": {"value": self.max_displacement_linf or 0.0, "unit": "norm"},
                },
            ],
        }

    def to_dicom_audit_event(self) -> Dict[str, Any]:
        """Convert telemetry event to a DICOM Part 18 / IHE ATNA Audit Record."""
        return {
            "EventIdentification": {
                "EventID": {"code": "110113", "displayName": "Security Alert"},
                "EventActionCode": "E",
                "EventDateTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
                "EventOutcomeIndicator": "0" if self.event_type in ("ACCEPTED", "RESET") else "4",
            },
            "ActiveParticipant": {
                "UserID": "SafeAdapt-CADx-Interlock",
                "UserIsRequestor": False,
                "RoleIDCode": {"code": "110150", "displayName": "Application"},
            },
            "AuditEnterpriseSiteID": "Clinical-Digital-Pathology-Laboratory",
            "EventDetail": {
                "EventType": self.event_type,
                "Reason": self.reason or "None",
                "InFallback": self.in_fallback,
            },
        }



class SafeAdaptTelemetry:
    """Telemetry aggregator for SafeAdapt controller."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.events: List[TelemetryEvent] = []

        # Aggregated counters
        self.batches_seen: int = 0
        self.updates_accepted: int = 0
        self.updates_rejected: int = 0
        self.rollback_events: int = 0
        self.fallback_transitions: int = 0
        self.clamping_events: int = 0
        self.rejection_reasons: Dict[str, int] = {
            "non_finite_loss": 0,
            "non_finite_gradient": 0,
            "non_finite_parameter": 0,
            "non_finite_prediction": 0,
            "exception": 0,
            "other": 0,
        }

        # Metric tracking
        self.max_observed_grad_norm: float = 0.0
        self.max_observed_linf_displacement: float = 0.0

    @property
    def fallback_triggered(self) -> bool:
        """Return True if at least one fallback transition occurred."""
        return self.fallback_transitions > 0

    def record_event(
        self,
        batch_idx: int,
        event_type: str,
        reason: Optional[str] = None,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        max_displacement_linf: Optional[float] = None,
        num_clamped_tensors: int = 0,
        in_fallback: bool = False,
    ) -> TelemetryEvent:
        """Record an event and update aggregate statistics."""
        self.batches_seen += 1

        if event_type == "ACCEPTED":
            self.updates_accepted += 1
        elif event_type in ("REJECTED", "ROLLBACK"):
            self.updates_rejected += 1
            self.rollback_events += 1
            if reason and reason.startswith("exception"):
                self.rejection_reasons["exception"] += 1
            elif reason in self.rejection_reasons:
                self.rejection_reasons[reason] += 1
            else:
                self.rejection_reasons["other"] += 1
        elif event_type == "FALLBACK":
            self.fallback_transitions += 1

        if num_clamped_tensors > 0:
            self.clamping_events += 1

        if grad_norm is not None and grad_norm > self.max_observed_grad_norm:
            self.max_observed_grad_norm = grad_norm

        if max_displacement_linf is not None and max_displacement_linf > self.max_observed_linf_displacement:
            self.max_observed_linf_displacement = max_displacement_linf

        event = TelemetryEvent(
            batch_idx=batch_idx,
            event_type=event_type,
            reason=reason,
            loss=loss,
            grad_norm=grad_norm,
            max_displacement_linf=max_displacement_linf,
            num_clamped_tensors=num_clamped_tensors,
            in_fallback=in_fallback,
        )

        if len(self.events) < self.max_history:
            self.events.append(event)

        return event

    def summary(self) -> Dict[str, Any]:
        """Return high-level summary statistics without identifying metadata."""
        return {
            "batches_seen": self.batches_seen,
            "updates_accepted": self.updates_accepted,
            "updates_rejected": self.updates_rejected,
            "rollback_events": self.rollback_events,
            "fallback_transitions": self.fallback_transitions,
            "clamping_events": self.clamping_events,
            "rejection_reasons": dict(self.rejection_reasons),
            "max_observed_grad_norm": round(self.max_observed_grad_norm, 6),
            "max_observed_linf_displacement": round(self.max_observed_linf_displacement, 6),
        }
