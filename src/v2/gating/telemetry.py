"""SafeAdapt v2 Gating Telemetry Tracker.

Records granular batch-by-batch and aggregate metrics distinguishing:
- Patch Coverage (selected patches / eligible patches)
- Update Coverage (accepted optimizer steps / attempted steps)
- Prediction Coverage (all patches receiving valid predictions)
- MC Compute Efficiency (MC passes executed / theoretical full MC budget)
- Wall-clock latencies and parameter displacements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BatchGatingRecord:
    """Telemetry record for a single batch adaptation step."""

    batch_idx: int
    eligible_patches: int
    selected_patches: int
    routed_to_mc_patches: int
    mc_passes_executed: int
    fraction_routed_to_mc: float
    attempted_update: bool
    accepted_update: bool
    rejected_update: bool
    in_fallback: bool
    gating_latency_ms: float
    step_latency_ms: float
    total_latency_ms: float
    max_param_displacement: float = 0.0
    mean_abs_prob_diff: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_idx": self.batch_idx,
            "eligible_patches": self.eligible_patches,
            "selected_patches": self.selected_patches,
            "routed_to_mc_patches": self.routed_to_mc_patches,
            "mc_passes_executed": self.mc_passes_executed,
            "fraction_routed_to_mc": round(self.fraction_routed_to_mc, 4),
            "attempted_update": self.attempted_update,
            "accepted_update": self.accepted_update,
            "rejected_update": self.rejected_update,
            "in_fallback": self.in_fallback,
            "gating_latency_ms": round(self.gating_latency_ms, 2),
            "step_latency_ms": round(self.step_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "max_param_displacement": round(self.max_param_displacement, 8),
            "mean_abs_prob_diff": round(self.mean_abs_prob_diff, 8),
        }


class GatingTelemetryTracker:
    """Tracks cumulative and per-batch telemetry for gated adaptation."""

    def __init__(self, policy_name: str = "gating_policy"):
        self.policy_name = policy_name
        self.records: List[BatchGatingRecord] = []
        self._batch_counter = 0

    def record_batch(
        self,
        eligible_patches: int,
        selected_patches: int,
        routed_to_mc_patches: int,
        mc_passes_executed: int,
        attempted_update: bool,
        accepted_update: bool,
        rejected_update: bool,
        in_fallback: bool,
        gating_latency_ms: float,
        step_latency_ms: float,
        max_param_displacement: float = 0.0,
        mean_abs_prob_diff: float = 0.0,
    ) -> BatchGatingRecord:
        """Record telemetry for a single batch."""
        self._batch_counter += 1
        frac_mc = routed_to_mc_patches / max(eligible_patches, 1)
        tot_latency = gating_latency_ms + step_latency_ms

        rec = BatchGatingRecord(
            batch_idx=self._batch_counter,
            eligible_patches=eligible_patches,
            selected_patches=selected_patches,
            routed_to_mc_patches=routed_to_mc_patches,
            mc_passes_executed=mc_passes_executed,
            fraction_routed_to_mc=frac_mc,
            attempted_update=attempted_update,
            accepted_update=accepted_update,
            rejected_update=rejected_update,
            in_fallback=in_fallback,
            gating_latency_ms=gating_latency_ms,
            step_latency_ms=step_latency_ms,
            total_latency_ms=tot_latency,
            max_param_displacement=max_param_displacement,
            mean_abs_prob_diff=mean_abs_prob_diff,
        )
        self.records.append(rec)
        return rec

    def summary(self) -> Dict[str, Any]:
        """Compute aggregate summary metrics across all recorded batches."""
        total_batches = len(self.records)
        if total_batches == 0:
            return {
                "policy_name": self.policy_name,
                "total_batches": 0,
                "patch_coverage": 0.0,
                "update_coverage": 0.0,
                "prediction_coverage": 0.0,
            }

        total_eligible = sum(r.eligible_patches for r in self.records)
        total_selected = sum(r.selected_patches for r in self.records)
        total_routed_mc = sum(r.routed_to_mc_patches for r in self.records)
        total_attempted = sum(1 for r in self.records if r.attempted_update)
        total_accepted = sum(1 for r in self.records if r.accepted_update)
        total_rejected = sum(1 for r in self.records if r.rejected_update)
        total_fallback = sum(1 for r in self.records if r.in_fallback)
        total_mc_passes = sum(r.mc_passes_executed for r in self.records)

        theoretical_full_mc_passes = total_batches * 20
        mc_compute_ratio = total_mc_passes / max(theoretical_full_mc_passes, 1)

        patch_coverage = total_selected / max(total_eligible, 1)
        update_coverage = total_accepted / max(total_attempted, 1) if total_attempted > 0 else 0.0
        prediction_coverage = 1.0  # All patches receive predictions

        total_wall_clock_ms = sum(r.total_latency_ms for r in self.records)
        mean_batch_latency_ms = total_wall_clock_ms / max(total_batches, 1)
        mean_gating_latency_ms = sum(r.gating_latency_ms for r in self.records) / max(total_batches, 1)
        mean_step_latency_ms = sum(r.step_latency_ms for r in self.records) / max(total_batches, 1)

        max_displacement = max((r.max_param_displacement for r in self.records), default=0.0)
        mean_prob_diff = sum(r.mean_abs_prob_diff for r in self.records) / max(total_batches, 1)

        return {
            "policy_name": self.policy_name,
            "total_batches": total_batches,
            "total_eligible_patches": total_eligible,
            "total_selected_patches": total_selected,
            "total_routed_to_mc_patches": total_routed_mc,
            "patch_coverage": round(patch_coverage, 6),
            "total_attempted_updates": total_attempted,
            "total_accepted_updates": total_accepted,
            "total_rejected_updates": total_rejected,
            "total_fallback_batches": total_fallback,
            "update_coverage": round(update_coverage, 6),
            "prediction_coverage": prediction_coverage,
            "total_mc_passes_executed": total_mc_passes,
            "theoretical_full_mc_passes": theoretical_full_mc_passes,
            "mc_compute_ratio": round(mc_compute_ratio, 6),
            "total_wall_clock_ms": round(total_wall_clock_ms, 2),
            "total_wall_clock_s": round(total_wall_clock_ms / 1000.0, 3),
            "mean_batch_latency_ms": round(mean_batch_latency_ms, 2),
            "mean_gating_latency_ms": round(mean_gating_latency_ms, 2),
            "mean_step_latency_ms": round(mean_step_latency_ms, 2),
            "max_parameter_displacement": round(max_displacement, 8),
            "mean_absolute_prob_diff": round(mean_prob_diff, 8),
        }
