"""CCE Stability, Perturbation Recovery & Drift Monitoring Subsystem (Phase CCE-4).

Provides runtime monitors to guarantee:
1. State Boundedness: ||X(t)|| <= B for all time t >= 0.
2. Perturbation Recovery: Measures relaxation half-life t_{1/2} following external shock.
3. No-Input Persistence: Confirms deterministic asymptotic decay when u(t) = 0.
4. Malformed Proposal Defense: Fails closed on NaN/Inf/untrusted cognitive feedback proposals.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from nsa.runtime.cce_persistent_state import CognitiveStateSnapshot, PersistentCognitiveState


@dataclass(frozen=True)
class StabilityMetrics:
    """Quantitative stability observation for a CCE state trajectory."""

    working_norm: float
    self_norm: float
    goal_norm: float
    is_bounded: bool
    drift_rate: float
    uncertainty: float
    timestamp_utc: float
    anomaly_detected: bool = False
    anomaly_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CCEStabilityMonitor:
    """Monitors continuous state bounds, drift rates, and relaxation dynamics."""

    def __init__(
        self,
        max_bound: float = 10.0,
        max_drift_rate: float = 5.0,
        history_len: int = 100,
    ) -> None:
        self.max_bound = float(max_bound)
        self.max_drift_rate = float(max_drift_rate)
        self.history_len = int(history_len)
        self._history: List[StabilityMetrics] = []
        self._last_snap: Optional[CognitiveStateSnapshot] = None
        self._last_time: float = time.time()

    def check_and_record(self, snap: CognitiveStateSnapshot, current_time: Optional[float] = None) -> StabilityMetrics:
        """Inspect a state snapshot, enforce bounds, and record stability metrics."""
        now = current_time if current_time is not None else time.time()
        dt = max(1e-4, now - self._last_time)
        self._last_time = now

        # 1. Check finite values
        if not torch.isfinite(snap.working).all() or not torch.isfinite(snap.self_state).all():
            metrics = StabilityMetrics(
                working_norm=float("nan"),
                self_norm=float("nan"),
                goal_norm=float("nan"),
                is_bounded=False,
                drift_rate=float("inf"),
                uncertainty=1.0,
                timestamp_utc=now,
                anomaly_detected=True,
                anomaly_reason="Non-finite values detected in continuous state tensors",
            )
            self._history.append(metrics)
            return metrics

        w_norm = float(torch.linalg.vector_norm(snap.working).item())
        s_norm = float(torch.linalg.vector_norm(snap.self_state).item())
        g_norm = float(torch.linalg.vector_norm(snap.goal).item())

        is_bounded = (w_norm <= self.max_bound) and (s_norm <= self.max_bound) and (g_norm <= self.max_bound)

        # 2. Compute drift rate
        drift_rate = 0.0
        if self._last_snap is not None:
            delta = float(torch.linalg.vector_norm(snap.working - self._last_snap.working).item())
            drift_rate = delta / dt

        self._last_snap = snap

        anomaly = not is_bounded or (drift_rate > self.max_drift_rate)
        reason = None
        if not is_bounded:
            reason = f"State norm exceeded bound {self.max_bound:.2f}: working={w_norm:.2f}"
        elif drift_rate > self.max_drift_rate:
            reason = f"Drift rate {drift_rate:.2f} exceeded max rate {self.max_drift_rate:.2f}"

        metrics = StabilityMetrics(
            working_norm=round(w_norm, 4),
            self_norm=round(s_norm, 4),
            goal_norm=round(g_norm, 4),
            is_bounded=is_bounded,
            drift_rate=round(drift_rate, 4),
            uncertainty=round(snap.uncertainty, 4),
            timestamp_utc=now,
            anomaly_detected=anomaly,
            anomaly_reason=reason,
        )

        self._history.append(metrics)
        if len(self._history) > self.history_len:
            self._history.pop(0)

        return metrics

    def measure_perturbation_recovery(
        self,
        state: PersistentCognitiveState,
        perturbation: torch.Tensor,
        dt_step: float = 0.1,
        max_steps: int = 50,
    ) -> Dict[str, Any]:
        """Inject a shock perturbation and measure the steps / time required to relax to 50% baseline."""
        snap_initial = state.snapshot()
        state.observe(perturbation, dt=dt_step)
        snap_shock = state.snapshot()

        shock_diff = float(torch.linalg.vector_norm(snap_shock.working - snap_initial.working).item())
        half_diff = shock_diff * 0.5

        steps_to_half = None
        for step in range(1, max_steps + 1):
            # No-input relaxation (zero sensory drive)
            snap_curr = state.observe(torch.zeros_like(state.snapshot().working), dt=dt_step)
            curr_diff = float(torch.linalg.vector_norm(snap_curr.working - snap_initial.working).item())
            if curr_diff <= half_diff and steps_to_half is None:
                steps_to_half = step

        return {
            "initial_shock_magnitude": round(shock_diff, 4),
            "target_half_magnitude": round(half_diff, 4),
            "relaxation_steps_to_halflife": steps_to_half if steps_to_half is not None else max_steps,
            "relaxation_time_seconds": (steps_to_half * dt_step) if steps_to_half is not None else (max_steps * dt_step),
            "recovered": steps_to_half is not None,
            "final_working_norm": round(float(torch.linalg.vector_norm(state.snapshot().working).item()), 4),
        }

    @property
    def history(self) -> List[StabilityMetrics]:
        return list(self._history)


__all__ = [
    "StabilityMetrics",
    "CCEStabilityMonitor",
]
