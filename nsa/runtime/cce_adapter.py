"""Adapters that bind CCE scheduling to the authoritative NSA substrate.

The continuous engine deliberately knows nothing about cognition or policy. This
module supplies the production boundary: each CCE tick is converted into exactly
one ``CognitiveDynamicsSubstrate.step`` call, and only the substrate's committed
``new_omega`` becomes the next scheduler state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch

from nsa.core.omega import UnifiedCognitiveState
from nsa.runtime.cognitive_substrate import CognitiveDynamicsSubstrate
from nsa.runtime.continuous_engine import CCEStatus, ContinuousCognitiveEngine

CandidateAction = Tuple[str, torch.Tensor, float, float, bool]
CandidateProvider = Callable[[UnifiedCognitiveState], List[CandidateAction]]


@dataclass(frozen=True)
class SubstrateTransitionConfig:
    """Policy parameters forwarded unchanged to the authoritative substrate."""

    user_clearance_limit: float = 0.5
    target_action_risk: float = 1.0


class SubstrateTransition:
    """Turn a six-layer NSA substrate into the CCE ``step(state)`` contract.

    No state mutation or safety decision occurs here. ``CognitiveDynamicsSubstrate``
    remains the sole authority for epistemic evaluation, simulation, governance,
    immutable-kernel evaluation and commit/rollback.
    """

    def __init__(
        self,
        substrate: CognitiveDynamicsSubstrate,
        candidate_provider: CandidateProvider,
        *,
        config: SubstrateTransitionConfig | None = None,
    ) -> None:
        self.substrate = substrate
        self.candidate_provider = candidate_provider
        self.config = config or SubstrateTransitionConfig()

    def __call__(self, omega: UnifiedCognitiveState) -> UnifiedCognitiveState:
        candidates = self.candidate_provider(omega)
        if not candidates:
            raise ValueError("candidate_provider returned no actions")

        result = self.substrate.step(
            omega,
            candidates,
            user_clearance_limit=self.config.user_clearance_limit,
            target_action_risk=self.config.target_action_risk,
        )
        return result.new_omega


class ContinuousSubstrateRuntime:
    """Canonical composition of wall-clock CCE and the trusted substrate.

    This is intentionally a thin composition layer. It does not duplicate any
    transition, governance, or safety logic: ``SubstrateTransition`` remains the
    only callable supplied to ``ContinuousCognitiveEngine``. The public methods
    expose one stable lifecycle for manual and wall-clock execution.
    """

    def __init__(
        self,
        initial_state: UnifiedCognitiveState,
        substrate: CognitiveDynamicsSubstrate,
        candidate_provider: CandidateProvider,
        *,
        transition_config: SubstrateTransitionConfig | None = None,
        interval_seconds: float = 0.1,
        enabled: bool = False,
        fail_closed: bool = True,
    ) -> None:
        transition = SubstrateTransition(
            substrate,
            candidate_provider,
            config=transition_config,
        )
        self.transition = transition
        self.engine = ContinuousCognitiveEngine(
            initial_state,
            transition,
            interval_seconds=interval_seconds,
            enabled=enabled,
            fail_closed=fail_closed,
        )

    @property
    def state(self) -> UnifiedCognitiveState:
        """Return the last state committed by the authoritative transition."""
        return self.engine.state

    def tick(self) -> bool:
        """Run exactly one authoritative transition when enabled."""
        return self.engine.tick()

    def start(self) -> bool:
        """Start wall-clock execution when explicitly enabled."""
        return self.engine.start()

    def stop(self, timeout: float | None = None) -> bool:
        """Stop wall-clock execution without changing committed state."""
        return self.engine.stop(timeout=timeout)

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable future transitions; disabling stops the loop."""
        self.engine.set_enabled(enabled)

    def status(self) -> CCEStatus:
        """Return the scheduler's immutable observability snapshot."""
        return self.engine.status()


__all__ = [
    "CandidateAction",
    "CandidateProvider",
    "ContinuousSubstrateRuntime",
    "SubstrateTransition",
    "SubstrateTransitionConfig",
]
