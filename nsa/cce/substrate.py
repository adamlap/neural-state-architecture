"""Optional bridge from the public CCE scheduler to NSA's six-layer substrate.

Heavy tensor/model dependencies are imported only when this adapter is
instantiated. The public ``nsa`` import therefore remains lightweight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from nsa.cce.engine import CCEStatus, ContinuousCognitiveEngine

CandidateAction = tuple[str, Any, float, float, bool]
CandidateProvider = Callable[[Any], Sequence[CandidateAction]]


@dataclass(frozen=True)
class SubstrateTransitionConfig:
    user_clearance_limit: float = 0.5
    target_action_risk: float = 1.0


class SubstrateTransition:
    """Bind one CCE tick to exactly one authoritative substrate step."""

    def __init__(self, substrate: Any, candidate_provider: CandidateProvider, *, config: SubstrateTransitionConfig | None = None) -> None:
        self.substrate = substrate
        self.candidate_provider = candidate_provider
        self.config = config or SubstrateTransitionConfig()

    def __call__(self, omega: Any) -> Any:
        candidates = list(self.candidate_provider(omega))
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
    """Canonical continuous composition for the six-layer cognitive substrate."""

    def __init__(
        self,
        initial_state: Any,
        substrate: Any,
        candidate_provider: CandidateProvider,
        *,
        transition_config: SubstrateTransitionConfig | None = None,
        interval_seconds: float = 0.1,
        enabled: bool = False,
        fail_closed: bool = True,
    ) -> None:
        self.transition = SubstrateTransition(substrate, candidate_provider, config=transition_config)
        self.engine = ContinuousCognitiveEngine(
            initial_state,
            self.transition,
            interval_seconds=interval_seconds,
            enabled=enabled,
            fail_closed=fail_closed,
        )

    @property
    def state(self) -> Any:
        return self.engine.state

    def tick(self) -> bool:
        return self.engine.tick()

    def start(self) -> bool:
        return self.engine.start()

    def stop(self, timeout: float | None = None) -> bool:
        return self.engine.stop(timeout=timeout)

    def set_enabled(self, enabled: bool) -> None:
        self.engine.set_enabled(enabled)

    def status(self) -> CCEStatus:
        return self.engine.status()


__all__ = ["CandidateAction", "CandidateProvider", "ContinuousSubstrateRuntime", "SubstrateTransition", "SubstrateTransitionConfig"]
