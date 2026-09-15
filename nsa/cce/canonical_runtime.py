"""Canonical CCE runtime composition."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Mapping
from nsa.cce.engine import CCEStatus, ContinuousCognitiveEngine
from nsa.cce.transaction import CognitiveTransaction, CognitiveTransactionEngine
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken
from nsa.core.state import CanonicalState

@dataclass(frozen=True)
class TickInput:
    observation: Any = None
    semantic_update: Any = None
    soft_updates: Mapping[str, float] | None = None
    action_candidates: tuple[ActionCandidate, ...] = ()
    action_id: str = "cognitive_tick"
    reason: str = "CCE state transition"
    provenance_source: str | None = None
    evidence_id: str | None = None
    metadata: Mapping[str, Any] | None = None

class CanonicalCCERuntime:
    """Run the canonical NSA state machine continuously or one tick at a time."""
    def __init__(self, initial_state: CanonicalState, *, selector: Callable | None = None,
                 policy: Callable | None = None, safety_gate: Callable | None = None,
                 executor: Callable | None = None, capability_authority: CapabilityAuthority | None = None,
                 capability_tokens: Mapping[str, CapabilityToken] | None = None,
                 interval_seconds: float = 0.1, enabled: bool = False, fail_closed: bool = True) -> None:
        self.transaction_engine = CognitiveTransactionEngine(
            initial_state, selector=selector, policy=policy, safety_gate=safety_gate,
            executor=executor, capability_authority=capability_authority,
            capability_tokens=capability_tokens)
        self._pending: TickInput | None = None
        self.engine = ContinuousCognitiveEngine(initial_state, self._step,
            interval_seconds=interval_seconds, enabled=enabled, fail_closed=fail_closed)

    @property
    def state(self) -> CanonicalState:
        return self.transaction_engine.state

    @property
    def trajectory(self):
        return self.transaction_engine.trajectory

    def submit(self, tick: TickInput) -> None:
        self._pending = tick

    def tick(self, tick: TickInput | None = None) -> CognitiveTransaction | None:
        if tick is not None:
            self._pending = tick
        if self._pending is None:
            result = self.transaction_engine.tick()
        else:
            item = self._pending
            self._pending = None
            result = self.transaction_engine.tick(
                observation=item.observation, semantic_update=item.semantic_update,
                soft_updates=item.soft_updates, action_candidates=item.action_candidates,
                action_id=item.action_id, reason=item.reason,
                provenance_source=item.provenance_source, evidence_id=item.evidence_id,
                metadata=item.metadata)
        self.engine.set_state(self.transaction_engine.state)
        return result

    def _step(self, _state: CanonicalState) -> CanonicalState:
        self.tick()
        return self.transaction_engine.state

    def start(self) -> bool:
        return self.engine.start()
    def stop(self, timeout: float | None = None) -> bool:
        return self.engine.stop(timeout=timeout)
    def set_enabled(self, enabled: bool) -> None:
        self.engine.set_enabled(enabled)
    def status(self) -> CCEStatus:
        return self.engine.status()

__all__ = ["CanonicalCCERuntime", "TickInput"]
