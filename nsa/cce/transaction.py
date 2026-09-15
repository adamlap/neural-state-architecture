"""Canonical CCE transaction coordinator.

This is the bridge between the scheduler and the rest of NSA.  It does not
execute tools itself: callers provide governance/execution hooks.  Every tick
produces one auditable decision and one canonical state transition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.trajectory import CognitiveTrajectory
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState
from nsa.core.transition import TransitionProposal, TransitionReceipt, TransitionValidator

PolicyHook = Callable[[CanonicalState, ActionCandidate], bool | tuple[bool, str]]
ExecutionHook = Callable[[ActionCandidate, CanonicalState], Any]


@dataclass(frozen=True)
class CognitiveTransaction:
    state: CanonicalState
    proposal: TransitionProposal
    selected_action: Optional[ActionCandidate]
    policy_allowed: bool
    policy_reason: str
    executed: bool
    execution_result: Any
    receipt: TransitionReceipt
    events: tuple[CognitiveEvent, ...]


class CognitiveTransactionEngine:
    """Turn a cognitive tick into a governed, atomic state transition."""

    def __init__(
        self,
        initial_state: CanonicalState,
        *,
        selector: Optional[Callable[[CanonicalState, Sequence[ActionCandidate]], ActionCandidate | None]] = None,
        policy: Optional[PolicyHook] = None,
        executor: Optional[ExecutionHook] = None,
        validator: Optional[TransitionValidator] = None,
        trajectory: Optional[CognitiveTrajectory] = None,
    ) -> None:
        self.state = initial_state
        self.selector = selector or self._default_selector
        self.policy = policy
        self.executor = executor
        self.validator = validator or TransitionValidator()
        self.trajectory = trajectory or CognitiveTrajectory(initial_state)
        self._event_counter = 0

    def _event(self, kind: EventKind, payload: Mapping[str, Any], *, source: str = "cce") -> CognitiveEvent:
        self._event_counter += 1
        return CognitiveEvent(kind, self.state.step, f"evt-{self.state.step}-{self._event_counter}", dict(payload), source)

    @staticmethod
    def _default_selector(state: CanonicalState, candidates: Sequence[ActionCandidate]) -> ActionCandidate | None:
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.expected_utility - c.risk, c.reversible))

    def tick(
        self,
        *,
        action_candidates: Sequence[ActionCandidate] = (),
        observation: Any = None,
        semantic_update: Any = None,
        soft_updates: Optional[Mapping[str, float]] = None,
        action_id: str = "cognitive_tick",
        reason: str = "CCE state transition",
        provenance_source: Optional[str] = None,
        evidence_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> CognitiveTransaction:
        events: list[CognitiveEvent] = []
        if observation is not None:
            events.append(self._event(EventKind.OBSERVATION, {"observation": observation}))

        selected = self.selector(self.state, action_candidates)
        if selected is not None:
            events.append(self._event(EventKind.ACTION_PROPOSAL, {
                "action_id": selected.action_id,
                "risk": selected.risk,
                "expected_utility": selected.expected_utility,
            }))

        allowed, policy_reason = True, "no policy hook"
        if selected is not None and self.policy is not None:
            decision = self.policy(self.state, selected)
            if isinstance(decision, tuple):
                allowed, policy_reason = bool(decision[0]), str(decision[1])
            else:
                allowed = bool(decision)
                policy_reason = "policy allowed" if allowed else "policy denied"
            events.append(self._event(EventKind.POLICY, {"allowed": allowed, "reason": policy_reason}))

        if selected is not None and not allowed:
            proposal = TransitionProposal(action_id, reason, metadata=metadata)
            receipt = TransitionReceipt(action_id, self.trajectory.latest.state_digest, self.trajectory.latest.state_digest,
                                        "", False, policy_reason, 0.0, self.state.step)
            events.append(self._event(EventKind.ROLLBACK, {"reason": policy_reason}))
            return CognitiveTransaction(self.state, proposal, selected, False, policy_reason, False, None, receipt, tuple(events))

        executed = False
        execution_result: Any = None
        if selected is not None and self.executor is not None:
            execution_result = self.executor(selected, self.state)
            executed = True
            events.append(self._event(EventKind.EXECUTION, {"action_id": selected.action_id, "result": execution_result}))

        proposal = TransitionProposal(
            action_id=selected.action_id if selected is not None else action_id,
            reason=reason,
            semantic=semantic_update,
            soft_updates=soft_updates,
            provenance_source=provenance_source,
            evidence_id=evidence_id,
            metadata=metadata,
        )
        next_state, receipt = self.validator.apply(self.state, proposal)
        if receipt.committed:
            events.append(self._event(EventKind.STATE_COMMIT, {
                "source_digest": receipt.source_digest,
                "target_digest": receipt.target_digest,
            }))
            self.state = next_state
            self.trajectory.append(self.state, receipt=receipt, events=events)
        else:
            events.append(self._event(EventKind.ROLLBACK, {"reason": receipt.reason}))
            self.trajectory.append(self.state, receipt=receipt, events=events)

        return CognitiveTransaction(self.state, proposal, selected, allowed, policy_reason, executed,
                                    execution_result, receipt, tuple(events))


__all__ = ["CognitiveTransaction", "CognitiveTransactionEngine"]
