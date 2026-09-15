"""Canonical CCE transaction coordinator.

The engine separates proposal, governance, capability authorization, safety
mediation, transition validation, execution and commit. Model output is never
itself authority; external effects occur only after every configured gate has
approved the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Callable, Mapping, Optional, Sequence

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.trajectory import CognitiveTrajectory
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken, TrustTier
from nsa.core.state import CanonicalState, StateTransition
from nsa.core.transition import TransitionProposal, TransitionReceipt, TransitionValidator, state_digest

PolicyHook = Callable[[CanonicalState, ActionCandidate], bool | tuple[bool, str]]
GateHook = Callable[[CanonicalState, ActionCandidate], bool | tuple[bool, str]]
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
    """Turn a cognitive tick into one governed, auditable state transaction."""

    def __init__(
        self,
        initial_state: CanonicalState,
        *,
        selector: Optional[Callable[[CanonicalState, Sequence[ActionCandidate]], ActionCandidate | None]] = None,
        policy: Optional[PolicyHook] = None,
        safety_gate: Optional[GateHook] = None,
        executor: Optional[ExecutionHook] = None,
        validator: Optional[TransitionValidator] = None,
        trajectory: Optional[CognitiveTrajectory] = None,
        capability_authority: Optional[CapabilityAuthority] = None,
        capability_tokens: Optional[Mapping[str, CapabilityToken]] = None,
    ) -> None:
        self.state = initial_state
        self.selector = selector or self._default_selector
        self.policy = policy
        self.safety_gate = safety_gate
        self.executor = executor
        self.validator = validator or TransitionValidator()
        self.trajectory = trajectory or CognitiveTrajectory(initial_state)
        self.capability_authority = capability_authority
        self.capability_tokens = dict(capability_tokens or {})
        self._event_counter = 0

    def _event(self, kind: EventKind, payload: Mapping[str, Any]) -> CognitiveEvent:
        self._event_counter += 1
        return CognitiveEvent(kind, self.state.step, f"evt-{self.state.step}-{self._event_counter}", dict(payload))

    @staticmethod
    def _default_selector(state: CanonicalState, candidates: Sequence[ActionCandidate]) -> ActionCandidate | None:
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.expected_utility - c.risk, c.reversible))

    @staticmethod
    def _decision(hook: GateHook, state: CanonicalState, action: ActionCandidate) -> tuple[bool, str]:
        decision = hook(state, action)
        if isinstance(decision, tuple):
            return bool(decision[0]), str(decision[1])
        return bool(decision), "gate allowed" if decision else "gate denied"

    @staticmethod
    def _required_tier(action: ActionCandidate) -> TrustTier:
        # Capability scope must reflect the effect class rather than epistemic confidence.
        return TrustTier.T2_REVERSIBLE if action.reversible else TrustTier.T3_SIDE_EFFECTS

    @staticmethod
    def _check_constraints(token: CapabilityToken, action: ActionCandidate) -> Optional[str]:
        constraints = token.constraints
        max_risk = constraints.get("max_risk")
        if max_risk is not None and action.risk > float(max_risk):
            return f"capability constraint max_risk={max_risk} violated by risk={action.risk}"
        if constraints.get("reversible_only") and not action.reversible:
            return "capability constraint reversible_only violated"
        return None

    def _capability_gate(self, action: ActionCandidate) -> tuple[bool, str]:
        """Check declared capabilities and consume supplied cryptographic tokens."""
        required = tuple(action.required_capabilities)
        if not required:
            return True, "no capabilities required"

        for capability in required:
            if self.state.hard.has_permission(capability):
                continue
            token = self.capability_tokens.get(capability) or self.capability_tokens.get(action.action_id)
            if token is None:
                return False, f"missing capability: {capability}"
            constraint_reason = self._check_constraints(token, action)
            if constraint_reason is not None:
                return False, constraint_reason
            if self.capability_authority is None:
                return False, "capability token supplied without a capability authority"
            ok, reason = self.capability_authority.verify_and_consume_capability(
                token=token,
                action_id=action.action_id,
                required_tier=self._required_tier(action),
                current_time=time(),
            )
            if not ok:
                return False, reason
        return True, "capabilities authorized"

    @staticmethod
    def _proposal_receipt(state: CanonicalState, proposal: TransitionProposal, reason: str) -> TransitionReceipt:
        source = state_digest(state)
        return TransitionReceipt(
            proposal.action_id,
            source,
            source,
            state_digest(proposal),
            False,
            reason,
            time(),
            state.step,
        )

    def tick(
        self,
        *,
        action_candidates: Sequence[ActionCandidate] = (),
        observation: Any = None,
        semantic_update: Any = None,
        soft_updates: Optional[Mapping[str, float]] = None,
        hard_transition: Optional[StateTransition] = None,
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

        proposal = TransitionProposal(
            action_id=selected.action_id if selected is not None else action_id,
            reason=reason,
            semantic=semantic_update,
            soft_updates=soft_updates,
            hard_transition=hard_transition,
            provenance_source=provenance_source,
            evidence_id=evidence_id,
            metadata=metadata,
        )

        allowed, policy_reason = True, "no policy hook"
        if selected is not None and self.policy is not None:
            allowed, policy_reason = self._decision(self.policy, self.state, selected)
            events.append(self._event(EventKind.POLICY, {"allowed": allowed, "reason": policy_reason}))
        if not allowed:
            receipt = self._proposal_receipt(self.state, proposal, policy_reason)
            events.append(self._event(EventKind.ROLLBACK, {"reason": policy_reason}))
            return CognitiveTransaction(self.state, proposal, selected, False, policy_reason, False, None, receipt, tuple(events))

        if selected is not None:
            capability_allowed, capability_reason = self._capability_gate(selected)
            events.append(self._event(EventKind.CAPABILITY, {"allowed": capability_allowed, "reason": capability_reason}))
            if not capability_allowed:
                receipt = self._proposal_receipt(self.state, proposal, capability_reason)
                events.append(self._event(EventKind.ROLLBACK, {"reason": capability_reason}))
                return CognitiveTransaction(self.state, proposal, selected, True, capability_reason, False, None, receipt, tuple(events))

            if self.safety_gate is not None:
                safety_allowed, safety_reason = self._decision(self.safety_gate, self.state, selected)
                events.append(self._event(EventKind.POLICY, {"gate": "safety", "allowed": safety_allowed, "reason": safety_reason}))
                if not safety_allowed:
                    receipt = self._proposal_receipt(self.state, proposal, safety_reason)
                    events.append(self._event(EventKind.ROLLBACK, {"reason": safety_reason}))
                    return CognitiveTransaction(self.state, proposal, selected, True, safety_reason, False, None, receipt, tuple(events))

        # Structural validation occurs before any external side effect.
        ok, validation_reason = self.validator.validate(self.state, proposal)
        if not ok:
            receipt = self._proposal_receipt(self.state, proposal, validation_reason)
            events.append(self._event(EventKind.ROLLBACK, {"reason": validation_reason}))
            return CognitiveTransaction(self.state, proposal, selected, True, validation_reason, False, None, receipt, tuple(events))

        executed = False
        execution_result: Any = None
        if selected is not None and self.executor is not None:
            try:
                execution_result = self.executor(selected, self.state)
            except Exception as exc:
                reason_text = f"execution failed: {type(exc).__name__}: {exc}"
                receipt = self._proposal_receipt(self.state, proposal, reason_text)
                events.append(self._event(EventKind.ERROR, {"reason": reason_text}))
                events.append(self._event(EventKind.ROLLBACK, {"reason": reason_text}))
                return CognitiveTransaction(self.state, proposal, selected, True, reason_text, False, None, receipt, tuple(events))
            executed = True
            events.append(self._event(EventKind.EXECUTION, {"action_id": selected.action_id, "result": execution_result}))

        next_state, receipt = self.validator.apply(self.state, proposal)
        if receipt.committed:
            events.append(self._event(EventKind.STATE_COMMIT, {
                "source_digest": receipt.source_digest,
                "target_digest": receipt.target_digest,
            }))
            self.state = next_state
        else:
            events.append(self._event(EventKind.ROLLBACK, {"reason": receipt.reason}))
        self.trajectory.append(self.state, receipt=receipt, events=events)
        return CognitiveTransaction(self.state, proposal, selected, allowed, policy_reason, executed,
                                    execution_result, receipt, tuple(events))


__all__ = ["CognitiveTransaction", "CognitiveTransactionEngine"]
