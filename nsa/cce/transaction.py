"""Canonical CCE transaction coordinator."""
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

PolicyHook = Callable[[Any, ActionCandidate], bool | tuple[bool, str]]
GateHook = Callable[[Any, ActionCandidate], bool | tuple[bool, str]]
ExecutionHook = Callable[[ActionCandidate, Any], Any]


@dataclass(frozen=True)
class CognitiveTransaction:
    state: Any
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
    def __init__(self, initial_state, *, selector=None, policy=None, safety_gate=None, executor=None,
                 validator=None, trajectory=None, capability_authority=None, capability_tokens=None):
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

    def _event(self, kind, payload):
        self._event_counter += 1
        return CognitiveEvent(kind, self.state.step, f"evt-{self.state.step}-{self._event_counter}", dict(payload))

    @staticmethod
    def _default_selector(state, candidates):
        if not candidates:
            return None
        return max(candidates, key=lambda c: (c.expected_utility - c.risk, c.reversible))

    @staticmethod
    def _decision(hook, state, action):
        decision = hook(state, action)
        if isinstance(decision, tuple):
            return bool(decision[0]), str(decision[1])
        return bool(decision), "gate allowed" if decision else "gate denied"

    @staticmethod
    def _required_tier(action):
        return TrustTier.T2_REVERSIBLE if action.reversible else TrustTier.T3_SIDE_EFFECTS

    @staticmethod
    def _check_constraints(token, action):
        max_risk = token.constraints.get("max_risk")
        if max_risk is not None and action.risk > float(max_risk):
            return f"capability constraint max_risk={max_risk} violated by risk={action.risk}"
        if token.constraints.get("reversible_only") and not action.reversible:
            return "capability constraint reversible_only violated"
        return None

    def _verify_capabilities(self, action):
        required = tuple(action.required_capabilities)
        if not required:
            return True, "no capabilities required", ()
        verified = []
        for capability in required:
            if self.state.hard.has_permission(capability):
                continue
            token = self.capability_tokens.get(capability) or self.capability_tokens.get(action.action_id)
            if token is None:
                return False, f"missing capability: {capability}", ()
            reason = self._check_constraints(token, action)
            if reason is not None:
                return False, reason, ()
            if self.capability_authority is None:
                return False, "capability token supplied without a capability authority", ()
            ok, reason = self.capability_authority.verify_capability(token, action.action_id, self._required_tier(action), time())
            if not ok:
                return False, reason, ()
            verified.append(token)
        return True, "capabilities verified", tuple(verified)

    def _consume_capabilities(self, tokens):
        if self.capability_authority is None:
            return True, "no cryptographic capabilities to consume"
        for token in tokens:
            ok, reason = self.capability_authority.consume_capability(token)
            if not ok:
                return False, reason
        return True, "capabilities consumed"

    @staticmethod
    def _proposal_receipt(state, proposal, reason):
        source = state_digest(state)
        return TransitionReceipt(proposal.action_id, source, source, state_digest(proposal), False, reason, time(), state.step)

    def _reject(self, proposal, selected, policy_allowed, reason, events):
        receipt = self._proposal_receipt(self.state, proposal, reason)
        events.append(self._event(EventKind.ROLLBACK, {"reason": reason}))
        self.trajectory.append(self.state, receipt=receipt, events=events)
        return CognitiveTransaction(self.state, proposal, selected, policy_allowed, reason, False, None, receipt, tuple(events))

    def tick(self, *, action_candidates=(), observation=None, semantic_update=None, soft_updates=None,
             hard_transition=None, action_id="cognitive_tick", reason="CCE state transition",
             provenance_source=None, evidence_id=None, metadata=None,
             cognitive_events: Sequence[CognitiveEvent] = ()):
        events = list(cognitive_events)
        if observation is not None:
            events.append(self._event(EventKind.OBSERVATION, {"observation": observation}))
        selected = self.selector(self.state, action_candidates)
        if selected is not None:
            events.append(self._event(EventKind.ACTION_PROPOSAL, {"action_id": selected.action_id,
                "risk": selected.risk, "expected_utility": selected.expected_utility}))
        proposal = TransitionProposal(action_id=selected.action_id if selected else action_id, reason=reason,
            semantic=semantic_update, soft_updates=soft_updates, hard_transition=hard_transition,
            provenance_source=provenance_source, evidence_id=evidence_id, metadata=metadata)

        allowed, policy_reason = True, "no policy hook"
        if selected is not None and self.policy is not None:
            allowed, policy_reason = self._decision(self.policy, self.state, selected)
            events.append(self._event(EventKind.POLICY, {"allowed": allowed, "reason": policy_reason}))
        if not allowed:
            return self._reject(proposal, selected, False, policy_reason, events)

        verified_tokens = ()
        if selected is not None:
            capability_allowed, capability_reason, verified_tokens = self._verify_capabilities(selected)
            events.append(self._event(EventKind.CAPABILITY, {"allowed": capability_allowed, "reason": capability_reason, "consumed": False}))
            if not capability_allowed:
                return self._reject(proposal, selected, True, capability_reason, events)
            if self.safety_gate is not None:
                safety_allowed, safety_reason = self._decision(self.safety_gate, self.state, selected)
                events.append(self._event(EventKind.POLICY, {"gate": "safety", "allowed": safety_allowed, "reason": safety_reason}))
                if not safety_allowed:
                    return self._reject(proposal, selected, True, safety_reason, events)

        ok, validation_reason = self.validator.validate(self.state, proposal)
        if not ok:
            return self._reject(proposal, selected, True, validation_reason, events)

        if selected is not None and verified_tokens:
            consumed, consume_reason = self._consume_capabilities(verified_tokens)
            events.append(self._event(EventKind.CAPABILITY, {"allowed": consumed, "reason": consume_reason, "consumed": consumed}))
            if not consumed:
                return self._reject(proposal, selected, True, consume_reason, events)

        executed, execution_result = False, None
        if selected is not None and self.executor is not None:
            try:
                execution_result = self.executor(selected, self.state)
            except Exception as exc:
                reason_text = f"execution failed: {type(exc).__name__}: {exc}"
                events.append(self._event(EventKind.ERROR, {"reason": reason_text}))
                return self._reject(proposal, selected, True, reason_text, events)
            executed = True
            events.append(self._event(EventKind.EXECUTION, {"action_id": selected.action_id, "result": execution_result}))

        next_state, receipt = self.validator.apply(self.state, proposal)
        if receipt.committed:
            events.append(self._event(EventKind.STATE_COMMIT, {"source_digest": receipt.source_digest, "target_digest": receipt.target_digest}))
            self.state = next_state
        else:
            events.append(self._event(EventKind.ROLLBACK, {"reason": receipt.reason}))
        self.trajectory.append(self.state, receipt=receipt, events=events)
        return CognitiveTransaction(self.state, proposal, selected, allowed, policy_reason, executed, execution_result, receipt, tuple(events))


__all__ = ["CognitiveTransaction", "CognitiveTransactionEngine"]
