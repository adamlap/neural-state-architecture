"""Canonical CCE transaction coordinator."""
from __future__ import annotations
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Mapping, Optional, Sequence
from nsa.cce.effects import EffectReceipt, TwoPhaseExecutor
from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.trajectory import CognitiveTrajectory
from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken, TrustTier
from nsa.core.capability_constraints import CapabilityConstraintEvaluator, ConstraintContext
from nsa.core.transition import TransitionProposal, TransitionReceipt, TransitionValidator, proposal_digest, state_digest

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
    effect_receipt: EffectReceipt | None = None

class CognitiveTransactionEngine:
    """One governed, auditable cognitive state transaction."""
    def __init__(self, initial_state, *, selector=None, policy=None, safety_gate=None, executor=None,
                 effect_executor: TwoPhaseExecutor | None = None, validator=None, trajectory=None,
                 capability_authority=None, capability_tokens=None, constraint_evaluator=None):
        if executor is not None and effect_executor is not None: raise ValueError("provide executor or effect_executor, not both")
        self.state = initial_state; self.selector = selector or self._default_selector
        self.policy = policy; self.safety_gate = safety_gate; self.executor = executor; self.effect_executor = effect_executor
        self.validator = validator or TransitionValidator(); self.trajectory = trajectory or CognitiveTrajectory(initial_state)
        self.capability_authority = capability_authority; self.capability_tokens = dict(capability_tokens or {})
        self.constraint_evaluator = constraint_evaluator or CapabilityConstraintEvaluator(); self._event_counter = 0

    def _event(self, kind, payload):
        self._event_counter += 1
        return CognitiveEvent(kind, self.state.step, f"evt-{self.state.step}-{self._event_counter}", dict(payload))

    @staticmethod
    def _default_selector(state, candidates):
        return max(candidates, key=lambda c: (c.expected_utility - c.risk, c.reversible)) if candidates else None

    @staticmethod
    def _decision(hook, state, action):
        decision = hook(state, action)
        return (bool(decision[0]), str(decision[1])) if isinstance(decision, tuple) else (bool(decision), "gate allowed" if decision else "gate denied")

    @staticmethod
    def _required_tier(action): return TrustTier.T2_REVERSIBLE if action.reversible else TrustTier.T3_SIDE_EFFECTS

    @staticmethod
    def _constraint_context(action):
        payload = action.payload if isinstance(action.payload, Mapping) else {}
        return ConstraintContext(target=payload.get("target"), scope=payload.get("scope"), resource_cost=float(payload.get("resource_cost", 0.0)), now=time())

    def _verify_capabilities(self, action):
        required = tuple(action.required_capabilities)
        if not required: return True, "no capabilities required", ()
        verified = []
        for capability in required:
            if self.state.hard.has_permission(capability): continue
            token = self.capability_tokens.get(capability) or self.capability_tokens.get(action.action_id)
            if token is None: return False, f"missing capability: {capability}", ()
            if self.capability_authority is None: return False, "capability token supplied without a capability authority", ()
            decision = self.constraint_evaluator.evaluate(token, action, context=self._constraint_context(action))
            if not decision.allowed: return False, f"capability constraint rejected: {decision.reason}", ()
            ok, reason = self.capability_authority.verify_capability(token, action.action_id, self._required_tier(action), time())
            if not ok: return False, reason, ()
            verified.append(token)
        return True, "capabilities verified", tuple(verified)

    def _consume_capabilities(self, tokens, action):
        if self.capability_authority is None: return True, "no cryptographic capabilities to consume"
        for token in tokens:
            ok, reason = self.capability_authority.verify_capability(token, action.action_id, self._required_tier(action), time())
            if not ok: return False, reason
        for token in tokens:
            ok, reason = self.capability_authority.consume_capability(token)
            if not ok: return False, reason
            self.constraint_evaluator.record_use(token, action, context=self._constraint_context(action))
        return True, "capabilities consumed"

    @staticmethod
    def _proposal_receipt(state, proposal, reason):
        source = state_digest(state)
        return TransitionReceipt(proposal.action_id, source, source, proposal_digest(proposal), False, reason, time(), state.step)

    def _reject(self, proposal, selected, policy_allowed, reason, events, effect_receipt=None):
        receipt = self._proposal_receipt(self.state, proposal, reason)
        events.append(self._event(EventKind.ROLLBACK, {"reason": reason})); self.trajectory.append(self.state, receipt=receipt, events=events)
        return CognitiveTransaction(self.state, proposal, selected, policy_allowed, reason, False, None, receipt, tuple(events), effect_receipt)

    def tick(self, *, action_candidates=(), observation=None, semantic_update=None, soft_updates=None, hard_transition=None,
             action_id="cognitive_tick", reason="CCE state transition", provenance_source=None, evidence_id=None, metadata=None,
             cognitive_events: Sequence[CognitiveEvent] = ()):
        events = list(cognitive_events)
        if observation is not None: events.append(self._event(EventKind.OBSERVATION, {"observation": observation}))
        selected = self.selector(self.state, action_candidates)
        if selected is not None: events.append(self._event(EventKind.ACTION_PROPOSAL, {"action_id": selected.action_id, "risk": selected.risk, "expected_utility": selected.expected_utility}))
        proposal = TransitionProposal(action_id=selected.action_id if selected else action_id, reason=reason, semantic=semantic_update, soft_updates=soft_updates,
            hard_transition=hard_transition, provenance_source=provenance_source, evidence_id=evidence_id, metadata=metadata)
        allowed, policy_reason = True, "no policy hook"
        if selected is not None and self.policy is not None:
            allowed, policy_reason = self._decision(self.policy, self.state, selected)
            events.append(self._event(EventKind.POLICY, {"allowed": allowed, "reason": policy_reason}))
        if not allowed: return self._reject(proposal, selected, False, policy_reason, events)
        verified_tokens = ()
        if selected is not None:
            capability_allowed, capability_reason, verified_tokens = self._verify_capabilities(selected)
            events.append(self._event(EventKind.CAPABILITY, {"allowed": capability_allowed, "reason": capability_reason, "consumed": False}))
            if not capability_allowed: return self._reject(proposal, selected, True, capability_reason, events)
            if self.safety_gate is not None:
                safety_allowed, safety_reason = self._decision(self.safety_gate, self.state, selected)
                events.append(self._event(EventKind.POLICY, {"gate": "safety", "allowed": safety_allowed, "reason": safety_reason}))
                if not safety_allowed: return self._reject(proposal, selected, True, safety_reason, events)
        ok, validation_reason = self.validator.validate(self.state, proposal)
        if not ok: return self._reject(proposal, selected, True, validation_reason, events)
        prepared_id = prepared = None
        if selected is not None and self.effect_executor is not None:
            try:
                prepared_id, prepared = self.effect_executor.prepare(selected, self.state)
                events.append(self._event(EventKind.EXECUTION, {"phase": "prepared", "effect_id": prepared_id}))
            except Exception as exc:
                return self._reject(proposal, selected, True, f"effect prepare failed: {type(exc).__name__}: {exc}", events)
        if selected is not None and verified_tokens:
            consumed, consume_reason = self._consume_capabilities(verified_tokens, selected)
            events.append(self._event(EventKind.CAPABILITY, {"allowed": consumed, "reason": consume_reason, "consumed": consumed}))
            if not consumed:
                if prepared_id is not None:
                    try: self.effect_executor.effect.abort(prepared)
                    except Exception: pass
                return self._reject(proposal, selected, True, consume_reason, events)
        effect_receipt = None; executed = False; execution_result = None
        if selected is not None and self.effect_executor is not None:
            effect_receipt = self.effect_executor.commit(prepared_id, prepared, selected)
            events.append(self._event(EventKind.EXECUTION, {"phase": "committed" if effect_receipt.committed else "failed", "effect_id": effect_receipt.effect_id, "result": effect_receipt.result, "reason": effect_receipt.reason}))
            if not effect_receipt.committed: return self._reject(proposal, selected, True, effect_receipt.reason, events, effect_receipt)
            executed = True; execution_result = effect_receipt.result
        elif selected is not None and self.executor is not None:
            try: execution_result = self.executor(selected, self.state)
            except Exception as exc:
                reason_text = f"execution failed: {type(exc).__name__}: {exc}"; events.append(self._event(EventKind.ERROR, {"reason": reason_text})); return self._reject(proposal, selected, True, reason_text, events)
            executed = True; events.append(self._event(EventKind.EXECUTION, {"action_id": selected.action_id, "result": execution_result}))
        next_state, receipt = self.validator.apply(self.state, proposal)
        if receipt.committed:
            events.append(self._event(EventKind.STATE_COMMIT, {"source_digest": receipt.source_digest, "target_digest": receipt.target_digest, "effect_id": effect_receipt.effect_id if effect_receipt else None})); self.state = next_state
        else:
            if effect_receipt is not None and effect_receipt.committed:
                try: effect_receipt = self.effect_executor.compensate(effect_receipt)
                except Exception: pass
            events.append(self._event(EventKind.ROLLBACK, {"reason": receipt.reason}))
        self.trajectory.append(self.state, receipt=receipt, events=events)
        return CognitiveTransaction(self.state, proposal, selected, allowed, policy_reason, executed, execution_result, receipt, tuple(events), effect_receipt)

__all__ = ["CognitiveTransaction", "CognitiveTransactionEngine"]
