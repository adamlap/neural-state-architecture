"""Async transaction boundary for real-world tool/effect execution."""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Sequence
from time import time

from nsa.cce.events import CognitiveEvent, EventKind
from nsa.cce.transaction import CognitiveTransaction, CognitiveTransactionEngine
from nsa.core.transition import TransitionProposal


AsyncExecutionHook = Callable[[Any, Any], Awaitable[Any]]


class AsyncCognitiveTransactionEngine(CognitiveTransactionEngine):
    """CCE transaction engine that awaits side effects before state commit.

    This is the bridge needed by async MCP/HTTP/tool runtimes: model proposals
    are validated first, the effect is awaited, and canonical state is committed
    only after the effect succeeds. A failed effect therefore cannot appear as
    a successful canonical action merely because an event loop was involved.
    """

    async def tick_async(
        self,
        *,
        action_candidates=(),
        observation=None,
        semantic_update=None,
        soft_updates=None,
        hard_transition=None,
        action_id="cognitive_tick",
        reason="CCE state transition",
        provenance_source=None,
        evidence_id=None,
        metadata=None,
        cognitive_events: Sequence[CognitiveEvent] = (),
        executor: AsyncExecutionHook | None = None,
    ) -> CognitiveTransaction:
        events = list(cognitive_events)
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
            action_id=selected.action_id if selected else action_id,
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
            return self._reject(proposal, selected, False, policy_reason, events)

        verified_tokens = ()
        if selected is not None:
            capability_allowed, capability_reason, verified_tokens = self._verify_capabilities(selected)
            events.append(self._event(EventKind.CAPABILITY, {
                "allowed": capability_allowed,
                "reason": capability_reason,
                "consumed": False,
            }))
            if not capability_allowed:
                return self._reject(proposal, selected, True, capability_reason, events)
            if self.safety_gate is not None:
                safety_allowed, safety_reason = self._decision(self.safety_gate, self.state, selected)
                events.append(self._event(EventKind.POLICY, {
                    "gate": "safety", "allowed": safety_allowed, "reason": safety_reason,
                }))
                if not safety_allowed:
                    return self._reject(proposal, selected, True, safety_reason, events)

        ok, validation_reason = self.validator.validate(self.state, proposal)
        if not ok:
            return self._reject(proposal, selected, True, validation_reason, events)

        if selected is not None and verified_tokens:
            consumed, consume_reason = self._consume_capabilities(verified_tokens, selected)
            events.append(self._event(EventKind.CAPABILITY, {
                "allowed": consumed, "reason": consume_reason, "consumed": consumed,
            }))
            if not consumed:
                return self._reject(proposal, selected, True, consume_reason, events)

        executed = False
        execution_result = None
        if selected is not None:
            if executor is None:
                return self._reject(proposal, selected, True, "async executor required", events)
            try:
                execution_result = await executor(selected, self.state)
                executed = True
                events.append(self._event(EventKind.EXECUTION, {
                    "action_id": selected.action_id,
                    "result": execution_result,
                }))
            except Exception as exc:
                reason_text = f"execution failed: {type(exc).__name__}: {exc}"
                events.append(self._event(EventKind.ERROR, {"reason": reason_text}))
                return self._reject(proposal, selected, True, reason_text, events)

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
        return CognitiveTransaction(
            self.state, proposal, selected, allowed, policy_reason,
            executed, execution_result, receipt, tuple(events), None,
        )


__all__ = ["AsyncCognitiveTransactionEngine", "AsyncExecutionHook"]
