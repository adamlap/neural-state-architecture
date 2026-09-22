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

    Capability-bearing actions reserve their tokens while an external async
    effect is in flight. A failed effect releases the reservation; successful
    effects consume the reserved capability before canonical state is committed.
    This prevents concurrent transactions from reusing the same one-shot token.
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

        reservations: list[tuple[Any, str]] = []
        if selected is not None and verified_tokens:
            if self.capability_authority is None:
                return self._reject(proposal, selected, True, "capability authority unavailable", events)
            for token in verified_tokens:
                reserved, reserve_reason, reservation_id = self.capability_authority.reserve_capability(
                    token, selected.action_id, self._required_tier(selected), current_time=time()
                )
                if not reserved or reservation_id is None:
                    for reserved_token, rid in reservations:
                        self.capability_authority.release_capability(reserved_token, rid)
                    return self._reject(proposal, selected, True, reserve_reason, events)
                reservations.append((token, reservation_id))
            events.append(self._event(EventKind.CAPABILITY, {
                "allowed": True,
                "reason": "capabilities reserved for async effect",
                "reserved": len(reservations),
                "consumed": False,
            }))

        executed = False
        execution_result = None
        if selected is not None:
            if executor is None:
                for token, rid in reservations:
                    self.capability_authority.release_capability(token, rid)
                return self._reject(proposal, selected, True, "async executor required", events)
            try:
                execution_result = await executor(selected, self.state)
                executed = True
                events.append(self._event(EventKind.EXECUTION, {
                    "action_id": selected.action_id,
                    "result": execution_result,
                }))
            except Exception as exc:
                for token, rid in reservations:
                    self.capability_authority.release_capability(token, rid)
                reason_text = f"execution failed: {type(exc).__name__}: {exc}"
                events.append(self._event(EventKind.ERROR, {"reason": reason_text}))
                return self._reject(proposal, selected, True, reason_text, events)
            except BaseException:
                # Cancellation (asyncio.CancelledError) must not leave the one-shot
                # capability reserved forever; release it, then let the cancel propagate.
                for token, rid in reservations:
                    self.capability_authority.release_capability(token, rid)
                raise

        if reservations:
            for token, rid in reservations:
                consumed, consume_reason = self.capability_authority.consume_reserved_capability(
                    token, rid, burn=not token.is_rate_limited
                )
                if not consumed:
                    for other_token, other_rid in reservations:
                        if other_token.nonce != token.nonce:
                            self.capability_authority.release_capability(other_token, other_rid)
                    events.append(self._event(EventKind.ERROR, {"reason": consume_reason}))
                    # the effect already ran: the audit record must say so
                    return self._reject(proposal, selected, True, consume_reason, events,
                                        executed=executed, execution_result=execution_result)
                self.constraint_evaluator.record_use(token, selected, context=self._constraint_context(selected))
            events.append(self._event(EventKind.CAPABILITY, {
                "allowed": True,
                "reason": "reserved capabilities consumed after successful effect",
                "consumed": True,
            }))

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
