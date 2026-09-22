"""Effect/commit coordination for non-idempotent external actions.

A two-phase effect coordinator lets an integration prepare an effect, commit it
only after all governance checks, and compensate an already-prepared effect if
the transaction is abandoned. The canonical state transition itself remains
pure and deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class EffectReceipt:
    effect_id: str
    action_id: str
    prepared: bool
    committed: bool
    result: Any = None
    compensation_required: bool = False
    reason: str = ""


class TwoPhaseEffect(Protocol):
    def prepare(self, action: Any, state: Any) -> Any: ...
    def commit(self, prepared: Any) -> Any: ...
    def abort(self, prepared: Any) -> None: ...
    def compensate(self, result: Any) -> None: ...


class TwoPhaseExecutor:
    """Wrap a two-phase effect and produce an auditable effect receipt."""
    def __init__(self, effect: TwoPhaseEffect) -> None:
        self.effect = effect

    def prepare(self, action: Any, state: Any) -> tuple[str, Any]:
        return uuid4().hex, self.effect.prepare(action, state)

    def commit(self, effect_id: str, prepared: Any, action: Any) -> EffectReceipt:
        try:
            result = self.effect.commit(prepared)
        except Exception as exc:
            reason = f"effect commit failed: {type(exc).__name__}: {exc}"
            try:
                self.effect.abort(prepared)
            except Exception as abort_exc:  # never let abort mask the commit failure
                reason += f"; abort failed: {type(abort_exc).__name__}: {abort_exc}"
            return EffectReceipt(effect_id, action.action_id, True, False, reason=reason)
        return EffectReceipt(effect_id, action.action_id, True, True, result=result)

    def compensate(self, receipt: EffectReceipt) -> EffectReceipt:
        if not receipt.committed:
            return receipt
        try:
            self.effect.compensate(receipt.result)
        except Exception as exc:
            return EffectReceipt(receipt.effect_id, receipt.action_id, True, True,
                                 result=receipt.result, compensation_required=True,
                                 reason=f"compensation failed: {type(exc).__name__}: {exc}")
        return EffectReceipt(receipt.effect_id, receipt.action_id, True, False,
                             result=receipt.result, reason="effect compensated")


class CallableEffect:
    """Adapter for legacy callables; useful for idempotent or transactional APIs."""
    def __init__(self, fn: Callable[[Any, Any], Any]) -> None:
        self.fn = fn

    def prepare(self, action: Any, state: Any) -> tuple[Any, Any]:
        return action, state

    def commit(self, prepared: tuple[Any, Any]) -> Any:
        action, state = prepared
        return self.fn(action, state)

    def abort(self, prepared: Any) -> None:
        return None

    def compensate(self, result: Any) -> None:
        raise RuntimeError("legacy callable has no compensation contract")


__all__ = ["CallableEffect", "EffectReceipt", "TwoPhaseEffect", "TwoPhaseExecutor"]
