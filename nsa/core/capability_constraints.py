"""Typed capability constraint evaluation for governed actions.

Constraints are policy data authenticated by CapabilityAuthority. This module
only evaluates them; it never grants authority and never consumes tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any, Mapping

from nsa.cognition.interfaces import ActionCandidate
from nsa.core.capabilities import CapabilityToken


@dataclass(frozen=True)
class ConstraintContext:
    """Runtime facts against which a capability may be evaluated."""
    target: str | None = None
    scope: str | None = None
    resource_cost: float = 0.0
    now: float | None = None


@dataclass(frozen=True)
class ConstraintDecision:
    allowed: bool
    reason: str
    matched: tuple[str, ...] = ()


class CapabilityConstraintEvaluator:
    """Evaluate typed, fail-closed capability constraints.

    Supported constraints:
      * ``max_risk``
      * ``reversible_only``
      * ``target`` / ``targets``
      * ``scope`` / ``scopes``
      * ``max_resource_cost``
      * ``max_calls`` + ``window_seconds`` (per action/target key)

    ``strict=False`` tolerates unknown constraint keys, for a caller that
    knowingly holds capability tokens with extension keys owned by a
    higher-level policy component. The default is ``strict=True``: a
    constraint the evaluator doesn't recognise (e.g. a typo like
    "max_rsik") must never be silently treated as "no constraint".
    """

    SUPPORTED = frozenset({
        "max_risk", "reversible_only", "target", "targets", "scope", "scopes",
        "max_resource_cost", "max_calls", "window_seconds",
    })

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self._calls: dict[tuple[str, str, str], list[float]] = {}

    def evaluate(
        self,
        token: CapabilityToken,
        action: ActionCandidate,
        *,
        context: ConstraintContext | None = None,
    ) -> ConstraintDecision:
        context = context or ConstraintContext()
        c = token.constraints
        matched: list[str] = []

        if self.strict:
            unknown = sorted(set(c) - self.SUPPORTED)
            if unknown:
                return ConstraintDecision(False, f"unsupported capability constraints: {unknown}")

        max_risk = c.get("max_risk")
        if max_risk is not None:
            if action.risk > float(max_risk):
                return ConstraintDecision(False, f"max_risk={max_risk} violated by risk={action.risk}", tuple(matched))
            matched.append("max_risk")

        if c.get("reversible_only") and not action.reversible:
            return ConstraintDecision(False, "reversible_only constraint violated", tuple(matched))
        if "reversible_only" in c:
            matched.append("reversible_only")

        target = context.target
        allowed_targets = c.get("targets", c.get("target"))
        if allowed_targets is not None:
            values = {str(v) for v in (allowed_targets if isinstance(allowed_targets, (list, tuple, set, frozenset)) else [allowed_targets])}
            if target is None or target not in values:
                return ConstraintDecision(False, f"target {target!r} is outside capability target set", tuple(matched))
            matched.append("target")

        allowed_scopes = c.get("scopes", c.get("scope"))
        if allowed_scopes is not None:
            values = {str(v) for v in (allowed_scopes if isinstance(allowed_scopes, (list, tuple, set, frozenset)) else [allowed_scopes])}
            if context.scope is None or context.scope not in values:
                return ConstraintDecision(False, f"scope {context.scope!r} is outside capability scope set", tuple(matched))
            matched.append("scope")

        max_cost = c.get("max_resource_cost")
        if max_cost is not None:
            if context.resource_cost > float(max_cost):
                return ConstraintDecision(False, f"max_resource_cost={max_cost} violated by cost={context.resource_cost}", tuple(matched))
            matched.append("max_resource_cost")

        max_calls = c.get("max_calls")
        if max_calls is not None:
            window = float(c.get("window_seconds", 60.0))
            if window <= 0 or int(max_calls) < 1:
                return ConstraintDecision(False, "invalid rate-limit constraint", tuple(matched))
            now = context.now if context.now is not None else monotonic()
            key = (token.nonce, action.action_id, str(target or ""))
            recent = [t for t in self._calls.get(key, []) if now - t < window]
            if len(recent) >= int(max_calls):
                return ConstraintDecision(False, f"rate limit max_calls={max_calls}/{window}s exceeded", tuple(matched))
            matched.append("rate_limit")

        return ConstraintDecision(True, "capability constraints satisfied", tuple(matched))

    def record_use(
        self,
        token: CapabilityToken,
        action: ActionCandidate,
        *,
        context: ConstraintContext | None = None,
    ) -> None:
        c = token.constraints
        if "max_calls" not in c:
            return
        context = context or ConstraintContext()
        now = context.now if context.now is not None else monotonic()
        key = (token.nonce, action.action_id, str(context.target or ""))
        window = float(c.get("window_seconds", 60.0))
        self._calls[key] = [t for t in self._calls.get(key, []) if now - t < window] + [now]


__all__ = ["CapabilityConstraintEvaluator", "ConstraintContext", "ConstraintDecision"]
