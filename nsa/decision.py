"""Explicit, auditable decisions produced by the NSA policy engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Mapping, Optional, Tuple


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    REQUIRE_APPROVAL = "require_approval"
    REDACT = "redact"


@dataclass(frozen=True)
class SecurityDecision:
    """Result of evaluating a request or proposed action."""

    decision: Decision
    policy: str
    reason: str
    matched_categories: Tuple[str, ...] = ()
    hard_constraints_triggered: FrozenSet[str] = frozenset()
    required_capabilities: FrozenSet[str] = frozenset()
    risk: float = 0.0
    uncertainty: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")

    @property
    def allowed(self) -> bool:
        return self.decision == Decision.ALLOW

    @property
    def terminal(self) -> bool:
        return self.decision in {Decision.DENY, Decision.REDACT}

    def summary(self) -> dict:
        return {
            "decision": self.decision.value,
            "policy": self.policy,
            "reason": self.reason,
            "matched_categories": self.matched_categories,
            "hard_constraints_triggered": tuple(sorted(self.hard_constraints_triggered)),
            "required_capabilities": tuple(sorted(self.required_capabilities)),
            "risk": self.risk,
            "uncertainty": self.uncertainty,
        }
