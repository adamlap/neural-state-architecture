"""Generic runtime action contracts for CCE.

Capabilities are opaque deployment data. CCE does not define a privileged
command catalogue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ActionProposal:
    capability: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    risk: float = 1.0
    provenance: str = "model"
    reversible: bool = True


@dataclass(frozen=True)
class GovernanceDecision:
    status: str
    reason: str
    proposal: ActionProposal

    @property
    def allowed(self) -> bool:
        return self.status == "ALLOW"
