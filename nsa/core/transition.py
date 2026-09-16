"""Authoritative, auditable cognitive state transitions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from time import time
from typing import Any, Mapping, Optional

from nsa.core.state import CanonicalState, StateTransition


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def state_digest(state: CanonicalState) -> str:
    """Hash all canonical state channels, including semantic content."""
    payload = {"summary": state.summary(), "semantic": state.semantic.value}
    return sha256(_stable(payload).encode("utf-8")).hexdigest()


def proposal_digest(proposal: "TransitionProposal") -> str:
    """Hash a proposal without pretending it is a CanonicalState."""
    payload = {
        "action_id": proposal.action_id, "reason": proposal.reason, "semantic": proposal.semantic,
        "soft": proposal.soft_updates, "hard": proposal.hard_transition,
        "source": proposal.provenance_source, "evidence": proposal.evidence_id,
        "metadata": proposal.metadata,
    }
    return sha256(_stable(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransitionProposal:
    """A proposed state change before policy/kernel validation."""
    action_id: str
    reason: str = ""
    semantic: Any = None
    soft_updates: Mapping[str, float] = None  # type: ignore[assignment]
    hard_transition: Optional[StateTransition] = None
    provenance_source: Optional[str] = None
    evidence_id: Optional[str] = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not self.action_id: raise ValueError("action_id must be non-empty")
        object.__setattr__(self, "soft_updates", dict(self.soft_updates or {}))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class TransitionReceipt:
    """Immutable receipt linking a proposal, validation and committed state."""
    action_id: str
    source_digest: str
    target_digest: str
    proposal_digest: str
    committed: bool
    reason: str
    timestamp: float
    step: int


class TransitionValidator:
    """Validate and apply proposals without allowing hard-state bypasses."""
    ALLOWED_SOFT = frozenset({"uncertainty", "risk", "confidence", "resource_pressure"})

    def validate(self, state: CanonicalState, proposal: TransitionProposal) -> tuple[bool, str]:
        unknown = set(proposal.soft_updates) - self.ALLOWED_SOFT
        if unknown: return False, f"unknown soft-state fields: {sorted(unknown)}"
        if proposal.hard_transition is not None:
            if proposal.hard_transition.source != state.hard: return False, "hard transition source does not match current state"
            if not proposal.hard_transition.authorized: return False, "hard-state transition is not authorized"
        return True, "validated"

    def apply(self, state: CanonicalState, proposal: TransitionProposal) -> tuple[CanonicalState, TransitionReceipt]:
        ok, reason = self.validate(state, proposal)
        source = state_digest(state)
        digest = proposal_digest(proposal)
        if not ok:
            return state, TransitionReceipt(proposal.action_id, source, source, digest, False, reason, time(), state.step)

        target = state
        if proposal.semantic is not None:
            target = replace(target, semantic=replace(target.semantic, value=proposal.semantic))
        if proposal.soft_updates:
            target = replace(target, soft=replace(target.soft, **proposal.soft_updates))
        if proposal.provenance_source or proposal.evidence_id:
            target = replace(target, provenance=target.provenance.extend(source=proposal.provenance_source, evidence_id=proposal.evidence_id))
        if proposal.hard_transition is not None:
            target = replace(target, hard=proposal.hard_transition.target)
        target = replace(target, step=state.step + 1)
        return target, TransitionReceipt(proposal.action_id, source, state_digest(target), digest, True, "committed", time(), target.step)


__all__ = ["TransitionProposal", "TransitionReceipt", "TransitionValidator", "proposal_digest", "state_digest"]
