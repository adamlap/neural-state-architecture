"""Canonical typed state model for the Neural State Architecture."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from nsa.algebra import ConfidentialityLabel, IntegrityLabel


class StateKind(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    PROVENANCE = "provenance"
    GOAL = "goal"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class SemanticState:
    value: Any = None
    kind: StateKind = field(default=StateKind.SEMANTIC, init=False)


@dataclass(frozen=True)
class HardState:
    confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC
    integrity: IntegrityLabel = IntegrityLabel.TRUSTED
    authorizations: FrozenSet[str] = field(default_factory=frozenset)
    license_tier: int = 0
    kind: StateKind = field(default=StateKind.HARD, init=False)

    def join(self, other: "HardState") -> "HardState":
        return HardState(
            confidentiality=self.confidentiality.join(other.confidentiality),
            integrity=self.integrity.join(other.integrity),
            authorizations=self.authorizations | other.authorizations,
            license_tier=max(self.license_tier, other.license_tier),
        )

    def meet(self, other: "HardState") -> "HardState":
        return HardState(
            confidentiality=self.confidentiality.meet(other.confidentiality),
            integrity=self.integrity.meet(other.integrity),
            authorizations=self.authorizations & other.authorizations,
            license_tier=min(self.license_tier, other.license_tier),
        )

    def has_permission(self, permission: str) -> bool:
        return permission in self.authorizations


@dataclass(frozen=True)
class SoftState:
    uncertainty: float = 0.0
    risk: float = 0.0
    confidence: float = 1.0
    resource_pressure: float = 0.0
    kind: StateKind = field(default=StateKind.SOFT, init=False)

    def __post_init__(self) -> None:
        for name in ("uncertainty", "risk", "confidence", "resource_pressure"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")

    def join(self, other: "SoftState") -> "SoftState":
        return SoftState(
            uncertainty=max(self.uncertainty, other.uncertainty),
            risk=max(self.risk, other.risk),
            confidence=min(self.confidence, other.confidence),
            resource_pressure=max(self.resource_pressure, other.resource_pressure),
        )

    def meet(self, other: "SoftState") -> "SoftState":
        return SoftState(
            uncertainty=min(self.uncertainty, other.uncertainty),
            risk=min(self.risk, other.risk),
            confidence=max(self.confidence, other.confidence),
            resource_pressure=min(self.resource_pressure, other.resource_pressure),
        )


@dataclass(frozen=True)
class ProvenanceState:
    sources: Tuple[str, ...] = ()
    transformations: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    trust_domain: Optional[str] = None
    timestamp: Optional[float] = None
    # Entries older than a retention window, folded into a constant-size hash chain (see ``fold``). Zero / empty for every state that was never folded, in which case the
    # canonical payload and digest are byte-identical to the pre-folding format.
    folded_count: int = 0
    folded_chain: str = ""
    kind: StateKind = field(default=StateKind.PROVENANCE, init=False)

    def extend(self, *, source: Optional[str] = None, transformation: Optional[str] = None,
               evidence_id: Optional[str] = None) -> "ProvenanceState":
        return replace(
            self,
            sources=self.sources + ((source,) if source else ()),
            transformations=self.transformations + ((transformation,) if transformation else ()),
            evidence_ids=self.evidence_ids + ((evidence_id,) if evidence_id else ()),
        )

    def fold(self, window: Optional[int]) -> "ProvenanceState":
        """Keep only the last ``window`` entries of each channel; commit to the dropped ones in ``folded_chain``.

        ``state_digest`` hashes the whole provenance, so without folding the cost of every transition grows linearly with the age of the agent (measured: 0.09 ms per
        observation at 500 observations, 1.4 ms at 8,000). Folding is amortised (it runs only when a channel exceeds twice the window) and tamper-evident: the chain is
        sha256(previous chain | the dropped entries), so changing any dropped entry changes the digest. Returns ``self`` when ``window`` is None or nothing needs folding.
        """
        if not window or max(len(self.sources), len(self.transformations), len(self.evidence_ids)) <= 2 * window:
            return self
        import hashlib
        import json

        dropped = {"sources": list(self.sources[:-window]), "transformations": list(self.transformations[:-window]),
                   "evidence_ids": list(self.evidence_ids[:-window])}
        chain = hashlib.sha256((self.folded_chain + "|" + json.dumps(dropped, sort_keys=True, separators=(",", ":"))).encode("utf-8")).hexdigest()
        return replace(self, sources=self.sources[-window:], transformations=self.transformations[-window:], evidence_ids=self.evidence_ids[-window:],
                       folded_count=self.folded_count + sum(len(v) for v in dropped.values()), folded_chain=chain)


@dataclass(frozen=True)
class GoalState:
    goals: Tuple[str, ...] = ()
    active_goal: Optional[str] = None
    priority: float = 1.0
    kind: StateKind = field(default=StateKind.GOAL, init=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")
        if self.active_goal is not None and self.active_goal not in self.goals:
            raise ValueError("active_goal must be present in goals")


@dataclass(frozen=True)
class CanonicalState:
    """Canonical NSA activation.

    ``semantic`` defaults to an empty semantic value for backwards
    compatibility with the original propagation API. Hard authority remains
    separately typed and can only be replaced by an explicit StateTransition.
    """

    semantic: SemanticState = field(default_factory=SemanticState)
    hard: HardState = field(default_factory=HardState)
    soft: SoftState = field(default_factory=SoftState)
    provenance: ProvenanceState = field(default_factory=ProvenanceState)
    goals: GoalState = field(default_factory=GoalState)
    step: int = 0

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")

    @property
    def state_tuple(self) -> tuple:
        return (self.semantic, self.hard, self.soft, self.provenance, self.goals)

    def observe(self, **soft_updates: float) -> "CanonicalState":
        allowed = {"uncertainty", "risk", "confidence", "resource_pressure"}
        unknown = set(soft_updates) - allowed
        if unknown:
            raise ValueError(f"unknown soft-state fields: {sorted(unknown)}")
        return replace(self, soft=replace(self.soft, **soft_updates), step=self.step + 1)

    def with_semantic(self, value: Any) -> "CanonicalState":
        return replace(self, semantic=SemanticState(value), step=self.step + 1)

    def with_goal(self, goal: GoalState) -> "CanonicalState":
        return replace(self, goals=goal, step=self.step + 1)

    def transition(self, transition: "StateTransition") -> "CanonicalState":
        if transition.source != self.hard:
            raise ValueError("transition source does not match current hard state")
        if not transition.authorized:
            raise PermissionError("hard-state transition is not authorized")
        return replace(self, hard=transition.target, step=self.step + 1)

    def summary(self) -> Mapping[str, Any]:
        return {
            "step": self.step,
            "confidentiality": self.hard.confidentiality.name,
            "integrity": self.hard.integrity.name,
            "authorizations": tuple(sorted(self.hard.authorizations)),
            "license_tier": self.hard.license_tier,
            "uncertainty": self.soft.uncertainty,
            "risk": self.soft.risk,
            "confidence": self.soft.confidence,
            "resource_pressure": self.soft.resource_pressure,
            "sources": self.provenance.sources,
            "active_goal": self.goals.active_goal,
        }


@dataclass(frozen=True)
class StateTransition:
    source: HardState
    target: HardState
    authorized: bool = False
    capability_id: Optional[str] = None
    reason: Optional[str] = None

    def authorize(self, capability_id: str, reason: Optional[str] = None) -> "StateTransition":
        if not capability_id:
            raise ValueError("capability_id must be non-empty")
        return replace(self, authorized=True, capability_id=capability_id, reason=reason)


__all__ = ["CanonicalState", "GoalState", "HardState", "ProvenanceState", "SemanticState", "SoftState", "StateKind", "StateTransition"]
