"""Canonical typed state model for the Neural State Architecture.

The core design principle is that semantic cognition and authoritative state
are distinct streams carried by one typed activation.  Hard state is not a
free-form tensor that model-generated semantics can mutate directly.

The model is intentionally conservative: this module defines the state
contract and transition semantics; it does not claim to implement
consciousness, AGI alignment, or a complete trusted runtime.

Canonical activation:

    H_t = (M_t, Sigma_h,t, Sigma_s,t, Pi_t, G_t)

where:
    M       semantic representation
    Sigma_h hard/trusted policy state
    Sigma_s soft operational/epistemic state
    Pi      provenance/evidence state
    G       goal/intention state

Capabilities are deliberately represented as a separate authority boundary
in the existing NSA algebra rather than as a model-controlled field here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, FrozenSet, Mapping, Optional, Tuple

from nsa.algebra import ConfidentialityLabel, IntegrityLabel


class StateKind(str, Enum):
    """Trust/semantic class of a state component."""

    HARD = "hard"
    SOFT = "soft"
    PROVENANCE = "provenance"
    GOAL = "goal"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class SemanticState:
    """Model-controlled semantic representation.

    ``value`` is intentionally opaque: it may be a tensor, token sequence,
    latent representation, or another framework-specific object.  The core
    must not infer authority from its contents.
    """

    value: Any
    kind: StateKind = field(default=StateKind.SEMANTIC, init=False)


@dataclass(frozen=True)
class HardState:
    """Authoritative state whose security meaning must survive computation.

    These fields are deliberately constrained to typed values.  Arbitrary
    model-generated metadata should live in ``SoftState`` or provenance until
    an external authority explicitly promotes it.
    """

    confidentiality: ConfidentialityLabel = ConfidentialityLabel.PUBLIC
    integrity: IntegrityLabel = IntegrityLabel.TRUSTED
    authorizations: FrozenSet[str] = field(default_factory=frozenset)
    license_tier: int = 0
    kind: StateKind = field(default=StateKind.HARD, init=False)

    def join(self, other: "HardState") -> "HardState":
        """Conservative component-wise composition of trusted state."""
        return HardState(
            confidentiality=self.confidentiality.join(other.confidentiality),
            integrity=self.integrity.join(other.integrity),
            authorizations=self.authorizations | other.authorizations,
            license_tier=max(self.license_tier, other.license_tier),
        )

    def meet(self, other: "HardState") -> "HardState":
        """Component-wise meet of trusted state."""
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
    """Operational/epistemic state that can be estimated by the model.

    ``uncertainty`` and ``risk`` use [0, 1], where 1 means maximally
    uncertain/risky.  These values are *signals*, not hard guarantees.
    """

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
        """Worst-case operational composition."""
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
    """Machine-readable lineage for information represented by an activation."""

    sources: Tuple[str, ...] = ()
    transformations: Tuple[str, ...] = ()
    evidence_ids: Tuple[str, ...] = ()
    trust_domain: Optional[str] = None
    timestamp: Optional[float] = None
    kind: StateKind = field(default=StateKind.PROVENANCE, init=False)

    def extend(
        self,
        *,
        source: Optional[str] = None,
        transformation: Optional[str] = None,
        evidence_id: Optional[str] = None,
    ) -> "ProvenanceState":
        """Return a new lineage record; provenance is append-only by default."""
        return replace(
            self,
            sources=self.sources + ((source,) if source else ()),
            transformations=self.transformations + ((transformation,) if transformation else ()),
            evidence_ids=self.evidence_ids + ((evidence_id,) if evidence_id else ()),
        )


@dataclass(frozen=True)
class GoalState:
    """Explicit task/intent state kept separate from hard authority."""

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
    """Canonical NSA activation state.

    The semantic stream may propose state observations, but authoritative
    ``hard`` state should only change through an explicit transition object.
    This gives every future NSA subsystem a common contract.
    """

    semantic: SemanticState
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
        """Return the mathematical product-state representation."""
        return (self.semantic, self.hard, self.soft, self.provenance, self.goals)

    def observe(self, **soft_updates: float) -> "CanonicalState":
        """Update only soft operational state.

        This is intentionally separate from ``transition``: observation of a
        changed model estimate cannot silently grant or revoke hard authority.
        """
        allowed = {"uncertainty", "risk", "confidence", "resource_pressure"}
        unknown = set(soft_updates) - allowed
        if unknown:
            raise ValueError(f"unknown soft-state fields: {sorted(unknown)}")
        new_soft = replace(self.soft, **soft_updates)
        return replace(self, soft=new_soft, step=self.step + 1)

    def with_semantic(self, value: Any) -> "CanonicalState":
        """Return a new state with updated semantic content only."""
        return replace(self, semantic=SemanticState(value), step=self.step + 1)

    def with_goal(self, goal: GoalState) -> "CanonicalState":
        """Update explicit goal state without changing hard authority."""
        return replace(self, goals=goal, step=self.step + 1)

    def transition(self, transition: "StateTransition") -> "CanonicalState":
        """Apply an explicitly authorized state transition.

        The transition object is the only canonical-core API that can replace
        hard state.  It is deliberately not inferred from semantic content.
        """
        if transition.source != self.hard:
            raise ValueError("transition source does not match current hard state")
        if not transition.authorized:
            raise PermissionError("hard-state transition is not authorized")
        return replace(self, hard=transition.target, step=self.step + 1)

    def summary(self) -> Mapping[str, Any]:
        """Stable machine-readable summary for diagnostics and future runtimes."""
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
    """Explicit proposal/authorization for replacing hard state."""

    source: HardState
    target: HardState
    authorized: bool = False
    capability_id: Optional[str] = None
    reason: Optional[str] = None

    def authorize(self, capability_id: str, reason: Optional[str] = None) -> "StateTransition":
        """Return an explicitly authorized copy of this transition."""
        if not capability_id:
            raise ValueError("capability_id must be non-empty")
        return replace(self, authorized=True, capability_id=capability_id, reason=reason)


__all__ = [
    "CanonicalState",
    "GoalState",
    "HardState",
    "ProvenanceState",
    "SemanticState",
    "SoftState",
    "StateKind",
    "StateTransition",
]
