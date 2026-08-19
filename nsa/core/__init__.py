"""Canonical NSA typed neural state core.

This package defines the framework-level state contract shared by future
memory, capability, provenance, metacognition and runtime modules.
"""

from nsa.core.heterogeneous_algebra import (
    BooleanDomain,
    CapabilityDomain,
    ConstraintSetDomain,
    EnumDomain,
    HeterogeneousState,
    NumericRangeDomain,
    ProbabilityInterval,
    ProbabilityIntervalDomain,
    TemporalWindow,
    TemporalWindowDomain,
)
from nsa.core.state import (
    CanonicalState,
    GoalState,
    HardState,
    ProvenanceState,
    SemanticState,
    SoftState,
    StateKind,
    StateTransition,
)
from nsa.core.transition_cone import TransitionCone, TransitionDirection

__all__ = [
    "BooleanDomain",
    "CapabilityDomain",
    "ConstraintSetDomain",
    "EnumDomain",
    "HeterogeneousState",
    "NumericRangeDomain",
    "ProbabilityInterval",
    "ProbabilityIntervalDomain",
    "TemporalWindow",
    "TemporalWindowDomain",
    "TransitionCone",
    "TransitionDirection",
    "CanonicalState",
    "GoalState",
    "HardState",
    "ProvenanceState",
    "SemanticState",
    "SoftState",
    "StateKind",
    "StateTransition",
]
