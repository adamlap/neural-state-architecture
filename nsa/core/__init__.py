"""Canonical NSA typed neural state core.

This package defines the framework-level state contract shared by future
memory, capability, provenance, metacognition and runtime modules.
"""

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
