"""
Neural State Architecture (NSA)
================================
A mathematical framework for typed neural computation.

Core concept: every activation is a pair (m, σ)
  m = semantic representation
  σ = state vector (permissions, provenance, confidence, trust, ...)

Information flow is governed by a state algebra with conservation laws.
"""

from nsa.algebra import (
    StateLabel,
    StateLattice,
    ConservationLaw,
    DEFAULT_LATTICE,
)
from nsa.state import (
    StateVector,
    StateTransitionOperator,
    WeightedStateEdge,
)
from nsa.attention import StateAwareAttention
from nsa.layers import NSATransformerBlock
from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
from nsa.utils import count_parameters, print_model_summary, print_lattice

__version__ = "0.1.0"
__all__ = [
    # Algebra
    "StateLabel",
    "StateLattice",
    "ConservationLaw",
    "DEFAULT_LATTICE",
    # State primitives
    "StateVector",
    "StateTransitionOperator",
    "WeightedStateEdge",
    # Attention
    "StateAwareAttention",
    # Layers
    "NSATransformerBlock",
    # Objectives
    "SemanticLoss",
    "StateConstraintLoss",
    "NSALoss",
    # Utils
    "count_parameters",
    "print_model_summary",
    "print_lattice",
]
