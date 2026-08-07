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
from nsa.utils import print_lattice

try:
    from nsa.state import (
        StateVector,
        StateTransitionOperator,
        WeightedStateEdge,
    )
    from nsa.attention import StateAwareAttention
    from nsa.fused_attention import FusedStateAwareAttention
    from nsa.lora import NSALoRALinear, NSALoRAAttention, apply_nsa_lora_retrofit
    from nsa.triton_kernel import FusedTritonStateAttention
    from nsa.hf_integration import NSAConfig, NSAForCausalLM
    from nsa.kv_cache import NSAKVCache
    from nsa.layers import NSATransformerBlock, NSATransformer, NSACausalLM
    from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
    from nsa.utils import count_parameters, print_model_summary
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

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
    "FusedStateAwareAttention",
    "FusedTritonStateAttention",
    # LoRA Adapters & Retrofitting
    "NSALoRALinear",
    "NSALoRAAttention",
    "apply_nsa_lora_retrofit",
    # HuggingFace & KV-Cache Integration
    "NSAConfig",
    "NSAForCausalLM",
    "NSAKVCache",
    # Layers
    "NSATransformerBlock",
    "NSATransformer",
    "NSACausalLM",
    # Losses
    "SemanticLoss",
    "StateConstraintLoss",
    "NSALoss",
    # Utilities
    "count_parameters",
    "print_model_summary",
    "print_lattice",
]
