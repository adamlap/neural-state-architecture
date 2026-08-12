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
    ProductStateVector,
    ProductLattice,
    BitpackedStateVector,
    RAGMetadataIngressEncoder,
    build_label_attention_mask,
    build_level_attention_mask,
    bitpack_states,
    unpack_states,
)
from nsa.utils import print_lattice

try:
    from nsa.state import (
        StateVector,
        StateTransitionOperator,
        WeightedStateEdge,
        DeclassificationOperator,
    )
    from nsa.attention import StateAwareAttention
    from nsa.fused_attention import FusedStateAwareAttention
    from nsa.lora import (
        NSALoRALinear,
        NSALoRAAttention,
        apply_nsa_lora_retrofit,
        DynamicNSARetrofitBlock,
    )
    from nsa.triton_kernel import (
        FusedTritonStateAttention,
        HAS_TRITON,
        USING_TRITON_KERNEL,
        TRITON_KERNEL_DEFINED,
        last_backend,
    )
    from nsa.hf_integration import NSAConfig, NSAForCausalLM
    from nsa.kv_cache import NSAKVCache
    from nsa.residual_taint import ResidualTaintTracker, join_levels, meet_levels
    from nsa.layers import NSATransformerBlock, NSATransformer, NSACausalLM
    from nsa.objectives import SemanticLoss, StateConstraintLoss, NSALoss
    from nsa.value_layer import ValueAlignmentLoss, AlignmentStateProjector
    from nsa.utils import count_parameters, print_model_summary, state_labels_to_vectors
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__version__ = "0.1.2"
__all__ = [
    # Algebra
    "StateLabel",
    "StateLattice",
    "ConservationLaw",
    "DEFAULT_LATTICE",
    "ProductStateVector",
    "ProductLattice",
    "BitpackedStateVector",
    "RAGMetadataIngressEncoder",
    "build_label_attention_mask",
    "build_level_attention_mask",
    "bitpack_states",
    "unpack_states",
    # State primitives
    "StateVector",
    "StateTransitionOperator",
    "WeightedStateEdge",
    "DeclassificationOperator",
    # Attention
    "StateAwareAttention",
    "FusedStateAwareAttention",
    "FusedTritonStateAttention",
    "HAS_TRITON",
    "USING_TRITON_KERNEL",
    "TRITON_KERNEL_DEFINED",
    "last_backend",
    # LoRA Adapters & Retrofitting
    "NSALoRALinear",
    "NSALoRAAttention",
    "apply_nsa_lora_retrofit",
    "DynamicNSARetrofitBlock",
    # HuggingFace & KV-Cache Integration
    "NSAConfig",
    "NSAForCausalLM",
    "NSAKVCache",
    # Residual taint
    "ResidualTaintTracker",
    "join_levels",
    "meet_levels",
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
    "state_labels_to_vectors",
]
