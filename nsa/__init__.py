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
    DEFAULT_LATTICE,
    BitpackedStateVector,
    ConservationLaw,
    ProductLattice,
    ProductStateVector,
    RAGMetadataIngressEncoder,
    StateLabel,
    StateLattice,
    bitpack_states,
    build_label_attention_mask,
    build_level_attention_mask,
    unpack_states,
)
from nsa.utils import print_lattice

try:
    from nsa.attention import StateAwareAttention
    from nsa.fused_attention import FusedStateAwareAttention
    from nsa.hf_integration import (
        NSAConfig,
        NSAForCausalLM,
        retrofit_hf_attention,
        retrofit_llama_attention,
    )
    from nsa.kv_cache import NSAKVCache
    from nsa.layers import NSACausalLM, NSATransformer, NSATransformerBlock
    from nsa.lora import (
        DynamicNSARetrofitBlock,
        NSALoRAAttention,
        NSALoRALinear,
        apply_nsa_lora_retrofit,
    )
    from nsa.mask_injector import NSAMaskInjector
    from nsa.objectives import NSALoss, SemanticLoss, StateConstraintLoss
    from nsa.residual_taint import ResidualTaintTracker, join_levels, meet_levels
    from nsa.state import (
        DeclassificationOperator,
        StateTransitionOperator,
        StateVector,
        WeightedStateEdge,
    )
    from nsa.triton_kernel import (
        HAS_TRITON,
        TRITON_KERNEL_DEFINED,
        USING_TRITON_KERNEL,
        FusedTritonStateAttention,
        last_backend,
    )
    from nsa.utils import count_parameters, print_model_summary, state_labels_to_vectors
    from nsa.value_layer import AlignmentStateProjector, ValueAlignmentLoss
    from nsa.verifier import (
        AdapterSwitchRecovery,
        AuditResult,
        HaltRecovery,
        MultiLayerStateAuditor,
        NSAGenerator,
        RecoveryPolicy,
        SemanticPivotRecovery,
        SpeculativeStateAuditor,
        StateControlTokens,
        StateEncoderHead,
        StreamRouter,
        TokenizerAligner,
        generate_with_auditor,
    )

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

__version__ = "0.2.0"
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
    # Mask Injection & HF Integration
    "NSAMaskInjector",
    "NSAConfig",
    "NSAForCausalLM",
    "retrofit_llama_attention",
    "retrofit_hf_attention",
    "NSAKVCache",
    # Verifier & NSA 2.0 Speculative Engine
    "StateEncoderHead",
    "SpeculativeStateAuditor",
    "MultiLayerStateAuditor",
    "AuditResult",
    "TokenizerAligner",
    "StateControlTokens",
    "StreamRouter",
    "RecoveryPolicy",
    "SemanticPivotRecovery",
    "AdapterSwitchRecovery",
    "HaltRecovery",
    "NSAGenerator",
    "generate_with_auditor",
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
