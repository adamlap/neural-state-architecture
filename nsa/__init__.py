"""Neural State Architecture public API.

The installable runtime is intentionally lightweight: importing ``nsa`` does
not import PyTorch, Transformers, Triton, or any research experiment.  Heavy
model-retrofitting features remain available from their explicit modules and
are lazily exposed for backwards compatibility.
"""
from __future__ import annotations

from importlib import import_module

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
from nsa.backends import BackendError, CallableBackend, EchoBackend, OllamaBackend
from nsa.cce.lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore
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
from nsa.decision import Decision, SecurityDecision
from nsa.enforcement import EvaluationContext, KeywordClassifier, PolicyClassifier, PolicyEngine
from nsa.policy import NSAPolicy, PolicyCompiler, PolicyRule
from nsa.runtime import AgentResult, ModelBackend, NSA, NSARuntime, RuntimeConfig

__version__ = "0.4.0"

# Explicitly heavy/legacy symbols are loaded only if a caller asks for them.
_LAZY = {
    "StateAwareAttention": ("nsa.attention", "StateAwareAttention"),
    "FusedStateAwareAttention": ("nsa.fused_attention", "FusedStateAwareAttention"),
    "NSAConfig": ("nsa.hf_integration", "NSAConfig"),
    "NSAForCausalLM": ("nsa.hf_integration", "NSAForCausalLM"),
    "retrofit_hf_attention": ("nsa.hf_integration", "retrofit_hf_attention"),
    "retrofit_llama_attention": ("nsa.hf_integration", "retrofit_llama_attention"),
    "NSAKVCache": ("nsa.kv_cache", "NSAKVCache"),
    "NSALoRALinear": ("nsa.lora", "NSALoRALinear"),
    "NSALoRAAttention": ("nsa.lora", "NSALoRAAttention"),
    "apply_nsa_lora_retrofit": ("nsa.lora", "apply_nsa_lora_retrofit"),
    "NSAMaskInjector": ("nsa.mask_injector", "NSAMaskInjector"),
    "NSALoss": ("nsa.objectives", "NSALoss"),
    "SemanticLoss": ("nsa.objectives", "SemanticLoss"),
    "StateConstraintLoss": ("nsa.objectives", "StateConstraintLoss"),
    "NSATransformerBlock": ("nsa.layers", "NSATransformerBlock"),
    "NSATransformer": ("nsa.layers", "NSATransformer"),
    "NSACausalLM": ("nsa.layers", "NSACausalLM"),
    "ResidualTaintTracker": ("nsa.residual_taint", "ResidualTaintTracker"),
    "join_levels": ("nsa.residual_taint", "join_levels"),
    "meet_levels": ("nsa.residual_taint", "meet_levels"),
    "NSAGenerator": ("nsa.verifier", "NSAGenerator"),
    "SecurityAutomaton": ("nsa.verifier", "SecurityAutomaton"),
    "generate_with_auditor": ("nsa.verifier", "generate_with_auditor"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "NSA", "NSARuntime", "AgentResult", "RuntimeConfig", "ModelBackend",
    "OllamaBackend", "EchoBackend", "CallableBackend", "BackendError",
    "CanonicalState", "SemanticState", "HardState", "SoftState", "ProvenanceState",
    "GoalState", "StateTransition", "StateKind",
    "CheckpointEnvelope", "StateCheckpointStore", "CognitiveInputEvent", "CognitiveInputQueue",
    "Decision", "SecurityDecision", "NSAPolicy", "PolicyRule", "PolicyCompiler",
    "PolicyEngine", "PolicyClassifier", "KeywordClassifier", "EvaluationContext",
    "StateLabel", "StateLattice", "ConservationLaw", "DEFAULT_LATTICE", "ProductStateVector",
    "ProductLattice", "BitpackedStateVector", "RAGMetadataIngressEncoder",
    "build_label_attention_mask", "build_level_attention_mask", "bitpack_states", "unpack_states",
] + sorted(_LAZY)
