"""Neural State Architecture public API.

Agent/model implementations are exposed through lazy attributes where possible.
The core package still imports existing CCE components at module load time.
"""
from __future__ import annotations
from importlib import import_module

from nsa.algebra import (
    DEFAULT_LATTICE, BitpackedStateVector, ConservationLaw, ProductLattice,
    ProductStateVector, RAGMetadataIngressEncoder, StateLabel, StateLattice,
    bitpack_states, build_label_attention_mask, build_level_attention_mask, unpack_states,
)
from nsa.backends import BackendError, CallableBackend, EchoBackend, OllamaBackend
from nsa.cce import (
    CCEStatus, CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue,
    ContinuousCognitiveEngine, StateCheckpointStore, CognitiveEvent, EventKind,
    CognitiveTrajectory, TrajectoryRecord, TrajectoryExample, TrajectoryLearner,
    CognitiveTransaction, CognitiveTransactionEngine, AsyncCognitiveTransactionEngine,
    AsyncExecutionHook, CanonicalCCERuntime, TickInput, CognitiveCycle, CognitiveLoop,
    default_prediction_error, AsyncCognitiveCycle, AsyncCognitiveLoop, CognitiveCycleResult,
    CognitiveOrchestrator, AsyncCognitiveCycleResult, AsyncCognitiveOrchestrator, AsyncEffect,
    TrajectoryJournal, record_to_dict, CallableEffect, EffectReceipt, TwoPhaseExecutor,
)
from nsa.core.state import CanonicalState, GoalState, HardState, ProvenanceState, SemanticState, SoftState, StateKind, StateTransition
from nsa.core.transition import TransitionProposal, TransitionReceipt, TransitionValidator, state_digest, proposal_digest
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken, TrustThermodynamicsVector, TrustTier
from nsa.core.capability_constraints import CapabilityConstraintEvaluator, ConstraintContext, ConstraintDecision
from nsa.core.state_codec import SCHEMA_VERSION, decode_state, dumps_state, encode_state, round_trip
from nsa.cognition import (
    ActionCandidate, ActionSelector, BeliefUpdater, InformationGainModel, PredictionError,
    Predictor, Prediction, DeliberationDecision, InformationNeed, InformationSeekingPlanner,
    UncertaintyDrivenDeliberator, CognitiveContext, CognitiveModel, CognitiveProposal,
    InformationNeedProposal, ToolRegistry, ToolSpec,
)
from nsa.decision import Decision, SecurityDecision
from nsa.enforcement import EvaluationContext, KeywordClassifier, PolicyClassifier, PolicyEngine
from nsa.policy import NSAPolicy, PolicyCompiler, PolicyRule

__version__ = "0.5.0"

_LAZY = {
    "NSA": ("nsa.agent", "NSA"), "NSARuntime": ("nsa.agent", "NSARuntime"),
    "AgentResult": ("nsa.agent", "AgentResult"), "RuntimeConfig": ("nsa.agent", "RuntimeConfig"),
    "ModelBackend": ("nsa.agent", "ModelBackend"),
    "StateAwareAttention": ("nsa.attention", "StateAwareAttention"),
    "FusedStateAwareAttention": ("nsa.fused_attention", "FusedStateAwareAttention"),
    "NSAConfig": ("nsa.hf_integration", "NSAConfig"), "NSAForCausalLM": ("nsa.hf_integration", "NSAForCausalLM"),
    "retrofit_hf_attention": ("nsa.hf_integration", "retrofit_hf_attention"),
    "retrofit_llama_attention": ("nsa.hf_integration", "retrofit_llama_attention"),
    "NSAKVCache": ("nsa.kv_cache", "NSAKVCache"), "NSALoRALinear": ("nsa.lora", "NSALoRALinear"),
    "NSALoRAAttention": ("nsa.lora", "NSALoRAAttention"), "apply_nsa_lora_retrofit": ("nsa.lora", "apply_nsa_lora_retrofit"),
    "NSAMaskInjector": ("nsa.mask_injector", "NSAMaskInjector"), "NSALoss": ("nsa.objectives", "NSALoss"),
    "SemanticLoss": ("nsa.objectives", "SemanticLoss"), "StateConstraintLoss": ("nsa.objectives", "StateConstraintLoss"),
    "NSATransformerBlock": ("nsa.layers", "NSATransformerBlock"), "NSATransformer": ("nsa.layers", "NSATransformer"),
    "NSACausalLM": ("nsa.layers", "NSACausalLM"), "ResidualTaintTracker": ("nsa.residual_taint", "ResidualTaintTracker"),
    "join_levels": ("nsa.residual_taint", "join_levels"), "meet_levels": ("nsa.residual_taint", "meet_levels"),
    "NSAGenerator": ("nsa.verifier", "NSAGenerator"),
    "SelectiveStorageTransformersBackend": ("nsa.runtime.inference.resident_transformers", "SelectiveStorageTransformersBackend"), "SecurityAutomaton": ("nsa.verifier", "SecurityAutomaton"),
    "generate_with_auditor": ("nsa.verifier", "generate_with_auditor"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None: raise AttributeError(name)
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = sorted(set([
    "NSA", "NSARuntime", "AgentResult", "RuntimeConfig", "ModelBackend", "OllamaBackend", "EchoBackend", "CallableBackend", "BackendError",
    "CCEStatus", "ContinuousCognitiveEngine", "CheckpointEnvelope", "StateCheckpointStore", "CognitiveInputEvent", "CognitiveInputQueue",
    "CognitiveEvent", "EventKind", "CognitiveTrajectory", "TrajectoryRecord", "TrajectoryExample", "TrajectoryLearner",
    "CognitiveTransaction", "CognitiveTransactionEngine", "AsyncCognitiveTransactionEngine", "AsyncExecutionHook", "CanonicalCCERuntime", "TickInput",
    "CognitiveCycle", "CognitiveLoop", "default_prediction_error", "AsyncCognitiveCycle", "AsyncCognitiveLoop", "CognitiveCycleResult", "CognitiveOrchestrator",
    "AsyncCognitiveCycleResult", "AsyncCognitiveOrchestrator", "AsyncEffect", "TrajectoryJournal", "record_to_dict", "CallableEffect", "EffectReceipt", "TwoPhaseExecutor",
    "CanonicalState", "SemanticState", "HardState", "SoftState", "ProvenanceState", "GoalState", "StateTransition", "StateKind",
    "TransitionProposal", "TransitionReceipt", "TransitionValidator", "state_digest", "proposal_digest", "CapabilityAuthority", "CapabilityToken",
    "TrustThermodynamicsVector", "TrustTier", "CapabilityConstraintEvaluator", "ConstraintContext", "ConstraintDecision", "SCHEMA_VERSION", "decode_state",
    "dumps_state", "encode_state", "round_trip", "ActionCandidate", "ActionSelector", "BeliefUpdater", "InformationGainModel", "PredictionError", "Predictor",
    "Prediction", "DeliberationDecision", "InformationNeed", "InformationSeekingPlanner", "UncertaintyDrivenDeliberator", "CognitiveContext", "CognitiveModel",
    "CognitiveProposal", "InformationNeedProposal", "ToolRegistry", "ToolSpec", "Decision", "SecurityDecision", "NSAPolicy", "PolicyRule", "PolicyCompiler",
    "PolicyEngine", "PolicyClassifier", "KeywordClassifier", "EvaluationContext", "StateLabel", "StateLattice", "ConservationLaw", "DEFAULT_LATTICE",
    "ProductStateVector", "ProductLattice", "BitpackedStateVector", "RAGMetadataIngressEncoder", "build_label_attention_mask", "build_level_attention_mask",
    "bitpack_states", "unpack_states",
] + list(_LAZY)))
