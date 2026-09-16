"""Continuous Cognitive Engine public package."""
from .engine import CCEStatus, ContinuousCognitiveEngine
from .events import CognitiveEvent, EventKind
from .lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore
from .trajectory import CognitiveTrajectory, TrajectoryRecord
from .transaction import CognitiveTransaction, CognitiveTransactionEngine
from .async_transaction import AsyncCognitiveTransactionEngine, AsyncExecutionHook
from .canonical_runtime import CanonicalCCERuntime, TickInput
from .loop import CognitiveCycle, CognitiveLoop, default_prediction_error
from .orchestrator import CognitiveCycleResult, CognitiveOrchestrator
from .async_orchestrator import AsyncCognitiveCycleResult, AsyncCognitiveOrchestrator, AsyncEffect
from .persistence import TrajectoryJournal, record_to_dict
from .effects import CallableEffect, EffectReceipt, TwoPhaseExecutor

__all__ = [
    "CCEStatus", "ContinuousCognitiveEngine", "CognitiveEvent", "EventKind",
    "CheckpointEnvelope", "CognitiveInputEvent", "CognitiveInputQueue", "StateCheckpointStore",
    "CognitiveTrajectory", "TrajectoryRecord", "CognitiveTransaction", "CognitiveTransactionEngine",
    "AsyncCognitiveTransactionEngine", "AsyncExecutionHook",
    "CanonicalCCERuntime", "TickInput", "CognitiveCycle", "CognitiveLoop", "default_prediction_error",
    "CognitiveCycleResult", "CognitiveOrchestrator", "AsyncCognitiveCycleResult", "AsyncCognitiveOrchestrator", "AsyncEffect",
    "TrajectoryJournal", "record_to_dict", "CallableEffect", "EffectReceipt", "TwoPhaseExecutor",
    "ImmutableKernelGate", "SafetyDecision", "CanonicalOmegaAdapter", "OmegaDimensions",
    "SixLayerCanonicalAdapter", "SubstrateProposal", "OmegaFeedback", "OmegaFeedbackAdapter",
]

_LAZY = {
    "ImmutableKernelGate": ("nsa.cce.safety_adapter", "ImmutableKernelGate"),
    "SafetyDecision": ("nsa.cce.safety_adapter", "SafetyDecision"),
    "CanonicalOmegaAdapter": ("nsa.cce.omega_adapter", "CanonicalOmegaAdapter"),
    "OmegaDimensions": ("nsa.cce.omega_adapter", "OmegaDimensions"),
    "SixLayerCanonicalAdapter": ("nsa.cce.substrate_adapter", "SixLayerCanonicalAdapter"),
    "SubstrateProposal": ("nsa.cce.substrate_adapter", "SubstrateProposal"),
    "OmegaFeedback": ("nsa.cce.omega_feedback", "OmegaFeedback"),
    "OmegaFeedbackAdapter": ("nsa.cce.omega_feedback", "OmegaFeedbackAdapter"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None: raise AttributeError(name)
    import importlib
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
