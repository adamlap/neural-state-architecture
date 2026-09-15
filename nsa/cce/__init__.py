"""Continuous Cognitive Engine public package."""
from .engine import CCEStatus, ContinuousCognitiveEngine
from .events import CognitiveEvent, EventKind
from .lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore
from .trajectory import CognitiveTrajectory, TrajectoryRecord
from .transaction import CognitiveTransaction, CognitiveTransactionEngine
from .canonical_runtime import CanonicalCCERuntime, TickInput
from .loop import CognitiveCycle, CognitiveLoop, default_prediction_error
from .persistence import TrajectoryJournal, record_to_dict
from .safety_adapter import ImmutableKernelGate, SafetyDecision

__all__ = [
    "CCEStatus", "ContinuousCognitiveEngine", "CognitiveEvent", "EventKind",
    "CheckpointEnvelope", "CognitiveInputEvent", "CognitiveInputQueue", "StateCheckpointStore",
    "CognitiveTrajectory", "TrajectoryRecord", "CognitiveTransaction", "CognitiveTransactionEngine",
    "CanonicalCCERuntime", "TickInput", "CognitiveCycle", "CognitiveLoop",
    "default_prediction_error", "TrajectoryJournal", "record_to_dict", "ImmutableKernelGate", "SafetyDecision",
]
