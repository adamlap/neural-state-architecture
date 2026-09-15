"""Continuous Cognitive Engine public package.

CCE owns durable machine-state lifecycle and continuous scheduling. Heavy
model-specific substrate integrations are optional and loaded from explicit
submodules so ``import nsa`` remains lightweight.
"""

from .engine import CCEStatus, ContinuousCognitiveEngine
from .events import CognitiveEvent, EventKind
from .lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore
from .trajectory import CognitiveTrajectory, TrajectoryRecord
from .transaction import CognitiveTransaction, CognitiveTransactionEngine

__all__ = [
    "CCEStatus",
    "ContinuousCognitiveEngine",
    "CognitiveEvent",
    "EventKind",
    "CheckpointEnvelope",
    "CognitiveInputEvent",
    "CognitiveInputQueue",
    "StateCheckpointStore",
    "CognitiveTrajectory",
    "TrajectoryRecord",
    "CognitiveTransaction",
    "CognitiveTransactionEngine",
]
