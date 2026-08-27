"""Continuous Cognitive Engine public package.

CCE owns durable machine-state lifecycle and continuous scheduling. Heavy
model-specific substrate integrations are optional and loaded from explicit
submodules so ``import nsa`` remains lightweight.
"""

from .engine import CCEStatus, ContinuousCognitiveEngine
from .lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore

__all__ = [
    "CCEStatus",
    "ContinuousCognitiveEngine",
    "CheckpointEnvelope",
    "CognitiveInputEvent",
    "CognitiveInputQueue",
    "StateCheckpointStore",
]
