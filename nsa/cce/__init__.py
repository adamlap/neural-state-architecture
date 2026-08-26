"""Continuous Cognitive Engine lifecycle primitives.

The CCE package owns durable machine-state lifecycle concerns; model-specific
cognitive engines remain in ``nsa.runtime`` while they are consolidated.
"""

from .lifecycle import CheckpointEnvelope, CognitiveInputEvent, CognitiveInputQueue, StateCheckpointStore

__all__ = ["CheckpointEnvelope", "CognitiveInputEvent", "CognitiveInputQueue", "StateCheckpointStore"]
