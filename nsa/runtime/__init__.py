"""
nsa.runtime
===========
Trusted Cognitive Runtime for NSA Autonomous Execution.
"""

from .cce_adapter import SubstrateTransition, SubstrateTransitionConfig
from .continuous_engine import CCEStatus, ContinuousCognitiveEngine
from .engine import CognitiveRuntime, ExecutionContext
from .typed_runtime import NSATypedRuntime, RuntimeGeneration

__all__ = [
    "CCEStatus",
    "ContinuousCognitiveEngine",
    "CognitiveRuntime",
    "ExecutionContext",
    "NSATypedRuntime",
    "RuntimeGeneration",
    "SubstrateTransition",
    "SubstrateTransitionConfig",
]