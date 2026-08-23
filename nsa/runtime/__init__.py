"""
nsa.runtime
===========
Trusted Cognitive Runtime for NSA Autonomous Execution.
"""

from .cce_adapter import (
    ContinuousSubstrateRuntime,
    SubstrateTransition,
    SubstrateTransitionConfig,
)
from .cce_salience import AdaptiveSalienceGate, SalienceDecision, SalienceObservation
from .continuous_engine import CCEStatus, ContinuousCognitiveEngine
from .continuous_state_field import ContinuousFieldStatus, ContinuousStateField
from .engine import CognitiveRuntime, ExecutionContext
from .phantom_maintenance import MaintenanceResult, PhantomMaintenanceLoop, maintain
from .predictive_dynamics import PredictionMetrics, StatePredictor, prediction_metrics, train_predictor
from .typed_runtime import NSATypedRuntime, RuntimeGeneration

__all__ = [
    "CCEStatus",
    "ContinuousCognitiveEngine",
    "ContinuousFieldStatus",
    "ContinuousStateField",
    "ContinuousSubstrateRuntime",
    "CognitiveRuntime",
    "ExecutionContext",
    "NSATypedRuntime",
    "RuntimeGeneration",
    "SubstrateTransition",
    "SubstrateTransitionConfig",
    "PredictionMetrics",
    "StatePredictor",
    "prediction_metrics",
    "train_predictor",
    "AdaptiveSalienceGate",
    "SalienceDecision",
    "SalienceObservation",
    "MaintenanceResult",
    "PhantomMaintenanceLoop",
    "maintain",
]
