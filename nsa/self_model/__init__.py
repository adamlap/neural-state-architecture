"""Predictive self-model and live trajectory APIs."""

from .core import (
    CapabilityMonitor,
    ConditionedPredictiveSelfModel,
    CounterfactualInternalSimulator,
    PredictiveSelfState,
    RegulationDecision,
    SelfRegulationController,
    SimulationResult,
)
from .trajectory import ActionFeatures, TrajectoryStep, build_action_features, trajectory_step

__all__ = [
    "CapabilityMonitor", "ConditionedPredictiveSelfModel", "CounterfactualInternalSimulator",
    "PredictiveSelfState", "RegulationDecision", "SelfRegulationController", "SimulationResult",
    "ActionFeatures", "TrajectoryStep", "build_action_features", "trajectory_step",
]
