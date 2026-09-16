"""Cognition package with a lightweight canonical control plane.

Protocols and deliberation are always importable. The historical tensor-based
substrate is loaded lazily so CCE can operate without PyTorch.
"""
from .interfaces import ActionCandidate, ActionSelector, BeliefUpdater, InformationGainModel, PredictionError, Predictor, Prediction
from .deliberation import DeliberationDecision, InformationNeed, InformationSeekingPlanner, UncertaintyDrivenDeliberator

__all__ = [
    "ActionCandidate", "ActionSelector", "BeliefUpdater", "InformationGainModel", "Prediction", "PredictionError", "Predictor",
    "DeliberationDecision", "InformationNeed", "InformationSeekingPlanner", "UncertaintyDrivenDeliberator",
    "CognitiveMetrics", "CognitiveState", "CognitiveSubstrate", "CognitiveSwitches", "IntegrationGraph",
    "PredictionState", "SelfModelState", "WorkspaceCandidate", "WorkspaceState",
]

_LAZY = {
    name: ("nsa.cognition.substrate", name)
    for name in ("CognitiveMetrics", "CognitiveState", "CognitiveSubstrate", "CognitiveSwitches",
                 "IntegrationGraph", "PredictionState", "SelfModelState", "WorkspaceCandidate", "WorkspaceState")
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None: raise AttributeError(name)
    import importlib
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
