"""Cognition package with a lightweight canonical control plane.

Protocols and deliberation are always importable. The historical tensor-based
substrate is loaded lazily so CCE can operate without PyTorch.
"""
from .interfaces import ActionCandidate, ActionSelector, BeliefUpdater, InformationGainModel, PredictionError, Predictor, Prediction
from .deliberation import DeliberationDecision, InformationNeed, InformationSeekingPlanner, UncertaintyDrivenDeliberator
from .model import CognitiveContext, CognitiveModel, CognitiveProposal, InformationNeedProposal
from .tools import ToolRegistry, ToolSpec
__all__ = [
    "ActionCandidate", "ActionSelector", "BeliefUpdater", "InformationGainModel", "Prediction", "PredictionError", "Predictor",
    "DeliberationDecision", "InformationNeed", "InformationSeekingPlanner", "UncertaintyDrivenDeliberator",
    "LatentCognitiveField", "LatentFieldConfig", "LatentThoughtVector", "CounterfactualSimulator", "CounterfactualBranch", "CounterfactualEvaluation", "SystemOneDecisionEngine", "TypedDecisionSchema", "CalibratedDecision", "CognitiveContext", "CognitiveModel", "CognitiveProposal", "InformationNeedProposal", "ToolRegistry", "ToolSpec",
    "CognitiveMetrics", "CognitiveState", "CognitiveSubstrate", "CognitiveSwitches", "IntegrationGraph",
    "PredictionState", "SelfModelState", "WorkspaceCandidate", "WorkspaceState",
]

_LAZY = {
    **{
        name: ("nsa.cognition.substrate", name)
        for name in ("CognitiveMetrics", "CognitiveState", "CognitiveSubstrate", "CognitiveSwitches",
                     "IntegrationGraph", "PredictionState", "SelfModelState", "WorkspaceCandidate", "WorkspaceState")
    },
    **{
        name: ("nsa.cognition.latent_field", name)
        for name in ("LatentCognitiveField", "LatentFieldConfig", "LatentThoughtVector")
    },
    **{
        name: ("nsa.cognition.counterfactual", name)
        for name in ("CounterfactualSimulator", "CounterfactualBranch", "CounterfactualEvaluation")
    },
    **{
        name: ("nsa.cognition.system_one", name)
        for name in ("SystemOneDecisionEngine", "TypedDecisionSchema", "CalibratedDecision")
    },
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None: raise AttributeError(name)
    import importlib
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
