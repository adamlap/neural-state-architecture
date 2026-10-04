"""Optional strategic control layer for NSA."""
from .model import ScenarioOutcome, StrategyCandidate, StrategyDistribution, StrategicField
from .engine import StrategicScenarioEngine
from .config import StrategicConfig
from .modulation import StrategicAttentionModulator, configured_modulator
from .controller import StrategicController, StrategicEvaluation
from .system_one import SystemOneStrategyProvider, StrategyDecision
from .torch_adapter import StrategicAttentionBias
from .feedback import StrategicFeedback, StrategicFeedbackUpdater
from .update_policy import StrategicUpdatePolicy

__all__ = [
    "ScenarioOutcome", "StrategyCandidate", "StrategyDistribution", "StrategicField",
    "StrategicScenarioEngine", "StrategicConfig", "StrategicAttentionModulator",
    "configured_modulator", "StrategicController", "StrategicEvaluation",
    "SystemOneStrategyProvider", "StrategyDecision", "StrategicAttentionBias",
    "StrategicFeedback", "StrategicFeedbackUpdater", "StrategicUpdatePolicy",
]
