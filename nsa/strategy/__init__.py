"""Strategic neural modulation primitives for NSA.

This package is an optional control layer: it augments canonical state and
does not replace NSA governance, cognition, memory, or authority boundaries.
"""
from .model import ScenarioOutcome, StrategyCandidate, StrategyDistribution, StrategicField
from .engine import StrategicScenarioEngine
from .modulation import StrategicAttentionModulator
from .controller import StrategicController

__all__ = ["ScenarioOutcome", "StrategyCandidate", "StrategyDistribution",
           "StrategicField", "StrategicScenarioEngine",
           "StrategicAttentionModulator", "StrategicController"]
