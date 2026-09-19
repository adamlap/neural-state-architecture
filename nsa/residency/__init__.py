"""Selective neural storage and residency control."""
from nsa.residency.types import MemoryTier, ResidencyState, NeuralRegion, ResidencyEvent, ResidencySnapshot
from nsa.residency.policy import ResidencyPolicy, ResidencyDecision
from nsa.residency.cache import ResidencyCache
from nsa.residency.predictor import ResidencyPredictor, HeuristicResidencyPredictor
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.instrumentation import instrument_decoder_layers
from nsa.residency.controller import ActiveResidencyController, PrefetchTask
from nsa.residency.cognitive import cognitive_state_features
from nsa.residency.learned import OnlineResidencyPredictor
from nsa.residency.trace import ResidencyTrace
from nsa.residency.accelerate_prefetch import AccelerateDiskPrefetcher

__all__ = [
    "MemoryTier", "ResidencyState", "NeuralRegion", "ResidencyEvent", "ResidencySnapshot",
    "ResidencyPolicy", "ResidencyDecision", "ResidencyCache", "ResidencyPredictor",
    "HeuristicResidencyPredictor", "NeuralResidencyManager", "instrument_decoder_layers",
    "ActiveResidencyController", "PrefetchTask", "cognitive_state_features",
    "OnlineResidencyPredictor", "ResidencyTrace", "AccelerateDiskPrefetcher",
]
