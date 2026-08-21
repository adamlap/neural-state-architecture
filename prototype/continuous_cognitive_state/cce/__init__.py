"""Continuous Cognitive Engine (CCE) integration layer.

CCE is isolated from the core NSA implementation and consumes NSA's public
algebra/policy primitives without modifying nsa/.
"""

from .action import ActionProposal, GovernanceDecision
from .engine import ContinuousCognitiveEngine, CCEConfig
from .governor import CCEGovernor, CCEPolicy
from .ollama import OllamaProposalGenerator, OllamaReasoner
from .state import CCEState

__all__ = [
    "ActionProposal", "GovernanceDecision", "ContinuousCognitiveEngine", "CCEConfig",
    "CCEGovernor", "CCEPolicy", "OllamaProposalGenerator", "OllamaReasoner", "CCEState",
]
