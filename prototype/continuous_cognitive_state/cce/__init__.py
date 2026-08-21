"""Continuous Cognitive Engine (CCE) integration layer.

CCE is intentionally isolated from the core NSA implementation. It consumes
NSA's public algebra/policy primitives without modifying them.
"""

from .governor import CCEGovernor, CCEPolicy
from .engine import ContinuousCognitiveEngine

__all__ = ["CCEGovernor", "CCEPolicy", "ContinuousCognitiveEngine"]
