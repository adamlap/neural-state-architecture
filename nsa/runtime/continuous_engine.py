"""Compatibility shim for the canonical public CCE scheduler.

New code must import from ``nsa.cce.engine``. This module remains temporarily
for backwards compatibility with existing research integrations.
"""
from nsa.cce.engine import CCEStatus, ContinuousCognitiveEngine

__all__ = ["CCEStatus", "ContinuousCognitiveEngine"]
