"""Compatibility shim for the canonical public CCE substrate adapter.

New code should import these symbols from ``nsa.cce.substrate``. Keeping the
old import path avoids breaking existing experiments while removing duplicated
scheduler and transition logic.
"""
from nsa.cce.substrate import (
    CandidateAction,
    CandidateProvider,
    ContinuousSubstrateRuntime,
    SubstrateTransition,
    SubstrateTransitionConfig,
)

__all__ = [
    "CandidateAction",
    "CandidateProvider",
    "ContinuousSubstrateRuntime",
    "SubstrateTransition",
    "SubstrateTransitionConfig",
]
