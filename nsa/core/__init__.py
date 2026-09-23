"""Canonical NSA typed state core.

State, transition and capability primitives are lightweight. The tensor-backed
immutable safety kernel is exposed lazily so canonical CCE does not require
PyTorch merely to import its control plane.
"""
from nsa.core.heterogeneous_algebra import (
    BooleanDomain, CapabilityDomain, ConstraintSetDomain, EnumDomain, HeterogeneousState,
    NumericRangeDomain, ProbabilityInterval, ProbabilityIntervalDomain, TemporalWindow, TemporalWindowDomain,
)
from nsa.core.state import CanonicalState, GoalState, HardState, ProvenanceState, SemanticState, SoftState, StateKind, StateTransition
from nsa.core.substrate_governance import GovernedSubstrateGovernor, SubstrateState
from nsa.core.transition import TransitionProposal, TransitionReceipt, TransitionValidator, state_digest
from nsa.core.transition_cone import TransitionCone, TransitionDirection
from nsa.core.capabilities import CapabilityAuthority, CapabilityToken, TrustThermodynamicsVector, TrustTier
from nsa.core.capability_constraints import CapabilityConstraintEvaluator, ConstraintContext, ConstraintDecision
from nsa.core.state_codec import SCHEMA_VERSION, decode_state, dumps_state, encode_state, round_trip

__all__ = [
    "BooleanDomain", "CapabilityDomain", "ConstraintSetDomain", "EnumDomain", "HeterogeneousState", "NumericRangeDomain",
    "ProbabilityInterval", "ProbabilityIntervalDomain", "TemporalWindow", "TemporalWindowDomain", "TransitionCone",
    "TransitionDirection", "CanonicalState", "SubstrateState", "GovernedSubstrateGovernor", "GoalState", "HardState", "ProvenanceState", "SemanticState", "SoftState",
    "StateKind", "StateTransition", "TransitionProposal", "TransitionReceipt", "TransitionValidator", "state_digest",
    "CapabilityAuthority", "CapabilityToken", "TrustThermodynamicsVector", "TrustTier",
    "CapabilityConstraintEvaluator", "ConstraintContext", "ConstraintDecision",
    "SCHEMA_VERSION", "decode_state", "dumps_state", "encode_state", "round_trip",
    "ImmutableSafetyKernel", "InvariantResult", "KernelEvaluationResult", "KernelVerdict",
]

_LAZY = {
    "ImmutableSafetyKernel": ("nsa.core.safety_kernel", "ImmutableSafetyKernel"),
    "InvariantResult": ("nsa.core.safety_kernel", "InvariantResult"),
    "KernelEvaluationResult": ("nsa.core.safety_kernel", "KernelEvaluationResult"),
    "KernelVerdict": ("nsa.core.safety_kernel", "KernelVerdict"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(name)
    import importlib
    module = importlib.import_module(target[0])
    value = getattr(module, target[1])
    globals()[name] = value
    return value
