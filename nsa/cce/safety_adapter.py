"""Canonical CCE adapter for the immutable NSA safety kernel.

The adapter is intentionally callback-based: projects choose how their neural
Omega representation is constructed, while the safety kernel remains a
non-neural authority. Kernel approval is a gate only; it never commits state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nsa.cognition.interfaces import ActionCandidate
from nsa.core.omega import UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel, KernelVerdict
from nsa.core.state import CanonicalState

OmegaBuilder = Callable[[CanonicalState], UnifiedCognitiveState]
ActionTensorBuilder = Callable[[ActionCandidate, UnifiedCognitiveState], object]


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    verdict: KernelVerdict


class ImmutableKernelGate:
    """Turn the six-layer immutable safety kernel into a CCE GateHook."""
    def __init__(self, kernel: ImmutableSafetyKernel, omega_builder: OmegaBuilder,
                 action_tensor_builder: ActionTensorBuilder) -> None:
        self.kernel = kernel
        self.omega_builder = omega_builder
        self.action_tensor_builder = action_tensor_builder

    def evaluate(self, state: CanonicalState, action: ActionCandidate) -> SafetyDecision:
        omega = self.omega_builder(state)
        tensor = self.action_tensor_builder(action, omega)
        clearance = min(1.0, max(0.0, 1.0 - action.risk))
        result = self.kernel.evaluate_transition(
            omega_current=omega,
            action_id=action.action_id,
            action_clearance=clearance,
            user_clearance_limit=clearance,
            predicted_self_error=state.soft.uncertainty,
            proposed_action_risk=action.risk,
            is_verification_action=bool(action.payload.get("verification", False)) if isinstance(action.payload, dict) else False,
            target_action_risk=max(action.risk, 1e-9),
            supplied_capability=None,
            valid_capability_supplied=False,
        )
        if result.verdict == KernelVerdict.COMMIT:
            return SafetyDecision(True, "immutable safety kernel approved transition", result.verdict)
        details = "; ".join(inv.details for inv in result.invariant_results if not inv.passed)
        return SafetyDecision(False, details or f"immutable safety kernel verdict: {result.verdict.value}", result.verdict)

    def __call__(self, state: CanonicalState, action: ActionCandidate):
        decision = self.evaluate(state, action)
        return decision.allowed, decision.reason


__all__ = ["ImmutableKernelGate", "SafetyDecision"]
