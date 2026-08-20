"""Trusted transition engine for canonical and heterogeneous NSA state.

The engine separates model proposals from authoritative application. A model
may propose a target state, but policy decides whether that transition is legal.
For heterogeneous state, a transition cone supplies the per-coordinate algebraic
invariant and exact projection provides a deterministic safe candidate.

This is state-level enforcement at the NSA runtime boundary. It does not claim
to modify transformer weights or hidden activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from nsa.core.heterogeneous_algebra import HeterogeneousState
from nsa.core.state import CanonicalState, HardState, StateTransition
from nsa.core.transition_cone import TransitionCone


@dataclass(frozen=True)
class TransitionPolicy:
    """Policy controlling legal hard-state transitions."""

    validator: Optional[Callable[[HardState, HardState], bool]] = None
    max_license_step: Optional[int] = None
    allow_authorization_additions: bool = False

    def validate(self, source: HardState, target: HardState) -> None:
        if self.max_license_step is not None:
            if target.license_tier - source.license_tier > self.max_license_step:
                raise PermissionError("license-tier transition exceeds policy")

        if not self.allow_authorization_additions:
            added = target.authorizations - source.authorizations
            if added:
                raise PermissionError(
                    f"policy forbids authorization additions: {sorted(added)}"
                )

        if self.validator is not None and not self.validator(source, target):
            raise PermissionError("custom transition policy rejected target state")


@dataclass(frozen=True)
class TransitionResult:
    """Result of evaluating a proposed transition."""

    accepted: bool
    state: CanonicalState
    reason: Optional[str] = None


@dataclass(frozen=True)
class HeterogeneousTransitionResult:
    """Result of applying a heterogeneous transition cone."""

    accepted: bool
    state: HeterogeneousState
    projected: bool = False
    reason: Optional[str] = None


class TransitionEngine:
    """Evaluate and apply explicit canonical or heterogeneous state transitions."""

    def __init__(self, policy: Optional[TransitionPolicy] = None) -> None:
        self.policy = policy or TransitionPolicy()

    def propose(self, state: CanonicalState, target: HardState) -> StateTransition:
        """Create an unauthorised proposal from the current hard state."""
        return StateTransition(source=state.hard, target=target)

    def authorize(
        self,
        transition: StateTransition,
        capability_id: str,
        reason: Optional[str] = None,
    ) -> StateTransition:
        """Validate policy before attaching an external capability."""
        self.policy.validate(transition.source, transition.target)
        return transition.authorize(capability_id, reason)

    def apply(
        self,
        state: CanonicalState,
        transition: StateTransition,
    ) -> TransitionResult:
        """Apply an authorized transition, returning a structured result."""
        try:
            self.policy.validate(transition.source, transition.target)
            new_state = state.transition(transition)
        except (PermissionError, ValueError) as exc:
            return TransitionResult(accepted=False, state=state, reason=str(exc))
        return TransitionResult(accepted=True, state=new_state)

    def apply_heterogeneous(
        self,
        source: HeterogeneousState,
        candidate: HeterogeneousState,
        cone: TransitionCone,
        *,
        project_illegal: bool = True,
    ) -> HeterogeneousTransitionResult:
        """Validate a heterogeneous candidate against a typed transition cone.

        If ``project_illegal`` is true, an invalid candidate is deterministically
        projected into the legal cone. Otherwise the source is retained and the
        result is rejected. No scalar safety score is introduced: every
        coordinate is governed by its own algebraic domain.
        """
        try:
            if cone.allows(source, candidate):
                return HeterogeneousTransitionResult(True, candidate)
            if project_illegal:
                return HeterogeneousTransitionResult(
                    True, cone.project(source, candidate), projected=True
                )
            return HeterogeneousTransitionResult(
                False, source, reason="candidate violates transition cone"
            )
        except (TypeError, ValueError) as exc:
            return HeterogeneousTransitionResult(False, source, reason=str(exc))


__all__ = [
    "HeterogeneousTransitionResult",
    "TransitionEngine",
    "TransitionPolicy",
    "TransitionResult",
]
