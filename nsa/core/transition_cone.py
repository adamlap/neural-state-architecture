"""Algebraic legal-transition cones and exact product projection.

A transition cone specifies, independently for each state coordinate, whether a
legal transition may increase, decrease, or must preserve that coordinate in
its domain-specific partial order. The projection operator then maps an
arbitrary candidate into the closest legal state under the product algebra:
join for increasing coordinates, meet for decreasing coordinates, and the
source value for immutable coordinates.

This remains deterministic state algebra. It is not a claim that transformer
hidden states are intrinsically constrained until a neural adapter consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from nsa.core.heterogeneous_algebra import AlgebraDomain, HeterogeneousState

T = TypeVar("T")


class TransitionDirection(str, Enum):
    """Allowed direction of motion in one domain's partial order."""

    INCREASE = "increase"
    DECREASE = "decrease"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class TransitionCone(Generic[T]):
    """Coordinate-wise legal transition cone over a heterogeneous product."""

    directions: tuple[TransitionDirection, ...]

    def _validate_arity(self, state: HeterogeneousState[T]) -> None:
        if len(self.directions) != len(state.values):
            raise ValueError("transition cone arity must match state arity")

    def allows(self, source: HeterogeneousState[T], target: HeterogeneousState[T]) -> bool:
        """Return whether target lies inside the cone rooted at source."""
        source._compatible(target)
        self._validate_arity(source)
        for direction, left, right, domain in zip(
            self.directions, source.values, target.values, source.domains
        ):
            if direction is TransitionDirection.INCREASE:
                if domain.join(left, right) != right:
                    return False
            elif direction is TransitionDirection.DECREASE:
                if domain.meet(left, right) != right:
                    return False
            elif direction is TransitionDirection.UNCHANGED:
                if left != right:
                    return False
            else:  # pragma: no cover - Enum makes this unreachable
                raise ValueError(f"unsupported transition direction: {direction}")
        return True

    def project(self, source: HeterogeneousState[T], candidate: HeterogeneousState[T]) -> HeterogeneousState[T]:
        """Project candidate exactly onto the legal cone rooted at source."""
        source._compatible(candidate)
        self._validate_arity(source)
        values = []
        for direction, left, right, domain in zip(
            self.directions, source.values, candidate.values, source.domains
        ):
            if direction is TransitionDirection.INCREASE:
                values.append(domain.join(left, right))
            elif direction is TransitionDirection.DECREASE:
                values.append(domain.meet(left, right))
            elif direction is TransitionDirection.UNCHANGED:
                values.append(left)
            else:  # pragma: no cover
                raise ValueError(f"unsupported transition direction: {direction}")
        return HeterogeneousState(tuple(values), source.domains)

    def is_projected(self, source: HeterogeneousState[T], candidate: HeterogeneousState[T]) -> bool:
        """Check the exact projection fixed-point property."""
        return self.project(source, candidate) == candidate
