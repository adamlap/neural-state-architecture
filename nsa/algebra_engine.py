"""General heterogeneous state algebra for NSA.

The existing ``nsa.algebra`` module contains the original security lattice.
This module provides the next abstraction layer without breaking that API:
independent state dimensions compose into a product algebra, while each
dimension retains its own join, meet, and transition semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Generic, Mapping, Protocol, TypeVar


class Algebra(Protocol):
    """Minimal algebra contract for a typed NSA state dimension."""

    def join(self, other: object) -> object: ...
    def meet(self, other: object) -> object: ...
    def leq(self, other: object) -> bool: ...


T = TypeVar("T")


@dataclass(frozen=True)
class OrderedValue(Generic[T]):
    """Finite ordered value with lattice operations induced by its rank."""

    value: T
    rank: int

    def join(self, other: "OrderedValue[T]") -> "OrderedValue[T]":
        return self if self.rank >= other.rank else other

    def meet(self, other: "OrderedValue[T]") -> "OrderedValue[T]":
        return self if self.rank <= other.rank else other

    def leq(self, other: "OrderedValue[T]") -> bool:
        return self.rank <= other.rank


@dataclass(frozen=True)
class BooleanSetAlgebra:
    """Power-set lattice: join = union, meet = intersection."""

    values: frozenset[str] = frozenset()

    def join(self, other: "BooleanSetAlgebra") -> "BooleanSetAlgebra":
        return BooleanSetAlgebra(self.values | other.values)

    def meet(self, other: "BooleanSetAlgebra") -> "BooleanSetAlgebra":
        return BooleanSetAlgebra(self.values & other.values)

    def leq(self, other: "BooleanSetAlgebra") -> bool:
        return self.values <= other.values


@dataclass(frozen=True)
class ProbabilityAlgebra:
    """Probability-like confidence state with bounded scalar semantics.

    Join is conservative (minimum confidence), meet is optimistic (maximum
    confidence). This is deliberately distinct from an ordinary numeric max
    lattice so the semantics are explicit.
    """

    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def join(self, other: "ProbabilityAlgebra") -> "ProbabilityAlgebra":
        return ProbabilityAlgebra(min(self.confidence, other.confidence))

    def meet(self, other: "ProbabilityAlgebra") -> "ProbabilityAlgebra":
        return ProbabilityAlgebra(max(self.confidence, other.confidence))

    def leq(self, other: "ProbabilityAlgebra") -> bool:
        return self.confidence >= other.confidence


@dataclass(frozen=True)
class ScalarRiskAlgebra:
    """Bounded risk domain: join is worst-case, meet is best-case."""

    risk: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")

    def join(self, other: "ScalarRiskAlgebra") -> "ScalarRiskAlgebra":
        return ScalarRiskAlgebra(max(self.risk, other.risk))

    def meet(self, other: "ScalarRiskAlgebra") -> "ScalarRiskAlgebra":
        return ScalarRiskAlgebra(min(self.risk, other.risk))

    def leq(self, other: "ScalarRiskAlgebra") -> bool:
        return self.risk <= other.risk


@dataclass(frozen=True)
class ProductAlgebra:
    """Heterogeneous product lattice over named independent dimensions.

    A product join/meet is legal only when both operands have the same typed
    dimensions. This prevents accidental scalarization of semantically
    different state domains.
    """

    dimensions: Mapping[str, Algebra]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", dict(self.dimensions))

    def join(self, other: "ProductAlgebra") -> "ProductAlgebra":
        self._check_schema(other)
        return ProductAlgebra({
            key: self.dimensions[key].join(other.dimensions[key])
            for key in self.dimensions
        })

    def meet(self, other: "ProductAlgebra") -> "ProductAlgebra":
        self._check_schema(other)
        return ProductAlgebra({
            key: self.dimensions[key].meet(other.dimensions[key])
            for key in self.dimensions
        })

    def leq(self, other: "ProductAlgebra") -> bool:
        self._check_schema(other)
        return all(
            self.dimensions[key].leq(other.dimensions[key])
            for key in self.dimensions
        )

    def _check_schema(self, other: "ProductAlgebra") -> None:
        if self.dimensions.keys() != other.dimensions.keys():
            raise TypeError("product states must have identical dimension schemas")

    def get(self, name: str) -> Algebra:
        return self.dimensions[name]

    def names(self) -> tuple[str, ...]:
        return tuple(self.dimensions.keys())


class TransitionDecision(IntEnum):
    REJECT = 0
    ACCEPT = 1


def legal_product_transition(
    source: ProductAlgebra,
    target: ProductAlgebra,
) -> TransitionDecision:
    """Return ACCEPT iff every component transition is algebraically legal."""
    return (
        TransitionDecision.ACCEPT
        if source.leq(target)
        else TransitionDecision.REJECT
    )


__all__ = [
    "Algebra",
    "OrderedValue",
    "BooleanSetAlgebra",
    "ProbabilityAlgebra",
    "ScalarRiskAlgebra",
    "ProductAlgebra",
    "TransitionDecision",
    "legal_product_transition",
]
