"""Heterogeneous algebra primitives for canonical NSA state.

Phase 12 starts by making the algebra explicit at the domain boundary. Each
state coordinate owns its own join/meet semantics; the product algebra then
combines coordinates without pretending that every domain is numeric.

This module is deliberately independent of model weights or hidden activations.
It is a deterministic mathematical substrate for future NSA adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import FrozenSet, Generic, Protocol, TypeVar


T = TypeVar("T")


class AlgebraDomain(Protocol[T]):
    """A bounded join/meet algebra for one typed state coordinate."""

    def join(self, left: T, right: T) -> T: ...
    def meet(self, left: T, right: T) -> T: ...
    def validate(self, value: T) -> None: ...


class BooleanDomain:
    """Boolean lattice: False <= True."""

    def join(self, left: bool, right: bool) -> bool:
        return left or right

    def meet(self, left: bool, right: bool) -> bool:
        return left and right

    def validate(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError("boolean domain requires bool")


class CapabilityDomain:
    """Capability lattice ordered by set inclusion."""

    def join(self, left: FrozenSet[str], right: FrozenSet[str]) -> FrozenSet[str]:
        return left | right

    def meet(self, left: FrozenSet[str], right: FrozenSet[str]) -> FrozenSet[str]:
        return left & right

    def validate(self, value: FrozenSet[str]) -> None:
        if not isinstance(value, frozenset) or not all(isinstance(v, str) for v in value):
            raise TypeError("capability domain requires FrozenSet[str]")


class NumericRangeDomain:
    """Closed numeric interval lattice ordered by increasing severity."""

    def __init__(self, minimum: float = 0.0, maximum: float = 1.0) -> None:
        if not isfinite(minimum) or not isfinite(maximum) or minimum > maximum:
            raise ValueError("invalid numeric range")
        self.minimum = minimum
        self.maximum = maximum

    def validate(self, value: float) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("numeric domain requires a real number")
        if not isfinite(float(value)) or not self.minimum <= float(value) <= self.maximum:
            raise ValueError(f"value must lie in [{self.minimum}, {self.maximum}]")

    def join(self, left: float, right: float) -> float:
        self.validate(left)
        self.validate(right)
        return max(float(left), float(right))

    def meet(self, left: float, right: float) -> float:
        self.validate(left)
        self.validate(right)
        return min(float(left), float(right))


class EnumDomain(Generic[T]):
    """Finite totally ordered enum domain."""

    def __init__(self, enum_type: type[T]) -> None:
        self.enum_type = enum_type
        if not issubclass(enum_type, Enum):
            raise TypeError("EnumDomain requires an Enum type")

    def validate(self, value: T) -> None:
        if not isinstance(value, self.enum_type):
            raise TypeError(f"expected {self.enum_type.__name__}")

    def join(self, left: T, right: T) -> T:
        self.validate(left)
        self.validate(right)
        return left if left.value >= right.value else right

    def meet(self, left: T, right: T) -> T:
        self.validate(left)
        self.validate(right)
        return left if left.value <= right.value else right


@dataclass(frozen=True)
class HeterogeneousState(Generic[T]):
    """Typed product element whose coordinates use independent domains."""

    values: tuple[T, ...]
    domains: tuple[AlgebraDomain[T], ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.domains):
            raise ValueError("values and domains must have identical arity")
        for value, domain in zip(self.values, self.domains):
            domain.validate(value)

    def join(self, other: "HeterogeneousState[T]") -> "HeterogeneousState[T]":
        self._compatible(other)
        return HeterogeneousState(
            tuple(domain.join(a, b) for a, b, domain in zip(self.values, other.values, self.domains)),
            self.domains,
        )

    def meet(self, other: "HeterogeneousState[T]") -> "HeterogeneousState[T]":
        self._compatible(other)
        return HeterogeneousState(
            tuple(domain.meet(a, b) for a, b, domain in zip(self.values, other.values, self.domains)),
            self.domains,
        )

    @staticmethod
    def _domain_signature(domain: AlgebraDomain[T]) -> tuple[object, ...]:
        """Return semantic domain configuration rather than object identity."""
        if isinstance(domain, BooleanDomain):
            return (BooleanDomain,)
        if isinstance(domain, CapabilityDomain):
            return (CapabilityDomain,)
        if isinstance(domain, NumericRangeDomain):
            return (NumericRangeDomain, domain.minimum, domain.maximum)
        if isinstance(domain, EnumDomain):
            return (EnumDomain, domain.enum_type)
        return (type(domain), repr(domain))

    def _compatible(self, other: "HeterogeneousState[T]") -> None:
        if len(self.values) != len(other.values):
            raise ValueError("states belong to incompatible product algebras")
        left_signature = tuple(self._domain_signature(domain) for domain in self.domains)
        right_signature = tuple(self._domain_signature(domain) for domain in other.domains)
        if left_signature != right_signature:
            raise ValueError("states belong to incompatible product algebras")

    def leq(self, other: "HeterogeneousState[T]") -> bool:
        """Coordinate-wise partial order: x <= y iff x ⊔ y = y."""
        self._compatible(other)
        return self.join(other).values == other.values
