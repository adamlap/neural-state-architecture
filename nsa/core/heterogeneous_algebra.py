"""Heterogeneous algebra primitives for canonical NSA state.

Phase 12 makes state algebra explicit at the domain boundary. Each coordinate
owns its own join/meet semantics; the product combines coordinates without
reducing heterogeneous state to a scalar score.

This module is deterministic state infrastructure. It does not modify model
weights or hidden activations and therefore is not, by itself, intrinsic neural
security.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import FrozenSet, Generic, Protocol, TypeVar

T = TypeVar("T")


class AlgebraDomain(Protocol[T]):
    def join(self, left: T, right: T) -> T: ...
    def meet(self, left: T, right: T) -> T: ...
    def validate(self, value: T) -> None: ...


class BooleanDomain:
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


class ConstraintSetDomain:
    """Constraint lattice ordered by set inclusion."""

    def join(self, left: FrozenSet[str], right: FrozenSet[str]) -> FrozenSet[str]:
        return left | right

    def meet(self, left: FrozenSet[str], right: FrozenSet[str]) -> FrozenSet[str]:
        return left & right

    def validate(self, value: FrozenSet[str]) -> None:
        if not isinstance(value, frozenset) or not all(isinstance(v, str) for v in value):
            raise TypeError("constraint domain requires FrozenSet[str]")


class NumericRangeDomain:
    """Closed scalar interval with max/min lattice operations."""

    def __init__(self, minimum: float = 0.0, maximum: float = 1.0) -> None:
        if not isfinite(minimum) or not isfinite(maximum) or minimum > maximum:
            raise ValueError("invalid numeric range")
        self.minimum = float(minimum)
        self.maximum = float(maximum)

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


@dataclass(frozen=True)
class ProbabilityInterval:
    """Closed probability interval; lower > upper is the explicit bottom element."""

    lower: float
    upper: float

    @property
    def is_empty(self) -> bool:
        return self.lower > self.upper


class ProbabilityIntervalDomain:
    """Lattice of closed probability sets ordered by inclusion."""

    def validate(self, value: ProbabilityInterval) -> None:
        if not isinstance(value, ProbabilityInterval):
            raise TypeError("probability domain requires ProbabilityInterval")
        if not (0.0 <= value.lower <= 1.0 and 0.0 <= value.upper <= 1.0) and not value.is_empty:
            raise ValueError("probability bounds must lie in [0, 1]")
        if value.is_empty and not (value.lower == 1.0 and value.upper == 0.0):
            raise ValueError("empty probability interval must be ProbabilityInterval(1, 0)")

    def join(self, left: ProbabilityInterval, right: ProbabilityInterval) -> ProbabilityInterval:
        self.validate(left)
        self.validate(right)
        if left.is_empty:
            return right
        if right.is_empty:
            return left
        return ProbabilityInterval(min(left.lower, right.lower), max(left.upper, right.upper))

    def meet(self, left: ProbabilityInterval, right: ProbabilityInterval) -> ProbabilityInterval:
        self.validate(left)
        self.validate(right)
        if left.is_empty or right.is_empty:
            return ProbabilityInterval(1.0, 0.0)
        lower = max(left.lower, right.lower)
        upper = min(left.upper, right.upper)
        return ProbabilityInterval(lower, upper) if lower <= upper else ProbabilityInterval(1.0, 0.0)


@dataclass(frozen=True)
class TemporalWindow:
    """Closed temporal window in monotonic numeric ticks; reversed is bottom."""

    start: float
    end: float

    @property
    def is_empty(self) -> bool:
        return self.start > self.end


class TemporalWindowDomain:
    """Interval lattice over monotonic time windows, including explicit bottom."""

    def validate(self, value: TemporalWindow) -> None:
        if not isinstance(value, TemporalWindow):
            raise TypeError("temporal domain requires TemporalWindow")
        if not isfinite(value.start) or not isfinite(value.end):
            raise ValueError("temporal bounds must be finite")

    def join(self, left: TemporalWindow, right: TemporalWindow) -> TemporalWindow:
        self.validate(left)
        self.validate(right)
        if left.is_empty:
            return right
        if right.is_empty:
            return left
        return TemporalWindow(min(left.start, right.start), max(left.end, right.end))

    def meet(self, left: TemporalWindow, right: TemporalWindow) -> TemporalWindow:
        self.validate(left)
        self.validate(right)
        if left.is_empty or right.is_empty:
            return TemporalWindow(1.0, 0.0)
        lower = max(left.start, right.start)
        upper = min(left.end, right.end)
        return TemporalWindow(lower, upper) if lower <= upper else TemporalWindow(1.0, 0.0)


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


@dataclass(frozen=True, eq=False)
class HeterogeneousState(Generic[T]):
    """Typed product element whose coordinates use independent domains."""

    values: tuple[T, ...]
    domains: tuple[AlgebraDomain[T], ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(self.domains):
            raise ValueError("values and domains must have identical arity")
        for value, domain in zip(self.values, self.domains):
            domain.validate(value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HeterogeneousState):
            return False
        if self.values != other.values:
            return False
        try:
            self._compatible(other)
            return True
        except ValueError:
            return False

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
        if isinstance(domain, BooleanDomain):
            return (BooleanDomain,)
        if isinstance(domain, CapabilityDomain):
            return (CapabilityDomain,)
        if isinstance(domain, ConstraintSetDomain):
            return (ConstraintSetDomain,)
        if isinstance(domain, NumericRangeDomain):
            return (NumericRangeDomain, domain.minimum, domain.maximum)
        if isinstance(domain, ProbabilityIntervalDomain):
            return (ProbabilityIntervalDomain,)
        if isinstance(domain, TemporalWindowDomain):
            return (TemporalWindowDomain,)
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
