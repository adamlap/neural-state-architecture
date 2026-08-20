"""Composable partial updates for the canonical NSA typed-state protocol.

A patch is a proposal, not an authority boundary. Applying a patch still goes
through ``CanonicalTypedActivation.runtime_commit`` so hard state cannot be
changed merely by constructing a patch object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from nsa.core.typed_activation import CANONICAL_FIELDS, CanonicalTypedActivation


class StatePatchConflict(ValueError):
    """Raised when two partial state updates disagree on the same field."""


@dataclass(frozen=True)
class CanonicalStatePatch:
    """A sparse, immutable update over canonical state fields."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        unknown = set(self.values).difference(CANONICAL_FIELDS)
        if unknown:
            raise KeyError(f"Unknown canonical state field(s): {sorted(unknown)}")

    @classmethod
    def empty(cls) -> "CanonicalStatePatch":
        return cls({})

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        """Compare scalar and tensor state values without ambiguous truth tests."""
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                return False
            return torch.equal(left, right)
        return bool(left == right)

    def compose(self, other: "CanonicalStatePatch") -> "CanonicalStatePatch":
        """Compose two patches, rejecting ambiguous writes."""
        overlap = set(self.values).intersection(other.values)
        conflicts = [
            field
            for field in overlap
            if not self._values_equal(self.values[field], other.values[field])
        ]
        if conflicts:
            raise StatePatchConflict(
                f"conflicting updates for canonical field(s): {sorted(conflicts)}"
            )
        merged = dict(self.values)
        merged.update(other.values)
        return CanonicalStatePatch(merged)

    def apply_runtime(self, activation: CanonicalTypedActivation) -> CanonicalTypedActivation:
        """Apply every field through the trusted runtime commit path."""
        result = activation
        for field, value in self.values.items():
            result = result.runtime_commit(field, value)
        return result

    def model_proposals(self, activation: CanonicalTypedActivation) -> tuple[dict[str, Any], ...]:
        """Return non-mutating proposals; hard/runtime-owned fields are rejected."""
        return tuple(
            activation.model_proposal(field, value) for field, value in self.values.items()
        )


__all__ = ["CanonicalStatePatch", "StatePatchConflict"]
