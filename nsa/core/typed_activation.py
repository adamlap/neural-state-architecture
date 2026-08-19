"""Canonical typed activation contract for NSA.

This module is the Phase 11 convergence layer.  It does not pretend that a
Python object is a security boundary: hard state remains trusted only when its
mutation is performed by the trusted runtime/kernel.  The purpose here is to
make state ownership, permissions, composition and serialization explicit so
future neural and runtime adapters share one representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import torch

from nsa.core.omega import UnifiedCognitiveState


CANONICAL_SCHEMA_VERSION = "1.0"


class StateDomain(str, Enum):
    """Canonical ownership classes for NSA state."""

    SEMANTIC = "semantic"
    SOFT = "soft"
    HARD = "hard"
    EPISTEMIC = "epistemic"
    PROVENANCE = "provenance"
    TEMPORAL = "temporal"
    GOAL = "goal"


class StatePermission(str, Enum):
    READ = "read"
    PROPOSE = "propose"
    COMMIT = "commit"


@dataclass(frozen=True)
class StateFieldSpec:
    """Static contract for one state component."""

    domain: StateDomain
    owner: str
    model_writable: bool
    runtime_writable: bool


CANONICAL_FIELDS: Mapping[str, StateFieldSpec] = {
    "semantic_state": StateFieldSpec(StateDomain.SEMANTIC, "model", True, True),
    "operational_self_state": StateFieldSpec(StateDomain.SOFT, "model", True, True),
    "epistemic_state": StateFieldSpec(StateDomain.EPISTEMIC, "runtime", False, True),
    "authority_state": StateFieldSpec(StateDomain.HARD, "trusted_runtime", False, True),
    "provenance_state": StateFieldSpec(StateDomain.PROVENANCE, "runtime", False, True),
    "temporal_state": StateFieldSpec(StateDomain.TEMPORAL, "runtime", False, True),
    "goal_state": StateFieldSpec(StateDomain.GOAL, "runtime", False, True),
}


class HardStateMutationError(PermissionError):
    """Raised when model-owned code attempts to mutate trusted hard state."""


@dataclass(frozen=True)
class CanonicalTypedActivation:
    """Versioned view of the canonical NSA activation/state protocol.

    ``state`` is deliberately retained as the existing Omega representation so
    current code remains compatible.  The protocol adds explicit ownership and
    mutation semantics around it rather than introducing a second state type.
    """

    state: UnifiedCognitiveState
    schema_version: str = CANONICAL_SCHEMA_VERSION

    def spec(self, field: str) -> StateFieldSpec:
        try:
            return CANONICAL_FIELDS[field]
        except KeyError as exc:
            raise KeyError(f"Unknown canonical state field: {field}") from exc

    def can_write(self, field: str, actor: str) -> bool:
        spec = self.spec(field)
        if actor == "runtime":
            return spec.runtime_writable
        if actor == "model":
            return spec.model_writable
        return False

    def model_proposal(self, field: str, value: Any) -> Dict[str, Any]:
        """Create a proposal without mutating the canonical state."""
        spec = self.spec(field)
        if not spec.model_writable:
            raise HardStateMutationError(
                f"Model cannot write {field}; it may only be proposed to the trusted runtime"
            )
        return {"schema_version": self.schema_version, "field": field, "value": value}

    def runtime_commit(self, field: str, value: Any) -> "CanonicalTypedActivation":
        """Commit a runtime-authorized state update.

        Hard fields are intentionally accepted only through this explicit
        runtime path.  The method creates a new activation instead of mutating
        the existing object in place.
        """
        spec = self.spec(field)
        if not spec.runtime_writable:
            raise HardStateMutationError(f"Runtime cannot write {field}")
        next_state = _replace_field(self.state, field, value)
        return CanonicalTypedActivation(next_state, self.schema_version)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the contract using JSON-compatible primitives."""
        return {
            "schema_version": self.schema_version,
            "state": {
                "semantic_state": _tensor(self.state.semantic_state),
                "operational_self_state": _tensor(self.state.operational_self_state),
                "epistemic_state": vars(self.state.epistemic_state),
                "authority_state": _tensor(self.state.authority_state),
                "provenance_state": vars(self.state.provenance_state),
                "temporal_state": vars(self.state.temporal_state),
                "goal_state": vars(self.state.goal_state),
            },
        }


def _tensor(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def _replace_field(state: UnifiedCognitiveState, field: str, value: Any) -> UnifiedCognitiveState:
    """Return a new Omega state with one validated component replaced."""
    allowed = set(CANONICAL_FIELDS)
    if field not in allowed:
        raise KeyError(field)
    values = {
        "semantic_state": state.semantic_state,
        "operational_self_state": state.operational_self_state,
        "epistemic_state": state.epistemic_state,
        "authority_state": state.authority_state,
        "provenance_state": state.provenance_state,
        "temporal_state": state.temporal_state,
        "goal_state": state.goal_state,
    }
    values[field] = value
    return UnifiedCognitiveState(**values)
