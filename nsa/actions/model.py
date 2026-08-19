"""
nsa.actions.model
=================
Typed Action & Tool Execution Primitives for NSA.

Governs tool invocations as typed state transitions crossing trust boundaries:
1. Tool requests require explicit capability validation and caller clearance.
2. Actions are classified by risk and reversibility.
3. State propagates across tool boundaries (tool outputs inherit or elevate state levels).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple

from nsa.core.state import CanonicalState, HardState


class ToolRiskLevel(str, Enum):
    LOW = "low"            # Read-only operations, public info
    MEDIUM = "medium"      # Ephemeral computations, internal queries
    HIGH = "high"          # Persistent writes, external network I/O
    CRITICAL = "critical"  # Financial transactions, deletion, auth escalation


class ActionReversibility(str, Enum):
    REVERSIBLE = "reversible"
    COMPENSATING_ACTION_REQUIRED = "compensating_action_required"
    IRREVERSIBLE = "irreversible"


class ToolApprovalStatus(str, Enum):
    APPROVED = "approved"
    PENDING_APPROVAL = "pending_approval"
    REJECTED = "rejected"


@dataclass(frozen=True)
class TypedToolRequest:
    """An explicit, typed request to execute an external tool or action."""

    tool_name: str
    arguments: Mapping[str, Any]
    caller_state: CanonicalState
    required_capability_id: str
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    reversibility: ActionReversibility = ActionReversibility.REVERSIBLE
    approval_status: ToolApprovalStatus = ToolApprovalStatus.APPROVED
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def is_authorized(self, has_capability: bool) -> bool:
        if not has_capability:
            return False
        if self.approval_status != ToolApprovalStatus.APPROVED:
            return False
        return True


@dataclass(frozen=True)
class TypedToolResponse:
    """Result of a governed tool execution carrying updated state and provenance."""

    request_id: str
    tool_name: str
    result: Any
    output_state: CanonicalState
    success: bool
    execution_timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    side_effects: Tuple[str, ...] = ()


__all__ = [
    "ToolRiskLevel",
    "ActionReversibility",
    "ToolApprovalStatus",
    "TypedToolRequest",
    "TypedToolResponse",
]
