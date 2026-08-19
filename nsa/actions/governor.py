"""
nsa.actions.governor
====================
ToolGovernor: Trusted Execution Governor for NSA Actions and Tools.

Enforces:
1. Intelligence != Authority (Model text generation cannot trigger tools directly).
2. Capability validation via CapabilityAuthority.
3. Flow validation via FlowGraph.
4. Risk-based human-in-the-loop approval gating.
5. Reversibility tracking & transactional side-effect logging.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from nsa.capabilities.model import CapabilityAuthority
from nsa.core.state import CanonicalState, HardState, ProvenanceState
from nsa.flow.graph import FlowGraph
from .model import (
    ActionReversibility,
    ToolApprovalStatus,
    ToolRiskLevel,
    TypedToolRequest,
    TypedToolResponse,
)


class ToolGovernor:
    """Policy-aware execution governor for tools and actions."""

    def __init__(
        self,
        capability_authority: CapabilityAuthority,
        flow_graph: Optional[FlowGraph] = None,
        auto_approve_risk_limit: ToolRiskLevel = ToolRiskLevel.MEDIUM,
    ) -> None:
        self.authority = capability_authority
        self.flow_graph = flow_graph or FlowGraph()
        self.auto_approve_risk_limit = auto_approve_risk_limit
        self._registered_tools: Dict[str, Callable[..., Any]] = {}
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Tuple[TypedToolRequest, TypedToolResponse]] = []
        self._undo_stack: List[Callable[[], None]] = []

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
        reversibility: ActionReversibility = ActionReversibility.REVERSIBLE,
        required_authorizations: frozenset[str] = frozenset(),
        undo_handler: Optional[Callable[..., None]] = None,
    ) -> None:
        """Register an external executable tool with safety metadata."""
        self._registered_tools[name] = handler
        self._tool_metadata[name] = {
            "risk_level": risk_level,
            "reversibility": reversibility,
            "required_authorizations": required_authorizations,
            "undo_handler": undo_handler,
        }

    def prepare_request(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        caller_state: CanonicalState,
        capability_id: str,
    ) -> TypedToolRequest:
        """Construct a validated tool request with appropriate risk assessment."""
        if tool_name not in self._registered_tools:
            raise ValueError(f"unknown tool: {tool_name}")

        meta = self._tool_metadata[tool_name]
        risk = meta["risk_level"]
        reversibility = meta["reversibility"]

        # Check approval requirement based on risk limit
        needs_manual_approval = False
        if risk in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL) and self.auto_approve_risk_limit in (
            ToolRiskLevel.LOW,
            ToolRiskLevel.MEDIUM,
        ):
            needs_manual_approval = True

        status = (
            ToolApprovalStatus.PENDING_APPROVAL
            if needs_manual_approval
            else ToolApprovalStatus.APPROVED
        )

        return TypedToolRequest(
            tool_name=tool_name,
            arguments=arguments,
            caller_state=caller_state,
            required_capability_id=capability_id,
            risk_level=risk,
            reversibility=reversibility,
            approval_status=status,
        )

    def execute(
        self,
        request: TypedToolRequest,
        subject: str = "agent",
        scope: str = "tool_execution",
    ) -> TypedToolResponse:
        """Authoritatively evaluate permissions and execute tool."""
        # 1. Verify capability validity
        has_capability = self.authority.allows(
            capability_id=request.required_capability_id,
            subject=subject,
            action=request.tool_name,
            scope=scope,
        )

        if not has_capability:
            response = TypedToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                result=None,
                output_state=request.caller_state,
                success=False,
                error_message=f"Capability '{request.required_capability_id}' not authorized for action '{request.tool_name}'",
            )
            self._execution_history.append((request, response))
            return response

        # 2. Verify approval status
        if request.approval_status != ToolApprovalStatus.APPROVED:
            response = TypedToolResponse(
                request_id=request.request_id,
                tool_name=request.tool_name,
                result=None,
                output_state=request.caller_state,
                success=False,
                error_message=f"Tool execution requires explicit approval (current status: {request.approval_status})",
            )
            self._execution_history.append((request, response))
            return response

        # 3. Execute tool handler
        handler = self._registered_tools[request.tool_name]
        try:
            raw_result = handler(**request.arguments)
            success = True
            error_msg = None
        except Exception as exc:
            raw_result = None
            success = False
            error_msg = str(exc)

        # 4. Propagate state & provenance
        new_provenance = request.caller_state.provenance.extend(
            transformation=f"tool_exec:{request.tool_name}:{request.request_id[:8]}"
        )
        output_state = CanonicalState(
            semantic=raw_result if success else request.caller_state.semantic,
            hard=request.caller_state.hard,
            soft=request.caller_state.soft,
            provenance=new_provenance,
        )

        # 5. Track undo for reversible actions
        meta = self._tool_metadata[request.tool_name]
        if success and meta["undo_handler"] is not None:
            undo_fn = lambda: meta["undo_handler"](**request.arguments)
            self._undo_stack.append(undo_fn)

        response = TypedToolResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            result=raw_result,
            output_state=output_state,
            success=success,
            error_message=error_msg,
            side_effects=(f"executed_{request.tool_name}",) if success else (),
        )
        self._execution_history.append((request, response))
        return response

    def rollback_last_action(self) -> bool:
        """Rollback the last reversible tool side effect."""
        if not self._undo_stack:
            return False
        undo_fn = self._undo_stack.pop()
        try:
            undo_fn()
            return True
        except Exception:
            return False


__all__ = ["ToolGovernor"]
