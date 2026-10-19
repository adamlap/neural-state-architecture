"""
nsa.actions.governor
====================
ToolGovernor: Trusted Execution Governor for NSA Actions and Tools.

Enforces:
1. Intelligence != Authority (Model text generation cannot trigger tools directly).
2. Capability validation via CapabilityAuthority.
3. Tool-specific hard authorizations declared at registration.
4. Optional declarative FlowGraph validation when a graph is configured.
5. Risk-based human-in-the-loop approval gating.
6. Reversibility tracking & transactional side-effect logging.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from nsa.capabilities.model import CapabilityAuthority
from nsa.core.state import CanonicalState
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
        # request_id -> (tool_name, arguments, capability_id) of governor-prepared requests.
        self._pending: Dict[str, Tuple[str, Dict[str, Any], str]] = {}
        self._approved: Dict[str, Tuple[str, Dict[str, Any], str]] = {}
        self._rejected: set[str] = set()
        self._consumed_nonces: set[str] = set()

    def _requires_manual_approval(self, risk: ToolRiskLevel) -> bool:
        return risk in (ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL) and self.auto_approve_risk_limit in (
            ToolRiskLevel.LOW,
            ToolRiskLevel.MEDIUM,
        )

    def register_tool(
        self,
        name: str,
        handler: Callable[..., Any],
        risk_level: ToolRiskLevel = ToolRiskLevel.LOW,
        reversibility: ActionReversibility = ActionReversibility.REVERSIBLE,
        required_authorizations: frozenset[str] = frozenset(),
        undo_handler: Optional[Callable[..., None]] = None,
        flow_source: str = "agent",
        flow_dimensions: frozenset[str] = frozenset({"semantic", "hard", "soft", "provenance"}),
    ) -> None:
        """Register an external executable tool with safety metadata.

        ``required_authorizations`` are checked against the caller's hard state
        before the handler can execute. If a FlowGraph has been configured with
        nodes/edges, the declared source and tool destination are also checked.
        """
        if not name:
            raise ValueError("tool name must be non-empty")
        self._registered_tools[name] = handler
        self._tool_metadata[name] = {
            "risk_level": risk_level,
            "reversibility": reversibility,
            "required_authorizations": frozenset(required_authorizations),
            "undo_handler": undo_handler,
            "flow_source": flow_source,
            "flow_dimensions": frozenset(flow_dimensions),
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

        needs_manual_approval = self._requires_manual_approval(risk)
        status = ToolApprovalStatus.PENDING_APPROVAL if needs_manual_approval else ToolApprovalStatus.APPROVED

        request = TypedToolRequest(
            tool_name=tool_name,
            arguments=arguments,
            caller_state=caller_state,
            required_capability_id=capability_id,
            risk_level=risk,
            reversibility=reversibility,
            approval_status=status,
        )
        if needs_manual_approval:
            self._pending[request.request_id] = (tool_name, dict(arguments), capability_id)
        return request

    def approve_request(self, request_id: str) -> bool:
        """Approve a pending governor-prepared request; returns False if unknown or already decided."""
        entry = self._pending.pop(request_id, None)
        if entry is None:
            return False
        self._approved[request_id] = entry
        return True

    def reject_request(self, request_id: str) -> bool:
        """Reject a pending or approved request so it can never execute."""
        entry = self._pending.pop(request_id, None) or self._approved.pop(request_id, None)
        if entry is None:
            return False
        self._rejected.add(request_id)
        return True

    def _approval_error(self, request: TypedToolRequest) -> Optional[str]:
        """Derive approval from registered metadata and the governor's registry, never from the request."""
        if request.request_id in self._rejected or request.approval_status == ToolApprovalStatus.REJECTED:
            return "Tool execution was rejected"
        risk = self._tool_metadata[request.tool_name]["risk_level"]
        if not self._requires_manual_approval(risk):
            return None
        entry = self._approved.get(request.request_id)
        if entry is None:
            return "Tool execution requires explicit approval (current status: pending_approval)"
        if entry != (request.tool_name, dict(request.arguments), request.required_capability_id):
            return "Tool execution requires explicit approval (request differs from the approved request)"
        return None

    def _validate_registration_constraints(self, request: TypedToolRequest) -> Optional[str]:
        meta = self._tool_metadata[request.tool_name]
        required = meta["required_authorizations"]
        if not required.issubset(request.caller_state.hard.authorizations):
            missing = sorted(required - request.caller_state.hard.authorizations)
            return f"missing required authorizations: {', '.join(missing)}"

        # An empty graph is the default and means no additional flow contract
        # was configured. Once a graph is populated, registered tool flows are
        # authoritative and must be explicitly permitted.
        if self.flow_graph.nodes or self.flow_graph.edges:
            violation = self.flow_graph.check_flow(
                meta["flow_source"],
                request.tool_name,
                meta["flow_dimensions"],
                request.caller_state.hard,
            )
            if violation is not None:
                return f"flow denied: {violation.reason}"
        return None

    def execute(
        self,
        request: TypedToolRequest,
        subject: str = "agent",
        scope: str = "tool_execution",
    ) -> TypedToolResponse:
        """Authoritatively evaluate permissions and execute tool."""
        if request.tool_name not in self._registered_tools:
            return self._reject(request, f"unknown tool: {request.tool_name}")

        capability = self.authority.match(
            capability_id=request.required_capability_id,
            subject=subject,
            action=request.tool_name,
            scope=scope,
        )

        if capability is None:
            return self._reject(request, f"Capability '{request.required_capability_id}' not authorized for action '{request.tool_name}'")

        constraint_error = self._validate_registration_constraints(request)
        if constraint_error is not None:
            return self._reject(request, constraint_error)

        approval_error = self._approval_error(request)
        if approval_error is not None:
            return self._reject(request, approval_error)

        if capability.nonce is not None and capability.nonce in self._consumed_nonces:
            return self._reject(request, f"Capability '{capability.capability_id}' nonce already consumed (replay rejected)")

        # Approvals and nonces are single-use; consume them before the handler runs.
        self._approved.pop(request.request_id, None)
        if capability.nonce is not None:
            self._consumed_nonces.add(capability.nonce)

        handler = self._registered_tools[request.tool_name]
        try:
            raw_result = handler(**request.arguments)
            success = True
            error_msg = None
        except Exception as exc:
            raw_result = None
            success = False
            error_msg = str(exc)

        new_provenance = request.caller_state.provenance.extend(
            transformation=f"tool_exec:{request.tool_name}:{request.request_id[:8]}"
        )
        output_state = CanonicalState(
            semantic=raw_result if success else request.caller_state.semantic,
            hard=request.caller_state.hard,
            soft=request.caller_state.soft,
            provenance=new_provenance,
            goals=request.caller_state.goals,
            step=request.caller_state.step + 1,
        )

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

    def _reject(self, request: TypedToolRequest, error_message: str) -> TypedToolResponse:
        response = TypedToolResponse(
            request_id=request.request_id,
            tool_name=request.tool_name,
            result=None,
            output_state=request.caller_state,
            success=False,
            error_message=error_message,
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
