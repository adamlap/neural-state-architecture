"""
tests/test_actions.py
=====================
Unit tests for NSA Tool & Action Governance subsystem (Phase 20).
"""

import unittest
from nsa.actions import (
    ActionReversibility,
    ToolApprovalStatus,
    ToolGovernor,
    ToolRiskLevel,
    TypedToolRequest,
)
from nsa.capabilities.model import Capability, CapabilityAuthority
from nsa.core.state import CanonicalState, HardState, SoftState, ProvenanceState
from nsa.flow.graph import FlowEdge, FlowGraph, FlowNode


class TestActionGovernance(unittest.TestCase):
    def setUp(self):
        self.cap_read = Capability(
            capability_id="cap_db_read", issuer="admin_authority", subject="agent",
            action="query_db", scope="tool_execution", purpose="retrieve user data",
        )
        self.cap_write = Capability(
            capability_id="cap_db_write", issuer="admin_authority", subject="agent",
            action="update_db", scope="tool_execution", purpose="modify user record",
        )
        self.authority = CapabilityAuthority(
            issuer_id="admin_authority", capabilities=frozenset([self.cap_read, self.cap_write])
        )
        self.governor = ToolGovernor(
            capability_authority=self.authority, auto_approve_risk_limit=ToolRiskLevel.MEDIUM
        )
        self.db = {"user_1": "Alice"}

        def query_db(user_id: str):
            return self.db.get(user_id, None)

        def update_db(user_id: str, new_name: str):
            self.db[user_id] = new_name
            return f"Updated {user_id} -> {new_name}"

        def undo_update_db(user_id: str, new_name: str):
            self.db[user_id] = "Alice"

        self.governor.register_tool(
            name="query_db", handler=query_db, risk_level=ToolRiskLevel.LOW,
            reversibility=ActionReversibility.REVERSIBLE,
        )
        self.governor.register_tool(
            name="update_db", handler=update_db, risk_level=ToolRiskLevel.HIGH,
            reversibility=ActionReversibility.REVERSIBLE, undo_handler=undo_update_db,
        )
        self.caller_state = CanonicalState(
            semantic="init", hard=HardState(), soft=SoftState(),
            provenance=ProvenanceState(sources=("user_input",)),
        )

    def test_authorized_tool_execution(self):
        req = self.governor.prepare_request("query_db", {"user_id": "user_1"}, self.caller_state, "cap_db_read")
        self.assertEqual(req.approval_status, ToolApprovalStatus.APPROVED)
        resp = self.governor.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(resp.result, "Alice")
        self.assertIn("tool_exec:query_db", resp.output_state.provenance.transformations[0])

    def test_unauthorized_tool_blocked(self):
        req = self.governor.prepare_request("query_db", {"user_id": "user_1"}, self.caller_state, "invalid_capability_xyz")
        resp = self.governor.execute(req)
        self.assertFalse(resp.success)
        self.assertIn("not authorized", resp.error_message)

    def test_high_risk_approval_gating(self):
        req = self.governor.prepare_request("update_db", {"user_id": "user_1", "new_name": "Bob"}, self.caller_state, "cap_db_write")
        self.assertEqual(req.approval_status, ToolApprovalStatus.PENDING_APPROVAL)
        resp = self.governor.execute(req)
        self.assertFalse(resp.success)
        self.assertIn("requires explicit approval", resp.error_message)
        self.assertEqual(self.db["user_1"], "Alice")
        approved_req = TypedToolRequest(
            tool_name=req.tool_name, arguments=req.arguments, caller_state=req.caller_state,
            required_capability_id=req.required_capability_id, risk_level=req.risk_level,
            reversibility=req.reversibility, approval_status=ToolApprovalStatus.APPROVED,
            request_id=req.request_id,
        )
        resp_approved = self.governor.execute(approved_req)
        self.assertTrue(resp_approved.success)
        self.assertEqual(self.db["user_1"], "Bob")

    def test_reversible_action_rollback(self):
        req = TypedToolRequest(
            tool_name="update_db", arguments={"user_id": "user_1", "new_name": "Charlie"},
            caller_state=self.caller_state, required_capability_id="cap_db_write",
            risk_level=ToolRiskLevel.HIGH, reversibility=ActionReversibility.REVERSIBLE,
            approval_status=ToolApprovalStatus.APPROVED,
        )
        resp = self.governor.execute(req)
        self.assertTrue(resp.success)
        self.assertEqual(self.db["user_1"], "Charlie")
        self.assertTrue(self.governor.rollback_last_action())
        self.assertEqual(self.db["user_1"], "Alice")

    def test_required_authorization_blocks_handler(self):
        calls = []
        capability = Capability(
            capability_id="cap_privileged", issuer="admin_authority", subject="agent",
            action="privileged_read", scope="tool_execution", purpose="privileged read",
        )
        governor = ToolGovernor(self.authority.issue(capability))
        governor.register_tool(
            name="privileged_read", handler=lambda: calls.append(True),
            required_authorizations=frozenset({"database.read"}),
        )
        request = governor.prepare_request("privileged_read", {}, self.caller_state, "cap_privileged")
        response = governor.execute(request)
        self.assertFalse(response.success)
        self.assertIn("missing required authorizations", response.error_message)
        self.assertEqual(calls, [])

    def test_configured_flow_blocks_handler(self):
        calls = []
        graph = FlowGraph()
        graph.add_node(FlowNode("agent", "caller"))
        graph.add_node(FlowNode("restricted_tool", "tool"))
        graph.add_edge(FlowEdge("agent", "restricted_tool", frozenset({"semantic"})))
        governor = ToolGovernor(self.authority, flow_graph=graph)
        governor.register_tool("restricted_tool", lambda: calls.append(True))
        request = governor.prepare_request("restricted_tool", {}, self.caller_state, "cap_db_read")
        response = governor.execute(request)
        self.assertFalse(response.success)
        self.assertIn("flow denied", response.error_message)
        self.assertEqual(calls, [])

    def test_configured_flow_allows_declared_dimensions(self):
        calls = []
        graph = FlowGraph()
        graph.add_node(FlowNode("agent", "caller"))
        graph.add_node(FlowNode("query_db", "tool"))
        graph.add_edge(FlowEdge("agent", "query_db", frozenset({"semantic", "hard", "soft", "provenance"})))
        governor = ToolGovernor(self.authority, flow_graph=graph)
        governor.register_tool("query_db", lambda: calls.append(True))
        request = governor.prepare_request("query_db", {}, self.caller_state, "cap_db_read")
        response = governor.execute(request)
        self.assertTrue(response.success)
        self.assertEqual(calls, [True])


if __name__ == "__main__":
    unittest.main()
