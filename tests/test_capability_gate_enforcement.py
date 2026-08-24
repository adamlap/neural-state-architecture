"""Tests for CapabilityGate, strict zero-invocation on DENY, and audit logging."""
from __future__ import annotations

import pytest
from nsa.capabilities.gate import (
    CapabilityAccessDenied,
    CapabilityApprovalRequired,
    CapabilityGate,
)
from nsa.capabilities.model import Capability, CapabilityAuthority
from nsa.decision import Decision, SecurityDecision


def test_capability_gate_blocks_deny_with_zero_invocation():
    gate = CapabilityGate()
    deny_decision = SecurityDecision(
        decision=Decision.DENY,
        policy="safety-core",
        reason="malicious shell command pattern detected",
        matched_categories=("destructive_execution",),
    )

    invocations = []

    def dangerous_tool(cmd: str) -> str:
        invocations.append(cmd)
        return f"executed {cmd}"

    with pytest.raises(CapabilityAccessDenied) as exc:
        gate.require(deny_decision, "system.shell", dangerous_tool, "rm -rf /")

    assert "DENIED by policy" in str(exc.value)
    # INVARIANT: Zero invocations on DENY
    assert len(invocations) == 0
    assert gate.total_invocations == 0
    assert gate.total_denials == 1
    assert len(gate.audit_log) == 1
    assert gate.audit_log[0].permitted is False
    assert gate.audit_log[0].capability_name == "system.shell"


def test_capability_gate_permits_allow_and_increments_count():
    gate = CapabilityGate()
    allow_decision = SecurityDecision(
        decision=Decision.ALLOW,
        policy="safety-core",
        reason="safe read operation",
    )

    invocations = []

    def safe_read(path: str) -> str:
        invocations.append(path)
        return f"content of {path}"

    res = gate.require(allow_decision, "filesystem.read", safe_read, "/tmp/test.txt")
    assert res == "content of /tmp/test.txt"
    assert len(invocations) == 1
    assert gate.total_invocations == 1
    assert gate.total_denials == 0
    assert gate.audit_log[0].permitted is True


def test_capability_gate_requires_approval():
    gate = CapabilityGate(strict_approval=True)
    approval_decision = SecurityDecision(
        decision=Decision.REQUIRE_APPROVAL,
        policy="safety-core",
        reason="database mutation requires human approval",
    )

    def write_db() -> str:
        return "written"

    # Without approval token -> raises
    with pytest.raises(CapabilityApprovalRequired):
        gate.require(approval_decision, "db.write", write_db)

    # With approval token -> allowed
    res = gate.require(approval_decision, "db.write", write_db, approval_token="token-approved-123")
    assert res == "written"


def test_capability_gate_with_authority_registry():
    cap = Capability(
        capability_id="cap-read-01",
        issuer="root_authority",
        subject="agent_alice",
        action="execute",
        scope="local",
        purpose="reading metrics",
    )
    auth = CapabilityAuthority("root_authority").issue(cap)
    gate = CapabilityGate(authority=auth)

    allow_decision = SecurityDecision(
        decision=Decision.ALLOW,
        policy="default",
        reason="authorized action",
    )

    def do_work() -> str:
        return "done"

    # Authorized subject
    res = gate.require(allow_decision, "cap-read-01", do_work, caller="agent_alice")
    assert res == "done"

    # Unauthorized subject
    with pytest.raises(CapabilityAccessDenied) as exc:
        gate.require(allow_decision, "cap-read-01", do_work, caller="agent_eve")
    assert "does not authorize subject 'agent_eve'" in str(exc.value)
