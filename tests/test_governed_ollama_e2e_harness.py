"""Tests for GovernedExecutionHarness (Phase C End-to-End Governance)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import pytest
from nsa.decision import Decision
from nsa.policy import NSAPolicy, PolicyRule
from nsa.runtime.governed_harness import GovernedExecutionHarness
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput


class MockInferenceBackend(InferenceBackend):
    def __init__(self, canned_response: str = "ok") -> None:
        self.canned = canned_response

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.7,
        extract_hidden: bool = False,
    ) -> LLMGenerationOutput:
        return LLMGenerationOutput(
            text=self.canned,
            tokens=[1, 2, 3],
            confidence_estimate=0.9,
            raw_response={"model": "mock-model"},
        )

    def propose_action(
        self,
        system_context: str,
        task_instruction: str,
        available_tools: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {"action": "none", "thought": self.canned}


def test_harness_allows_safe_text_turn():
    policy = NSAPolicy(
        name="test-policy",
        prohibited=(PolicyRule("malicious", "deny", patterns=("rm -rf", "exploit")),),
    )
    backend = MockInferenceBackend(canned_response="Here is the documentation.")
    harness = GovernedExecutionHarness(policy=policy, backend=backend)

    res = harness.run_turn("What is NSA?")
    assert res.decision.decision == Decision.ALLOW
    assert res.capability_executed is False
    assert res.normative_state.values.get("harm", 0.0) < 0.2


def test_harness_denies_prohibited_tool_with_zero_invocation():
    policy = NSAPolicy(
        name="test-policy",
        prohibited=(PolicyRule("dangerous_cmd", "deny", patterns=("delete", "destroy")),),
    )
    backend = MockInferenceBackend(canned_response="I will delete the database.")
    harness = GovernedExecutionHarness(policy=policy, backend=backend)

    calls = []

    def dangerous_delete(target: str) -> str:
        calls.append(target)
        return "deleted"

    harness.register_tool("db.delete", dangerous_delete)

    res = harness.run_turn("delete database", requested_tool="db.delete", tool_args=("production_db",))
    assert res.decision.decision == Decision.DENY
    assert res.capability_executed is False
    # Strict zero-invocation guarantee
    assert len(calls) == 0
    assert "DENIED" in str(res.denial_reason)
    assert res.normative_state.values.get("harm", 0.0) >= 0.2
