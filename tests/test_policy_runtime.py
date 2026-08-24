"""Tests for the practical declarative policy runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from nsa.decision import Decision
from nsa.policy import NSAPolicy
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.policy_runtime import NSAPolicyRuntime


@dataclass
class FakeBackend(InferenceBackend):
    output: str = "A safe compiler explanation."
    calls: int = 0

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, extract_hidden: bool = False) -> LLMGenerationOutput:
        self.calls += 1
        return LLMGenerationOutput(text=self.output, confidence_estimate=1.0)

    def propose_action(self, system_context: str, task_instruction: str, available_tools: List[Dict[str, str]]) -> Dict[str, str]:
        return {"action": "none", "confidence": 1.0}


def policy() -> NSAPolicy:
    return NSAPolicy.from_mapping({
        "name": "test-policy",
        "prohibited": [{
            "category": "dangerous_request",
            "mode": "deny",
            "patterns": ["forbidden operation"],
        }],
        "unknown_policy": "escalate",
    })


def test_runtime_blocks_before_backend_call():
    backend = FakeBackend()
    runtime = NSAPolicyRuntime(backend, policy(), model_name="fake")
    result = runtime.generate("Please explain a forbidden operation")
    assert result.request_decision.decision is Decision.DENY
    assert result.blocked_stage == "request"
    assert result.generated is False
    assert backend.calls == 0


def test_runtime_allows_safe_generation():
    backend = FakeBackend()
    runtime = NSAPolicyRuntime(backend, policy(), model_name="fake")
    result = runtime.generate("Explain what a compiler does")
    assert result.request_decision.decision is Decision.ALLOW
    assert result.output_decision.decision is Decision.ALLOW
    assert result.generated is True
    assert result.text == "A safe compiler explanation."
    assert backend.calls == 1


def test_runtime_blocks_policy_violation_in_model_output():
    backend = FakeBackend(output="forbidden operation")
    runtime = NSAPolicyRuntime(backend, policy(), model_name="fake")
    result = runtime.generate("Explain something safe")
    assert result.request_decision.decision is Decision.ALLOW
    assert result.output_decision.decision is Decision.DENY
    assert result.blocked_stage == "output"
    assert result.generated is True
    assert "can't help" in result.text
    assert backend.calls == 1
