from __future__ import annotations

from typing import Any, Dict, List

from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.inference.governed import NSAGovernedInference


class FakeBackend(InferenceBackend):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, extract_hidden: bool = False, **kwargs: Any) -> LLMGenerationOutput:
        self.calls += 1
        return LLMGenerationOutput(text=f"real-backend:{prompt}", tokens=[1], confidence_estimate=0.73)

    def propose_action(self, system_context: str, task_instruction: str, available_tools: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        return {"action": available_tools[0]["name"], "params": {}, "confidence": 0.73}


def test_governed_inference_calls_real_backend_and_commits_state() -> None:
    backend = FakeBackend()
    runtime = NSAGovernedInference(backend, model_name="test-model")

    output = runtime.generate("hello")

    assert output.text == "real-backend:hello"
    assert backend.calls == 1
    assert runtime.state.temporal_state.step_index == 1
    assert runtime.status()["nsa_governance"] is True
    assert runtime.status()["weight_modification"] is False
    assert runtime.status()["last_kernel_verdict"] == "COMMIT"
    assert runtime.status()["last_kernel_invariants_satisfied"] is True


def test_governed_inference_preserves_provenance_chain() -> None:
    runtime = NSAGovernedInference(FakeBackend(), model_name="test-model")
    first_hash = runtime.state.provenance_state.hash_signature

    runtime.generate("one")
    second = runtime.state.provenance_state

    assert second.parent_records == ["prov-0"]
    assert second.record_id == "prov-1"
    assert second.hash_signature != first_hash
