from typing import Any, Dict, List

import pytest
import torch

from nsa.core.typed_activation import HardStateMutationError
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.typed_runtime import NSATypedRuntime


class FakeLiveBackend(InferenceBackend):
    """Deterministic stand-in for a live HTTP backend in unit tests."""

    def __init__(self) -> None:
        self.prompts: List[str] = []

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7, extract_hidden: bool = False):
        self.prompts.append(prompt)
        return LLMGenerationOutput(
            text="live model response",
            tokens=[],
            confidence_estimate=0.81,
            raw_response={"live": True},
        )

    def propose_action(self, system_context: str, task_instruction: str, available_tools: List[Dict[str, Any]]):
        return {"action": available_tools[0]["name"]}


def test_live_generation_passes_canonical_state_to_real_backend_boundary() -> None:
    backend = FakeLiveBackend()
    runtime = NSATypedRuntime(backend, semantic_dim=8)

    result = runtime.generate("Explain why state matters.")

    assert backend.prompts
    assert "NSA RUNTIME STATE" in backend.prompts[0]
    assert "authority_clearance_dim" in backend.prompts[0]
    assert result.state.state.temporal_state.step_index == 1
    assert result.state.state.provenance_state.record_id == "generation-1"
    assert not torch.equal(
        result.state.state.semantic_state,
        torch.zeros_like(result.state.state.semantic_state),
    )


def test_model_cannot_mutate_hard_authority_through_runtime_state() -> None:
    runtime = NSATypedRuntime(FakeLiveBackend())
    with pytest.raises(HardStateMutationError):
        runtime.activation.model_proposal("authority_state", torch.tensor([99.0]))

    before = runtime.inspect()["state"]["authority_state"]
    runtime.generate("attempt a normal response")
    after = runtime.inspect()["state"]["authority_state"]
    assert before == after


def test_runtime_updates_only_trusted_post_generation_state() -> None:
    runtime = NSATypedRuntime(FakeLiveBackend())
    result = runtime.generate("hello")

    assert result.state_before["state"]["temporal_state"]["step_index"] == 0
    assert result.state_after["state"]["temporal_state"]["step_index"] == 1
    assert result.state_after["state"]["provenance_state"]["parent_records"] == ["runtime-genesis"]


def test_reset_preserves_hard_authority_and_resets_session_state() -> None:
    runtime = NSATypedRuntime(FakeLiveBackend())
    runtime.generate("hello")
    authority_before = runtime.activation.state.authority_state.clone()

    runtime.reset()

    assert torch.equal(runtime.activation.state.authority_state, authority_before)
    assert runtime.activation.state.temporal_state.step_index == 0
    assert torch.equal(runtime.activation.state.semantic_state, torch.zeros_like(runtime.activation.state.semantic_state))
