import torch

from experiments.live.ollama_matched_benchmark import _authority_is_unchanged, normalize, score
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.typed_runtime import NSATypedRuntime


class DeterministicBackend(InferenceBackend):
    def generate(self, prompt, max_tokens=256, temperature=0.7, extract_hidden=False):
        return LLMGenerationOutput(
            text="703",
            tokens=[703],
            confidence_estimate=0.9,
        )

    def propose_action(self, system_context, task_instruction, available_tools):
        return {"action": available_tools[0]["name"]}


def test_benchmark_normalization_and_scoring_are_deterministic():
    assert normalize("  Neural\nState   Safety ") == "neural state safety"
    assert score("703\n", "703")
    assert not score("704", "703")


def test_runtime_benchmark_boundary_preserves_hard_authority():
    runtime = NSATypedRuntime(DeterministicBackend(), goal_id="test")
    before = runtime.activation.state.authority_state.detach().clone()

    runtime.generate("Compute 37 * 19", max_tokens=8, temperature=0.0)

    assert torch.equal(runtime.activation.state.authority_state, before)
    assert _authority_is_unchanged(runtime)
    assert runtime.activation.state.temporal_state.step_index == 1
    assert runtime.activation.state.provenance_state.record_id == "generation-1"
