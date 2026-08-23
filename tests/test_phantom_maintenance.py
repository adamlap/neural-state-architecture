import torch

from nsa.runtime.typed_runtime import NSATypedRuntime
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.phantom_maintenance import PhantomMaintenanceLoop, maintain


class DummyBackend:
    def generate(self, prompt, *, max_tokens=256, temperature=0.7):
        from nsa.runtime.inference.base import LLMGenerationOutput
        return LLMGenerationOutput(text="ok", model_name="dummy", latency_seconds=0.0, confidence_estimate=1.0)


def test_maintenance_advances_persistent_state_without_model_call():
    runtime = NSATypedRuntime(DummyBackend(), goal_id="maintenance-test")
    before = runtime.inspect()
    authority = runtime.activation.state.authority_state.detach().clone()

    result = maintain(runtime, elapsed_seconds=0.5)

    after = runtime.inspect()
    assert result.changed is True
    assert result.step_after == result.step_before + 1
    assert after["temporal_state"]["step_index"] == before["temporal_state"]["step_index"] + 1
    assert torch.equal(authority, runtime.activation.state.authority_state)
    assert result.hard_authority_unchanged is True


def test_maintenance_loop_is_explicit_and_bounded():
    runtime = NSATypedRuntime(DummyBackend(), goal_id="maintenance-loop-test")
    loop = PhantomMaintenanceLoop(runtime, interval_seconds=0.01)
    first = loop.tick()
    second = loop.tick()
    assert first.step_after == first.step_before + 1
    assert second.step_after == second.step_before + 1
