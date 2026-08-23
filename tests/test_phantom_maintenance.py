import torch

from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.runtime.typed_runtime import NSATypedRuntime
from nsa.runtime.phantom_maintenance import PhantomMaintenanceLoop, maintain


class DummyBackend:
    def generate(self, prompt, *, max_tokens=256, temperature=0.7):
        return LLMGenerationOutput(text="ok", confidence_estimate=1.0)


def test_maintenance_advances_persistent_state_without_model_call():
    runtime = NSATypedRuntime(DummyBackend(), goal_id="maintenance-test")
    before_step = runtime.activation.state.temporal_state.step_index
    authority = runtime.activation.state.authority_state.detach().clone()

    result = maintain(runtime, elapsed_seconds=0.5)

    after_step = runtime.activation.state.temporal_state.step_index
    assert result.changed is True
    assert result.step_after == result.step_before + 1
    assert after_step == before_step + 1
    assert torch.equal(authority, runtime.activation.state.authority_state)
    assert result.hard_authority_unchanged is True


def test_maintenance_loop_is_explicit_and_bounded():
    runtime = NSATypedRuntime(DummyBackend(), goal_id="maintenance-loop-test")
    loop = PhantomMaintenanceLoop(runtime, interval_seconds=0.01)
    first = loop.tick()
    second = loop.tick()
    assert first.step_after == first.step_before + 1
    assert second.step_after == second.step_before + 1
