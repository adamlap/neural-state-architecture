import time

from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.runtime.typed_runtime import NSATypedRuntime
from nsa.runtime.continuous_supervisor import ContinuousRuntimeSupervisor


class CountingBackend:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, *, max_tokens=256, temperature=0.7):
        self.calls += 1
        return LLMGenerationOutput(text=f"heartbeat-{self.calls}", confidence_estimate=1.0)


def test_supervisor_keeps_state_alive_and_model_active():
    backend = CountingBackend()
    runtime = NSATypedRuntime(backend, goal_id="supervisor-test")
    supervisor = ContinuousRuntimeSupervisor(
        runtime,
        maintenance_interval=0.01,
        model_interval=0.03,
    )
    supervisor.start()
    time.sleep(0.12)
    status = supervisor.stop()

    assert status.maintenance_ticks >= 3
    assert status.model_ticks >= 1
    assert backend.calls == status.model_ticks
    assert status.state_step >= status.model_ticks
    assert status.hard_authority_unchanged is True
    assert status.last_error is None
