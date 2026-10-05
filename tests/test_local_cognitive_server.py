from unittest.mock import MagicMock, patch

from nsa.runtime.inference.base import BackendMode
from nsa.server.proxy import NSAProxyRuntime


class FakeTransformersBackend:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.mode = BackendMode.REMOTE

    def load_model(self):
        return True

    def generate(self, prompt, max_tokens=256, temperature=0.7, extract_hidden=False):
        from nsa.runtime.inference.base import LLMGenerationOutput
        return LLMGenerationOutput(text="local response", tokens=[1], confidence_estimate=0.9)

    def propose_action(self, **kwargs):
        return {"action": "none", "confidence": 0.9}


class FakeSystemOneBackend:
    name = "fake-system1"

    def decide(self, question, state, *, features=None):
        from nsa.cognition.system_one_runtime import SystemOneDecision
        choice = question.choices[0]
        return SystemOneDecision(
            question=question.name,
            selected_choice=choice,
            probabilities={c: (1.0 if c == choice else 0.0) for c in question.choices},
            confidence=1.0,
            uncertainty=0.0,
            risk=0.0,
            passed_gate=True,
            backend=self.name,
            latency_ms=0.0,
        )


def test_local_server_wires_a_generation_free_system_one():
    with patch("nsa.server.proxy.PyTorchTransformersBackend", FakeTransformersBackend),          patch("nsa.server.proxy.FrozenCausalLMLogitBackend", lambda *a, **k: FakeSystemOneBackend()):
        runtime = NSAProxyRuntime(
            backend_type="transformers",
            model="Qwen/Qwen2.5-3B-Instruct",
            enable_cce=False,
            system_one_model="Qwen/Qwen2.5-1.5B-Instruct",
            system_one_enabled=True,
        )
        try:
            assert runtime.system_one is not None
            tick = runtime.system_one.tick(
                runtime.canonical_state,
                observations=("test",),
                models=(runtime.system_one_model_name, runtime.model_name),
            )
            assert tick.selected_model == runtime.system_one_model_name
            assert tick.memory_policy in {"discard", "working", "episodic", "durable"}
        finally:
            runtime.close()
