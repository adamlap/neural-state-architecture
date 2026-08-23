from nsa.runtime.cce_loop import ClosedLoopCognitiveInvoker
from nsa.runtime.cce_salience import AdaptiveSalienceGate, SalienceObservation
from nsa.runtime.inference.base import BackendMode
from nsa.runtime.inference.ollama import OllamaInferenceBackend


def test_quiet_stream_does_not_invoke_backend():
    backend = OllamaInferenceBackend(mode=BackendMode.MOCK)
    invoker = ClosedLoopCognitiveInvoker(backend, gate=AdaptiveSalienceGate())
    decision = invoker.observe(SalienceObservation(), "observe")
    assert decision.triggered is False
    assert decision.response is None
    assert invoker.invocation_count == 0


def test_salient_event_invokes_backend_without_mutating_state():
    responses = []
    backend = OllamaInferenceBackend(mode=BackendMode.MOCK)
    invoker = ClosedLoopCognitiveInvoker(
        backend, gate=AdaptiveSalienceGate(), on_response=responses.append
    )
    decision = invoker.observe(
        SalienceObservation(prediction_error=1.0),
        "observe",
    )
    assert decision.triggered is True
    assert decision.response
    assert invoker.invocation_count == 1
    assert responses == [decision.response]


def test_invoker_uses_same_gate_across_events():
    backend = OllamaInferenceBackend(mode=BackendMode.MOCK)
    invoker = ClosedLoopCognitiveInvoker(
        backend, gate=AdaptiveSalienceGate(baseline_decay=0.5)
    )
    first = invoker.observe(SalienceObservation(input_delta=1.0), "observe")
    second = invoker.observe(SalienceObservation(input_delta=1.0), "observe")
    assert first.triggered is True
    assert second.baseline > first.baseline
    assert invoker.invocation_count == 2
