"""Concurrency guarantees for NSAProxyRuntime / NSAGovernedInference."""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.server.proxy import NSAProxyRuntime

N = 24


def _runtime(**kw):
    with patch.object(OllamaInferenceBackend, "_resolve_connection", lambda self: None):
        rt = NSAProxyRuntime(backend_type="ollama", model="qwen2.5:3b", **kw)
    state = {"active": 0, "max": 0}
    guard = threading.Lock()

    def generate(**kwargs):
        with guard:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        time.sleep(0.01)
        with guard:
            state["active"] -= 1
        return LLMGenerationOutput(text="ok", tokens=[1], confidence_estimate=0.9)

    rt.backend.generate = generate
    return rt, state


def test_parallel_chats_no_lost_provenance_updates():
    rt, state = _runtime()
    try:
        msgs = [{"role": "user", "content": "hello there, tell me about certificates"}]
        with ThreadPoolExecutor(max_workers=N) as ex:
            list(ex.map(lambda _: rt.process_chat(msgs), range(N)))
        status = rt.governed.status()
        assert status["state_step"] == N
        assert status["provenance_record"] == f"prov-{N}"
        assert state["max"] == 1, "backend forward must be serialized"
    finally:
        rt.close()


def test_background_loop_failure_is_logged(caplog):
    rt, _ = _runtime(enable_cce=True)
    try:
        def boom(*a, **k):
            raise RuntimeError("tick exploded")

        rt.cce_state.snapshot = boom
        with caplog.at_level(logging.ERROR, logger="NSAServer"):
            deadline = time.time() + 5
            while time.time() < deadline and "CCE background tick failed" not in caplog.text:
                time.sleep(0.1)
        assert "CCE background tick failed" in caplog.text
    finally:
        rt.close()
