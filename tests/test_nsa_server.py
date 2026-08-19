"""
tests/test_nsa_server.py
========================
Unit tests for the NSA OpenAI/Ollama compatible API proxy server.
"""

import json
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.server.proxy import NSAHTTPHandler, NSAProxyRuntime


@pytest.fixture(scope="module")
def nsa_test_server(monkeypatch_module):
    # These are HTTP/server unit tests. Keep the production backend real, but
    # isolate this suite from requiring an Ollama daemon on the CI runner.
    monkeypatch_module.setattr(OllamaInferenceBackend, "_resolve_connection", lambda self: None)
    runtime = NSAProxyRuntime(backend_type="ollama", model="qwen2.5:3b")
    runtime.backend.generate_text = lambda prompt, system_prompt=None, max_tokens=1024, temperature=0.7: "Test NSA Response: Cluster state nominal."
    NSAHTTPHandler.runtime = runtime

    server = ThreadingHTTPServer(("127.0.0.1", 18888), NSAHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    yield "http://127.0.0.1:18888"
    server.shutdown()


def test_server_health(nsa_test_server):
    url = f"{nsa_test_server}/health"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert data["status"] == "online"
        assert data["service"] == "Neural State Architecture Cognitive Runtime Server"


def test_server_v1_models(nsa_test_server):
    url = f"{nsa_test_server}/v1/models"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert "nsa-qwen2.5:3b" in model_ids


def test_server_api_tags(nsa_test_server):
    url = f"{nsa_test_server}/api/tags"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert "models" in data
        assert len(data["models"]) >= 1


def test_server_openai_chat_completions(nsa_test_server):
    url = f"{nsa_test_server}/v1/chat/completions"
    payload = {
        "model": "nsa-qwen2.5:3b",
        "messages": [
            {"role": "system", "content": "You are a helpful DevOps assistant."},
            {"role": "user", "content": "How do we recover from certificate expiration?"},
        ],
        "stream": False,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert "choices" in data
        assert len(data["choices"]) > 0
        content = data["choices"][0]["message"]["content"]
        assert "Cluster state nominal" in content
        assert "NSA Cognitive Governance" in content


def test_server_ollama_api_chat(nsa_test_server):
    url = f"{nsa_test_server}/api/chat"
    payload = {"model": "nsa-qwen2.5:3b", "messages": [{"role": "user", "content": "Status report?"}], "stream": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        assert resp.status == 200
        assert "message" in data
        assert "Cluster state nominal" in data["message"]["content"]
