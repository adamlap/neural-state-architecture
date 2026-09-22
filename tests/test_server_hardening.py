"""HTTP control-plane hardening: path traversal, CORS, auth, body handling."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import nsa.server.proxy as proxy
from nsa.runtime.cce_checkpoint import CCECheckpointManager, validate_checkpoint_id
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


@pytest.fixture
def server(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    fake = SimpleNamespace(
        enable_cce=True,
        checkpoint_mgr=CCECheckpointManager(ckpt_dir),
        cce_state=PersistentCognitiveState(dimension=4),
        process_sensor_input=lambda text, source, importance: {"ok": True, "text": text, "importance": importance},
        status=lambda: {"status": "online"},
    )
    handler = type("Handler", (proxy.NSAHTTPHandler,), {"runtime": fake})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{httpd.server_address[1]}", ckpt_dir=ckpt_dir, tmp=tmp_path)
    httpd.shutdown()
    httpd.server_close()


def _request(url, *, method="GET", body=None, headers=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), json.loads(err.read() or b"{}")


@pytest.mark.parametrize("bad", ["../escape", "..\\escape", "/etc/passwd", "a/b", "", ".hidden", "a..b", "x" * 200, "id with space"])
def test_checkpoint_ids_that_could_escape_the_directory_are_rejected(bad):
    with pytest.raises(ValueError):
        validate_checkpoint_id(bad)


@pytest.mark.parametrize("good", ["cce_ckpt_1", "run-2026.09.20", "A1"])
def test_normal_checkpoint_ids_are_accepted(good):
    assert validate_checkpoint_id(good) == good


def test_checkpoint_endpoint_refuses_path_traversal_and_writes_nothing_outside(server):
    status, _, body = _request(server.url + "/api/cce/checkpoint", method="POST", body={"checkpoint_id": "../../pwned"})
    assert status == 400 and "invalid checkpoint id" in body["error"]
    assert not list(server.tmp.rglob("pwned*"))

    status, _, body = _request(server.url + "/api/cce/checkpoint", method="POST", body={"checkpoint_id": "good-id"})
    assert status == 200 and (server.ckpt_dir / "good-id.json").is_file()


def test_manager_rejects_traversal_on_save_fork_and_id_load(tmp_path):
    manager = CCECheckpointManager(tmp_path / "c")
    state = PersistentCognitiveState(dimension=4)
    for call in (
        lambda: manager.save_persistent_state(state, checkpoint_id="../x"),
        lambda: manager.fork_persistent_state(state, "../x"),
        lambda: manager.load_persistent_state("../../secret"),
    ):
        with pytest.raises(ValueError):
            call()
    assert not (tmp_path / "x.json").exists()


def test_atomic_write_leaves_no_temp_files_and_cleans_up_on_failure(tmp_path):
    manager = CCECheckpointManager(tmp_path / "c")
    manager.save_persistent_state(PersistentCognitiveState(dimension=4), checkpoint_id="ok")
    with pytest.raises(TypeError):
        manager._atomic_write_json(tmp_path / "c" / "bad.json", {"x": object()})
    assert sorted(p.name for p in (tmp_path / "c").iterdir()) == ["ok.json"]


def test_cors_is_not_a_wildcard(server, monkeypatch):
    monkeypatch.delenv("NSA_CORS_ORIGINS", raising=False)
    _, headers, _ = _request(server.url + "/health", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in headers
    _, headers, _ = _request(server.url + "/health", headers={"Origin": "http://localhost:3000"})
    assert headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    monkeypatch.setenv("NSA_CORS_ORIGINS", "https://ui.example")
    _, headers, _ = _request(server.url + "/health", headers={"Origin": "https://ui.example"})
    assert headers["Access-Control-Allow-Origin"] == "https://ui.example"
    _, headers, _ = _request(server.url + "/health")
    assert "Access-Control-Allow-Origin" not in headers


def test_bearer_token_is_enforced_when_configured(server, monkeypatch):
    monkeypatch.setenv("NSA_API_TOKEN", "s3cret")
    assert _request(server.url + "/health")[0] == 401
    assert _request(server.url + "/health", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert _request(server.url + "/api/cce/sensor", method="POST", body={"text": "x"})[0] == 401
    assert _request(server.url + "/health", headers={"Authorization": "Bearer s3cret"})[0] == 200
    monkeypatch.delenv("NSA_API_TOKEN")
    assert _request(server.url + "/health")[0] == 200  # unset token keeps the open default


def test_malformed_requests_get_4xx_not_dropped_connections(server, monkeypatch):
    monkeypatch.setattr(proxy, "MAX_BODY_BYTES", 64)
    assert _request(server.url + "/api/cce/sensor", method="POST", raw=b"x" * 500)[0] == 413
    monkeypatch.setattr(proxy, "MAX_BODY_BYTES", 1024 * 1024)
    assert _request(server.url + "/api/cce/sensor", method="POST", raw=b"[1, 2]")[0] == 400
    assert _request(server.url + "/api/cce/sensor", method="POST", body={"text": "hi", "importance": "high"})[0] == 400
    assert _request(server.url + "/v1/chat/completions", method="POST", body={"messages": "nope"})[0] == 400
    status, _, body = _request(server.url + "/api/cce/sensor", method="POST", body={"text": "hi", "importance": 0.7})
    assert status == 200 and body["importance"] == 0.7


def test_warns_when_exposed_without_token(monkeypatch, caplog):
    monkeypatch.delenv("NSA_API_TOKEN", raising=False)
    with caplog.at_level("WARNING", logger="NSAServer"):
        proxy.warn_if_exposed("127.0.0.1")
        assert not caplog.records
        proxy.warn_if_exposed("0.0.0.0")
    assert any("NSA_API_TOKEN" in r.message for r in caplog.records)


def test_runtime_close_stops_the_background_thread_and_is_idempotent():
    """Regression: the daemon CCE thread was never stopped, aborting the interpreter at exit."""
    runtime = proxy.NSAProxyRuntime.__new__(proxy.NSAProxyRuntime)
    runtime._stop_cce = threading.Event()
    ticks = []
    runtime._cce_thread = threading.Thread(target=lambda: (runtime._stop_cce.wait(30), ticks.append(1)), daemon=True)
    runtime._cce_thread.start()
    runtime.close()
    runtime.close()
    assert not runtime._cce_thread.is_alive() and ticks == [1]
