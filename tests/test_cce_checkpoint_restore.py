"""Restore/list/overwrite behaviour of CCE checkpoints and their server endpoints."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import nsa.server.proxy as proxy
from nsa.runtime.cce_checkpoint import CCECheckpointManager
from nsa.runtime.cce_persistent_state import PersistentCognitiveState


def _advanced_state(dim: int = 4) -> PersistentCognitiveState:
    state = PersistentCognitiveState(dimension=dim, decay=0.05, learning_rate=0.4)
    state.observe(torch.ones(dim) * 0.5, dt=2.5)
    state.observe(torch.ones(dim) * 0.1, dt=1.5)
    return state


def _runtime(ckpt_dir, dim: int = 4):
    runtime = proxy.NSAProxyRuntime.__new__(proxy.NSAProxyRuntime)
    runtime.enable_cce = True
    runtime.cce_dimension = dim
    runtime.cce_state = PersistentCognitiveState(dimension=dim, decay=0.05, learning_rate=0.4)
    runtime.checkpoint_mgr = CCECheckpointManager(ckpt_dir)
    runtime._last_cce_tick = 0.0
    return runtime


@pytest.fixture
def served(tmp_path):
    runtime = _runtime(tmp_path / "ckpts")
    handler = type("Handler", (proxy.NSAHTTPHandler,), {"runtime": runtime})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{httpd.server_address[1]}", runtime=runtime)
    httpd.shutdown()
    httpd.server_close()


def _request(url, *, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"{}")


def test_save_refuses_to_overwrite_unless_requested(tmp_path):
    manager = CCECheckpointManager(tmp_path)
    manager.save_persistent_state(PersistentCognitiveState(dimension=4), checkpoint_id="keep")
    with pytest.raises(FileExistsError):
        manager.save_persistent_state(_advanced_state(), checkpoint_id="keep")
    assert manager.load_persistent_state("keep").snapshot().update_count == 0
    manager.save_persistent_state(_advanced_state(), checkpoint_id="keep", overwrite=True)
    assert manager.load_persistent_state("keep").snapshot().update_count == 2


def test_fork_refuses_existing_target_without_writing_a_parent_snapshot(tmp_path):
    manager = CCECheckpointManager(tmp_path)
    manager.save_persistent_state(PersistentCognitiveState(dimension=4), checkpoint_id="taken")
    with pytest.raises(FileExistsError):
        manager.fork_persistent_state(_advanced_state(), "taken")
    assert [p.name for p in tmp_path.iterdir()] == ["taken.json"]


def test_checkpoint_dir_is_configurable_via_env_and_absolute(tmp_path, monkeypatch):
    monkeypatch.setenv("NSA_CCE_CHECKPOINT_DIR", str(tmp_path / "from_env"))
    manager = CCECheckpointManager()
    assert manager.checkpoint_dir == (tmp_path / "from_env").resolve() and manager.checkpoint_dir.is_dir()
    monkeypatch.delenv("NSA_CCE_CHECKPOINT_DIR")
    monkeypatch.chdir(tmp_path)
    assert CCECheckpointManager("rel").checkpoint_dir.is_absolute()


def test_load_rejects_tensors_that_do_not_match_declared_dimension(tmp_path):
    manager = CCECheckpointManager(tmp_path)
    path = manager.save_persistent_state(_advanced_state(), checkpoint_id="bad")
    doc = json.loads(path.read_text())
    doc["metadata"]["dimension"] = 7  # digest covers only "state", so integrity still passes
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="Dimension Mismatch.*dimension 7.*'working' has 4"):
        manager.load_persistent_state("bad")


def test_runtime_restore_resumes_elapsed_and_update_count(tmp_path):
    source = _advanced_state()
    CCECheckpointManager(tmp_path).save_persistent_state(source, checkpoint_id="resume")
    runtime = _runtime(tmp_path)
    feedback_ref = runtime.cce_state  # in-place restore keeps references held elsewhere valid
    result = runtime.restore_checkpoint("resume")
    assert result["elapsed_seconds"] == pytest.approx(4.0) and result["update_count"] == 2
    assert runtime.cce_state is feedback_ref
    assert torch.allclose(runtime.cce_state.snapshot().working, source.snapshot().working)


def test_runtime_restore_rejects_dimension_mismatch_and_leaves_state_untouched(tmp_path):
    CCECheckpointManager(tmp_path).save_persistent_state(_advanced_state(dim=3), checkpoint_id="dim3")
    runtime = _runtime(tmp_path, dim=4)
    with pytest.raises(ValueError, match="dimension 3.*dimension 4"):
        runtime.restore_checkpoint("dim3")
    assert runtime.cce_state.snapshot().update_count == 0


def test_server_started_with_restore_checkpoint_resumes_state(tmp_path):
    CCECheckpointManager(tmp_path).save_persistent_state(_advanced_state(), checkpoint_id="boot")
    with patch("nsa.server.proxy.OllamaInferenceBackend") as backend_cls:
        backend_cls.return_value = MagicMock(model_name="qwen2.5:3b")
        runtime = proxy.NSAProxyRuntime(
            backend_type="ollama", model="qwen2.5:3b", enable_cce=True,
            restore_checkpoint="boot", checkpoint_dir=str(tmp_path),
        )
        try:
            snap = runtime.cce_state.snapshot()
            assert snap.update_count >= 2 and snap.elapsed_seconds >= 4.0
            with pytest.raises(FileNotFoundError):
                proxy.NSAProxyRuntime(
                    backend_type="ollama", model="qwen2.5:3b", enable_cce=True,
                    restore_checkpoint="missing", checkpoint_dir=str(tmp_path),
                )
        finally:
            runtime.close()


def test_list_and_restore_endpoints(served):
    served.runtime.cce_state.observe(torch.ones(4), dt=3.0)
    assert _request(served.url + "/api/cce/checkpoint", method="POST", body={"checkpoint_id": "a"})[0] == 200
    status, body = _request(served.url + "/api/cce/checkpoints")
    assert status == 200 and body["count"] == 1
    assert body["checkpoints"][0]["checkpoint_id"] == "a" and "file" not in body["checkpoints"][0]

    served.runtime.cce_state.observe(torch.zeros(4), dt=1.0)
    status, body = _request(served.url + "/api/cce/restore", method="POST", body={"checkpoint_id": "a"})
    assert status == 200 and body["update_count"] == 1 and body["elapsed_seconds"] == pytest.approx(3.0)
    assert served.runtime.cce_state.snapshot().update_count == 1


def test_checkpoint_endpoint_conflicts_unless_overwrite_true(served):
    url = served.url + "/api/cce/checkpoint"
    assert _request(url, method="POST", body={"checkpoint_id": "dup"})[0] == 200
    status, body = _request(url, method="POST", body={"checkpoint_id": "dup"})
    assert status == 409 and "already exists" in body["error"]
    assert _request(url, method="POST", body={"checkpoint_id": "dup", "overwrite": "yes"})[0] == 409
    assert _request(url, method="POST", body={"checkpoint_id": "dup", "overwrite": True})[0] == 200


def test_restore_endpoint_errors(served, tmp_path):
    url = served.url + "/api/cce/restore"
    assert _request(url, method="POST", body={"checkpoint_id": "nope"})[0] == 404
    assert _request(url, method="POST", body={"checkpoint_id": "../../x"})[0] == 400
    assert _request(url, method="POST", body={})[0] == 400
    # An id ending in .json must still be looked up inside the checkpoint dir, not as a path.
    assert _request(url, method="POST", body={"checkpoint_id": "x.json"})[0] == 404

    CCECheckpointManager(served.runtime.checkpoint_mgr.checkpoint_dir).save_persistent_state(
        _advanced_state(dim=3), checkpoint_id="dim3"
    )
    status, body = _request(url, method="POST", body={"checkpoint_id": "dim3"})
    assert status == 400 and "dimension 3" in body["error"]


def test_list_and_restore_require_the_token_like_other_endpoints(served, monkeypatch):
    monkeypatch.setenv("NSA_API_TOKEN", "s3cret")
    assert _request(served.url + "/api/cce/checkpoints")[0] == 401
    assert _request(served.url + "/api/cce/restore", method="POST", body={"checkpoint_id": "a"})[0] == 401
    auth = {"Authorization": "Bearer s3cret"}
    assert _request(served.url + "/api/cce/checkpoints", headers=auth)[0] == 200
    assert _request(served.url + "/api/cce/restore", method="POST", body={"checkpoint_id": "a"}, headers=auth)[0] == 404
