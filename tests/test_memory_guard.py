import pytest

from experiments.research_common import memory_guard as mg


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch):
    monkeypatch.delenv("NSA_BENCH_NO_GUARD", raising=False)
    monkeypatch.setattr(mg, "wait_until_unloaded", lambda *a, **k: True)       # never poll a real server from a unit test


def _meminfo(tmp_path, kib):
    p = tmp_path / "meminfo"
    p.write_text(f"MemTotal: 9000000 kB\nMemAvailable: {kib} kB\n")
    return str(p)


def test_kv_estimate_grows_with_context_and_large_models_cost_more():
    gib = 2**30
    assert mg.estimate_need_bytes(int(4.7 * gib), 32768) > mg.estimate_need_bytes(int(4.7 * gib), 2048)
    assert mg.estimate_need_bytes(int(4.7 * gib), 8192) - int(4.7 * gib) > mg.estimate_need_bytes(int(1 * gib), 8192) - int(1 * gib)


def test_refuses_a_model_that_does_not_fit_and_allows_one_that_does(tmp_path, monkeypatch):
    gib = 2**30
    monkeypatch.setattr(mg, "unload_all", lambda url: ["old:model"])
    monkeypatch.setattr(mg, "model_weight_bytes", lambda url, model: int(4.7 * gib))
    low = _meminfo(tmp_path, int(5.5 * gib / 1024))
    with pytest.raises(SystemExit) as exc:
        mg.prepare_model("http://localhost:11434", "qwen2.5:7b", 32768, meminfo_path=low)
    assert "refusing to load" in str(exc.value)
    ok = mg.prepare_model("http://localhost:11434", "qwen2.5:7b", 2048, meminfo_path=_meminfo(tmp_path, int(9 * gib / 1024)))
    assert ok["unloaded"] == ["old:model"] and ok["need"] < 9 * gib


def test_does_not_block_when_it_cannot_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(mg, "unload_all", lambda url: [])
    monkeypatch.setattr(mg, "model_weight_bytes", lambda url, model: None)
    assert mg.prepare_model("http://localhost:11434", "x", 32768, meminfo_path=_meminfo(tmp_path, 1))["need"] is None


def test_remote_servers_are_never_blocked_on_local_memory(tmp_path, monkeypatch):
    gib = 2**30
    monkeypatch.setattr(mg, "unload_all", lambda url: [])
    monkeypatch.setattr(mg, "model_weight_bytes", lambda url, model: 40 * gib)
    assert mg.prepare_model("http://gpu-box:11434", "big", 32768, meminfo_path=_meminfo(tmp_path, 1024))["need"] > 0


def test_the_measured_7b_run_would_have_been_refused_on_an_8_gib_vm(tmp_path, monkeypatch):
    """Regression: the 7B at num_ctx 8192 used 6.86 GB resident and exhausted a 7.9 GiB WSL VM with ~6.8 GiB 'available'."""
    gib = 2**30
    monkeypatch.setattr(mg, "unload_all", lambda url: [])
    monkeypatch.setattr(mg, "model_weight_bytes", lambda url, model: int(4.36 * gib))
    with pytest.raises(SystemExit):
        mg.prepare_model("http://localhost:11434", "qwen2.5:7b", 8192, meminfo_path=_meminfo(tmp_path, int(6.8 * gib / 1024)))
    assert mg.estimate_need_bytes(int(4.36 * gib), 8192) > 6.86e9      # never below what was measured


def test_guard_waits_for_the_unload_to_finish_before_measuring(tmp_path, monkeypatch):
    """Regression: the 7B suite was refused because available memory was read before the previous model had been freed."""
    gib = 2**30
    calls = {"n": 0}

    def fake_get(url, payload=None, timeout=10.0):
        if url.endswith("/api/ps"):
            calls["n"] += 1
            return {"models": [{"name": "old"}]} if calls["n"] < 3 else {"models": []}
        return {}

    monkeypatch.setattr(mg, "_get", fake_get)
    monkeypatch.undo()                                                          # restore the real wait for this test only
    monkeypatch.setattr(mg, "_get", fake_get)
    monkeypatch.delenv("NSA_BENCH_NO_GUARD", raising=False)
    monkeypatch.setattr(mg, "model_weight_bytes", lambda url, model: int(1 * gib))
    monkeypatch.setattr("time.sleep", lambda s: None)
    info = mg.prepare_model("http://localhost:11434", "m", 2048, meminfo_path=_meminfo(tmp_path, int(9 * gib / 1024)))
    assert calls["n"] >= 3 and info["unloaded"] == ["old"]
