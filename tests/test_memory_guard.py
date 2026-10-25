import pytest

from experiments.research_common import memory_guard as mg


@pytest.fixture(autouse=True)
def _guard_on(monkeypatch):
    monkeypatch.delenv("NSA_BENCH_NO_GUARD", raising=False)


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
