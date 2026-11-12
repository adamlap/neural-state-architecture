import json

from experiments.speed import ollama_flags as flags
from experiments.speed import remote_speed as rs


def test_prompt_prefix_is_unique_per_call_and_length_scales():
    a, b = flags.prompt_of(500, 1), flags.prompt_of(500, 2)
    assert a[:20] != b[:20] and len(flags.prompt_of(1000, 1)) > 1.8 * len(flags.prompt_of(500, 1))


def test_remote_speed_uses_measure_and_reports_prefill_seconds(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "get", lambda url, timeout=10.0: {"version": "9.9"})
    monkeypatch.setattr(rs, "measure", lambda base, model, tokens, threads, repeats: {
        "prefill_tok_s": 100.0, "decode_tok_s": 20.0, "prompt_tokens": tokens, "resident_mib": 1000.0})
    out = tmp_path / "o.json"
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "--ollama-url", "http://h:11434", "--models", "m", "--tokens", "100", "1000", "--label", "t", "--out", str(out)])
    rs.main()
    data = json.loads(out.read_text())
    assert data["ollama_version"] == "9.9" and [r["prefill_seconds"] for r in data["rows"]] == [1.0, 10.0]
