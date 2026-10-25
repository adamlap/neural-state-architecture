from types import SimpleNamespace

from experiments.research_v8 import resource_bounded_benchmark as v8
from experiments.research_v3.temporal_memory_benchmark import make_episode


def test_prompt_sizes_retrieval_is_flat_and_transcript_grows():
    sizes = {}
    for d in (10, 1000):
        obs, q, *_ = make_episode(7, "supersession_current", d, "unique", "plain")
        sizes[d] = {c: len(v8.build_prompt(c, obs, q)[0]) for c in v8.CONDITIONS}
    assert sizes[1000]["transcript"] > 20 * sizes[10]["transcript"]
    for c in ("rag_lines", "temporal_memory"):
        assert sizes[1000][c] - sizes[10][c] < 50


def test_rag_lines_returns_the_last_three_matching_lines_and_no_others():
    obs, q, _, key, _ = make_episode(7, "supersession_current", 50, "unique", "plain")
    prompt, n = v8.build_prompt("rag_lines", obs, q)
    body = prompt.split("\n\n")[0].splitlines()[1:]
    assert n == len(body) == 2 and all(key in line for line in body)


def test_temporal_memory_prompt_has_only_the_current_value():
    obs, q, expected, key, _ = make_episode(7, "supersession_current", 50, "unique", "plain")
    prompt, n = v8.build_prompt("temporal_memory", obs, q)
    assert n == 1 and expected in prompt and "STATUS=CURRENT" in prompt


class _Fake:
    model = "fake"

    def generate(self, prompt, **kw):
        return SimpleNamespace(text="VALUE=NOPE", raw_response={"prompt_eval_count": 7, "prompt_eval_duration": 3_000_000,
                                                                 "eval_duration": 1_000_000, "load_duration": 0})


def test_run_records_timings_skips_oversize_transcripts_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(v8, "OllamaInferenceBackend", lambda **kw: _Fake())
    monkeypatch.setattr(v8, "loaded_model_bytes", lambda *a: 123)
    monkeypatch.setattr(v8, "server_rss_bytes", lambda: 456)
    args = SimpleNamespace(models=["fake"], seeds=[7], num_ctx=[2048, 4096], distractors=[5, 400], trials=1,
                           transcript_token_cap=1000, timeout=1.0, out=str(tmp_path), ollama_url=None, summarize_only=False)
    manifest = v8.run(args)
    assert manifest["episodes_recorded"] == 2 * 2 * len(v8.TASKS) * len(v8.CONDITIONS)
    cell = manifest["by_model_ctx_condition"]["fake|2048|transcript"]
    assert cell["n_skipped"] == len(v8.TASKS) and cell["n_run"] == len(v8.TASKS)       # d=400 skipped, d=5 run
    mem = manifest["by_model_ctx_condition"]["fake|2048|temporal_memory"]
    assert mem["prompt_eval_ms_mean"] == 3.0 and mem["resident_mib_mean"] == 123 / 2**20
    assert v8.run(args)["episodes_recorded"] == manifest["episodes_recorded"]
