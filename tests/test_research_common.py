import pytest

from experiments.research_common.backends import FixedDecodingBackend
from experiments.research_common.episodes import facts, pool_size, target_occurrences
from experiments.research_common.grading import extract, lenient_correct, strict_correct
from experiments.research_common.stats import holm, mcnemar_exact, wilson


class TestEpisodePool:
    def test_unique_pool_never_wraps_onto_the_target_for_any_distractor_count(self):
        for d in (0, 1, 10, 63, 64, 65, 100, 250, 500, 1000, 2000):
            n = pool_size(d)
            assert n >= d + 16
            # distractor i uses pool index (i + 1) % n; the target is index 0
            assert all((i + 1) % n != 0 for i in range(d))
            # interference revisits use index (i + 9) % n for i < d // 2
            assert all((i + 9) % n != 0 for i in range(d // 2))

    def test_legacy_pool_reproduces_the_wraparound_bug_so_old_results_stay_explainable(self):
        assert pool_size(1000, "legacy") == 64
        assert any((i + 1) % 64 == 0 for i in range(100))  # d=100 already wraps onto the target

    def test_unknown_generator_is_rejected(self):
        with pytest.raises(ValueError):
            pool_size(10, "nope")

    def test_keys_and_values_are_unique_and_parseable_at_large_pool_sizes(self):
        pairs = facts(7, 1016)
        assert len({k for k, _ in pairs}) == 1016 and len({v for _, v in pairs}) == 1016
        assert pairs[0][1] == "V007_000"

    def test_target_occurrences_finds_every_mention(self):
        obs = ["FACT: ITEM_01 has value A.", "DISTRACTOR: ITEM_02 has value B.", "DISTRACTOR: ITEM_01 has value A."]
        assert target_occurrences(obs, "ITEM_01") == [0, 2]


class TestGrading:
    def test_strict_is_exact(self):
        assert strict_correct("VALUE=V007_000", "V007_000")
        assert strict_correct("value=v007_000", "V007_000")
        assert not strict_correct("VALUE=V007_001", "V007_000")
        assert not strict_correct("The value is V007_000", "V007_000")

    def test_lenient_accepts_echoed_metadata_but_not_wrong_values(self):
        for echoed in ("VALUE=V007_000_VERSION_2_STATUS_CURRENT", "VALUE=V007_000_VERSION=2_STATUS=CURRENT"):
            assert not strict_correct(echoed, "V007_000") and lenient_correct(echoed, "V007_000")
        assert not lenient_correct("VALUE=V007_001_VERSION_2_STATUS_CURRENT", "V007_000")
        assert not lenient_correct("VALUE=V007_0001", "V007_000")  # a different token, not an echo
        assert not lenient_correct("no answer", "V007_000")

    def test_multi_value_answers(self):
        assert strict_correct("VALUE=A_1|B_2|C_3", "A_1|B_2|C_3")
        assert lenient_correct("VALUE=A_1|B_2_VERSION_2|C_3", "A_1|B_2|C_3")
        assert not lenient_correct("VALUE=A_1|B_2", "A_1|B_2|C_3")
        assert not lenient_correct("VALUE=A_1|X_2|C_3", "A_1|B_2|C_3")

    def test_extract(self):
        assert extract("VALUE = ab_1") == "AB_1" and extract(None) is None


class _Inner:
    model = "m"

    def __init__(self, raw=None, error=None):
        self.kwargs, self.raw, self.error = None, raw or {}, error

    def generate(self, prompt, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error

        class Out:
            text = "VALUE=X"
            raw_response = self.raw
        return Out()


class TestFixedDecodingBackend:
    def test_forces_identical_decoding_and_records_the_real_prompt(self):
        inner = _Inner({"prompt_eval_count": 12, "eval_count": 3})
        backend = FixedDecodingBackend(inner, temperature=0.0, max_tokens=32, num_ctx=4096)
        backend.generate("PROMPT", state={"a": 1}, temperature=0.9, max_tokens=999)  # caller overrides are ignored
        assert inner.kwargs == {"max_tokens": 32, "temperature": 0.0, "num_ctx": 4096, "state": {"a": 1}}
        assert backend.last.prompt == "PROMPT" and backend.last.prompt_tokens == 12 and not backend.last.truncated

    def test_flags_truncation_when_the_prompt_fills_the_context(self):
        backend = FixedDecodingBackend(_Inner({"prompt_eval_count": 4096}), num_ctx=4096)
        backend.generate("x")
        assert backend.last.truncated

    def test_timeouts_are_recorded_not_raised_but_other_errors_propagate(self):
        backend = FixedDecodingBackend(_Inner(error=RuntimeError("request timed out")))
        assert backend.generate("x").text == "" and backend.last.timed_out
        with pytest.raises(RuntimeError):
            FixedDecodingBackend(_Inner(error=RuntimeError("model not found"))).generate("x")


class TestStats:
    def test_wilson_bounds(self):
        lo, hi = wilson(0, 72)
        assert lo == 0.0 and 0.04 < hi < 0.06
        assert wilson(0, 0) == (0.0, 0.0)

    def test_mcnemar_matches_the_known_exact_value(self):
        a = {i: True for i in range(15)}
        b = {i: False for i in range(15)}
        result = mcnemar_exact(a, b)
        assert (result["only_first"], result["only_second"]) == (15, 0)
        assert result["p"] == pytest.approx(2 / 2 ** 15)
        assert mcnemar_exact({1: True}, {1: True})["p"] == 1.0

    def test_holm_is_monotone_and_never_below_the_raw_p(self):
        adjusted = holm([0.01, 0.04, 0.03])
        assert adjusted == [pytest.approx(0.03), pytest.approx(0.06), pytest.approx(0.06)]


def test_ollama_num_ctx_is_sent_explicitly(monkeypatch):
    import json as _json
    import urllib.request
    from nsa.runtime.inference.ollama import BackendMode, OllamaInferenceBackend

    sent = []

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"response": "VALUE=X"}'

    def fake_urlopen(req, timeout=None):
        sent.append(_json.loads(req.data.decode("utf-8")))
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = object.__new__(OllamaInferenceBackend)
    backend.mode, backend.model_name, backend.base_url, backend.timeout_sec = BackendMode.OLLAMA, "m", "http://x", 5.0
    backend.generate("short", num_ctx=32768)
    backend.generate("x" * 5000, num_ctx=32768)          # long prompt must not flip the size
    backend.generate("x" * 5000)                         # legacy heuristic unchanged
    assert [p["options"].get("num_ctx") for p in sent] == [32768, 32768, 8192]
