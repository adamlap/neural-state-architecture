"""Opt-in speed options: compact memory rows, pinned (prefix-cacheable) memory, backend request options."""
from nsa import NSA, RuntimeConfig
from nsa.memory.extract import RuleFactExtractor
from nsa.runtime.inference.ollama import OllamaInferenceBackend


class Recorder:
    model = "rec"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return "ok"


def agent(**cfg):
    backend = Recorder()
    a = NSA(backend, config=RuntimeConfig(memory_enabled=True, history_limit=0, include_state_in_prompt=False, **cfg), extractor=RuleFactExtractor())
    for line in ("The staging database port is 5432.", "The billing admin email is ops@example.com.", "The search api region is eu-west."):
        a.observe(line, source="chat")
    return a, backend


def test_default_render_unchanged_and_compact_drops_metadata():
    a, b = agent()
    a.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    assert "STATUS=CURRENT" in b.prompts[-1]
    a, b = agent(compact_memory=True)
    a.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    assert "STATUS=" not in b.prompts[-1] and "VERSION=" not in b.prompts[-1] and "5432" in b.prompts[-1]


def test_pinned_prefix_is_identical_across_questions_and_append_only():
    a, b = agent(pin_memory_prefix=True)
    a.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    a.step("What is the search api region?", auto_memory=True, remember_prompt=False)
    first, second = (p.rsplit("\n\n", 1)[0] for p in b.prompts[-2:])
    assert first == second and all(v in first for v in ("5432", "ops@example.com", "eu-west"))
    a.observe("The reports cache size is 64.", source="chat")
    a.step("What is the reports cache size?", auto_memory=True, remember_prompt=False)
    third = b.prompts[-1].rsplit("\n\n", 1)[0]
    assert third.startswith(first) and "64" in third          # a new fact extends the prefix instead of reordering it


def test_backend_thread_default_and_auto(monkeypatch):
    monkeypatch.delenv("NSA_NUM_THREAD", raising=False)
    assert OllamaInferenceBackend(model="m", mode="mock").num_thread is None
    assert OllamaInferenceBackend(model="m", mode="mock", num_thread=4).num_thread == 4
    assert (OllamaInferenceBackend(model="m", mode="mock", num_thread="auto").num_thread or 0) >= 1
    monkeypatch.setenv("NSA_NUM_THREAD", "3")
    assert OllamaInferenceBackend(model="m", mode="mock").num_thread == 3


def test_autotune_picks_options_by_model_size_and_respects_explicit_settings():
    from nsa.autotune import auto_config, detect_model_size

    class B:
        def __init__(self, name):
            self.model = name
            self.num_thread = None

        def generate(self, prompt, **kw):
            return "ok"

    assert detect_model_size(B("qwen2.5:7b")) == 7.0 and detect_model_size(B("Qwen/Qwen2.5-0.5B-Instruct")) == 0.5 and detect_model_size(B("mystery")) is None
    big, small, unknown = B("qwen2.5:7b"), B("qwen2.5:1.5b"), B("mystery")
    c = auto_config(big)
    assert c.pin_memory_prefix and c.retrieve_lines == 0 and c.provenance_window == 256 and c.reuse_familiar_answers and big.num_thread
    c = auto_config(small)
    assert not c.pin_memory_prefix and c.retrieve_lines == 3
    assert not auto_config(unknown).pin_memory_prefix                                     # unknown size: conservative
    assert auto_config(small, RuntimeConfig(retrieve_lines=1, provenance_window=32)).retrieve_lines == 1
    a = NSA(B("qwen2.5:7b"), config=RuntimeConfig(memory_enabled=True, auto_tune=True))
    assert a.config.pin_memory_prefix and a.config.provenance_window == 256


def test_pinned_prefix_falls_back_to_retrieval_when_memory_outgrows_the_cap():
    a, b = agent(pin_memory_prefix=True, pin_memory_max=2)                               # 3 facts > cap
    a.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    assert "5432" in b.prompts[-1] and "ops@example.com" not in b.prompts[-1]


def test_answer_from_memory_only_for_exact_single_key_lookups():
    a, b = agent(answer_from_memory=True)
    r = a.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    assert r.text == "5432" and not b.prompts and a.call_stats["direct"] == 1               # no model call
    for q in ("Why is the staging database port 5432?", "What was the previous staging database port?", "What is the staging database port in hex?",
              "What is the database port?"):
        n = len(b.prompts)
        a.step(q, auto_memory=True, remember_prompt=False)
        assert len(b.prompts) == n + 1, q                                                   # anything beyond a plain one-key lookup goes to the model
    a.observe("The staging database port is now 6543.", source="chat")
    assert a.step("staging database port", auto_memory=True, remember_prompt=False).text == "6543"   # always the current value
    a2, b2 = agent()
    a2.step("What is the staging database port?", auto_memory=True, remember_prompt=False)
    assert len(b2.prompts) == 1                                                             # off by default
