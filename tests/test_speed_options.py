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
