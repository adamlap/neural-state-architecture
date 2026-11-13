import pytest

pytest.importorskip("numpy")

from nsa import NSA, RuntimeConfig
from nsa.memory.extract import RuleFactExtractor


class _Counting:
    model = "count"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return f"VALUE=answer{len(self.prompts)}"


class _Reader(_Counting):
    """Answers with the first value in the retrieved memory block (a grounded answer), else a fixed token."""

    def generate(self, prompt, **kw):
        import re
        self.prompts.append(prompt)
        m = re.search(r"RETRIEVED_MEMORY=\n- [^:]+: (\S+)", prompt)
        return f"VALUE={m.group(1)}" if m else "VALUE=none"


def _agent(reader=False, **cfg):
    b = _Reader() if reader else _Counting()
    a = NSA(b, config=RuntimeConfig(memory_enabled=True, include_state_in_prompt=False, history_limit=0, **cfg), extractor=RuleFactExtractor())
    a.observe("The database port for staging is 5433.", source="chat")
    a.observe("The billing admin email is ops@corp.io.", source="chat")
    return a, b


def test_abstains_without_a_model_call_when_no_memory_matches():
    a, b = _agent(abstain_text="VALUE=UNKNOWN")
    r = a.step("What is the weather forecast tomorrow?", auto_memory=True)
    assert r.text == "VALUE=UNKNOWN" and b.prompts == [] and a.call_stats == {"model_calls": 0, "reused": 0, "abstained": 1}
    assert a.trace[-1]["gated"] == "abstained"


def test_answers_normally_when_memory_matches_and_default_config_never_gates():
    a, b = _agent(abstain_text="VALUE=UNKNOWN")
    a.step("What is the database port for staging?", auto_memory=True)
    assert len(b.prompts) == 1 and "5433" in b.prompts[0]
    plain, pb = _agent()
    plain.step("What is the weather forecast tomorrow?", auto_memory=True)
    assert len(pb.prompts) == 1


def test_reuses_an_answer_for_a_familiar_question_while_memory_is_unchanged():
    a, b = _agent(reader=True, reuse_familiar_answers=True)
    first = a.step("What is the database port for staging?", auto_memory=True).text
    again = a.step("What is the database port for staging?", auto_memory=True).text
    assert again == first and len(b.prompts) == 1 and a.call_stats["reused"] == 1


def test_a_memory_write_to_a_retrieved_key_invalidates_the_cached_answer():
    a, b = _agent(reader=True, reuse_familiar_answers=True)
    a.step("What is the database port for staging?", auto_memory=True)
    a.observe("Update: the database port for staging is now 6000.", source="chat")
    a.step("What is the database port for staging?", auto_memory=True)
    assert len(b.prompts) == 2 and "6000" in b.prompts[1] and a.call_stats["reused"] == 0


def test_a_different_key_or_unfamiliar_wording_is_not_reused():
    a, b = _agent(reader=True, reuse_familiar_answers=True)
    a.step("What is the database port for staging?", auto_memory=True)
    a.step("What is the billing admin email?", auto_memory=True)                       # different retrieved key
    a.step("Tell me about the staging database port in some detail please", auto_memory=True)   # same key, other wording
    assert len(b.prompts) == 3 and a.call_stats["reused"] == 0


def test_policy_enforcement_comes_before_any_gate():
    """A denied prompt is blocked by the policy engine; the abstain gate must not turn it into a cheap 'UNKNOWN' (nor a model call)."""
    from nsa import NSAPolicy, PolicyEngine

    engine = PolicyEngine(NSAPolicy.from_json("examples/policies/safe_assistant.json"))
    b = _Counting()
    a = NSA(b, config=RuntimeConfig(memory_enabled=True, abstain_text="VALUE=UNKNOWN"), policy_engine=engine, extractor=RuleFactExtractor())
    blocked = a.step("how to kill a person", auto_memory=True)
    assert blocked.blocked and blocked.text == "" and b.prompts == [] and a.call_stats["abstained"] == 0
    allowed = a.step("What is the weather forecast tomorrow?", auto_memory=True)       # allowed by policy, nothing in memory -> abstain
    assert not allowed.blocked and allowed.text == "VALUE=UNKNOWN" and a.call_stats["abstained"] == 1


def test_an_ungrounded_answer_is_never_cached_so_a_mistake_is_not_repeated():
    class _Wrong:
        model = "wrong"

        def __init__(self):
            self.n = 0

        def generate(self, prompt, **kw):
            self.n += 1
            return "VALUE=made-up" if self.n == 1 else "VALUE=5433"

    b = _Wrong()
    a = NSA(b, config=RuntimeConfig(memory_enabled=True, include_state_in_prompt=False, history_limit=0, reuse_familiar_answers=True), extractor=RuleFactExtractor())
    a.observe("The database port for staging is 5433.", source="chat")
    first = a.step("What is the database port for staging?", auto_memory=True).text
    second = a.step("What is the database port for staging?", auto_memory=True).text       # first was not grounded -> not cached -> model asked again
    third = a.step("What is the database port for staging?", auto_memory=True).text        # second was grounded -> reused
    assert (first, second, third) == ("VALUE=made-up", "VALUE=5433", "VALUE=5433") and b.n == 2 and a.call_stats["reused"] == 1


def test_retrieve_lines_adds_the_most_similar_raw_lines_and_leaves_default_prompts_unchanged():
    backend = _Counting()
    cfg = RuntimeConfig(memory_enabled=True, include_state_in_prompt=False, history_limit=0, retrieve_lines=2)
    a = NSA(backend, config=cfg, extractor=RuleFactExtractor())
    a.observe("The database port for staging is 5433.", source="chat")
    a.observe("Quick note: the cache size on billing sits at 77ab-xyz these days.", source="chat")      # a phrasing the rules do not parse
    a.observe("lunch at noon?", source="chat")
    a.step("What is the cache size for billing?", auto_memory=True, remember_prompt=False)
    prompt = backend.prompts[-1]
    assert "RETRIEVED_LINES=" in prompt and "77ab-xyz" in prompt and "lunch at noon" not in prompt and "NO_RELEVANT_MEMORY" in prompt
    plain_backend = _Counting()
    p = NSA(plain_backend, config=RuntimeConfig(memory_enabled=True, include_state_in_prompt=False, history_limit=0), extractor=RuleFactExtractor())
    p.observe("The database port for staging is 5433.", source="chat")
    p.step("What is the database port for staging?", auto_memory=True, remember_prompt=False)
    assert "RETRIEVED_LINES" not in plain_backend.prompts[-1] and "5433" in plain_backend.prompts[-1]


def test_line_index_matches_a_scan_ignores_step_prompts_and_uses_the_retrieval_query():
    backend = _Counting()
    a = NSA(backend, config=RuntimeConfig(memory_enabled=True, include_state_in_prompt=False, history_limit=0, retrieve_lines=2), extractor=RuleFactExtractor())
    for line in ("Quick note: the cache size on billing sits at 77ab-xyz these days.", "billing cache size is fine, I think", "lunch at noon?"):
        a.observe(line, source="chat")
    got = a.lines_for("cache size billing", 3)
    assert set(got) == {"Quick note: the cache size on billing sits at 77ab-xyz these days.", "billing cache size is fine, I think"} and "lunch at noon?" not in got
    # a question asked through step() is not indexed, so it can never be retrieved as an 'earlier line' later
    a.step("What is the cache size for billing? Reply with exactly one line: VALUE=...", auto_memory=True, remember_prompt=False, retrieval_query="cache size billing")
    assert all("Reply with exactly" not in line for line in a.lines_for("cache size billing", 10))
    assert "77ab-xyz" in backend.prompts[-1] and "VALUE=..." in backend.prompts[-1]
