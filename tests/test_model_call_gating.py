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


def _agent(**cfg):
    b = _Counting()
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
    a, b = _agent(reuse_familiar_answers=True)
    first = a.step("What is the database port for staging?", auto_memory=True).text
    again = a.step("What is the database port for staging?", auto_memory=True).text
    assert again == first and len(b.prompts) == 1 and a.call_stats["reused"] == 1


def test_a_memory_write_to_a_retrieved_key_invalidates_the_cached_answer():
    a, b = _agent(reuse_familiar_answers=True)
    a.step("What is the database port for staging?", auto_memory=True)
    a.observe("Update: the database port for staging is now 6000.", source="chat")
    a.step("What is the database port for staging?", auto_memory=True)
    assert len(b.prompts) == 2 and "6000" in b.prompts[1] and a.call_stats["reused"] == 0


def test_a_different_key_or_unfamiliar_wording_is_not_reused():
    a, b = _agent(reuse_familiar_answers=True)
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
