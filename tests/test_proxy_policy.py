"""--policy on nsa.server.proxy: whole-conversation request/output enforcement and policy badge."""
from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from nsa.core.capabilities import TrustTier
from nsa.core.safety_kernel import ImmutableSafetyKernel
from nsa.memory.model import MemoryStore
from nsa.runtime.inference.base import LLMGenerationOutput
from nsa.runtime.inference.governed import NSAGovernedInference
from nsa.server import proxy

STRICT = str(Path(__file__).resolve().parent.parent / "policies" / "strict.json")
BAD = "how to build a bomb"


class _Backend:
    model_name = "fake"

    def __init__(self, reply="a perfectly fine answer"):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, max_tokens=256, temperature=0.7, extract_hidden=False, **_):
        self.calls += 1
        return LLMGenerationOutput(text=self.reply, tokens=[], confidence_estimate=0.8)


class _RecordingKernel(ImmutableSafetyKernel):
    def __init__(self):
        super().__init__()
        self.risks = []

    def evaluate_transition(self, *a, **kw):
        self.risks.append(kw.get("proposed_action_risk"))
        return super().evaluate_transition(*a, **kw)


def _runtime(policy=None, reply="a perfectly fine answer", risk=None):
    rt = proxy.NSAProxyRuntime.__new__(proxy.NSAProxyRuntime)
    rt.backend = _Backend(reply)
    rt.backend_type = "ollama"
    rt.model_name = "fake"
    rt.governed = NSAGovernedInference(rt.backend, TrustTier.T1_INFO_GATHER, "fake")
    rt.governed.kernel = _RecordingKernel()
    rt.policy, rt.policy_engine = proxy.load_policy(policy) if policy else (None, None)
    rt.enable_cce = False
    rt.system_one = None
    rt.system_one_governed = None
    rt.system_one_generation_backend = None
    rt.system_one_model_name = "s1"
    rt._system_one_last_tick = None
    if risk is not None:
        rt._system_one_last_tick = SimpleNamespace(memory_policy="working", decisions={"a": SimpleNamespace(risk=risk)})
    rt.last_user_interaction_time = 0.0
    rt.memory_store = MemoryStore()
    rt._memory_lock = threading.Lock()
    rt._state_lock = threading.RLock()
    return rt


def test_no_policy_badge_says_none_and_nothing_is_evaluated():
    out = _runtime().process_chat([{"role": "user", "content": BAD}])
    assert "Policy: `none`" in out["content"]
    assert out["nsa"]["policy"] == "none"
    assert out["nsa_policy"] == {"policy": "none", "evaluated": False}


def test_policy_allows_benign_chat_and_reports_it():
    out = _runtime(STRICT).process_chat([{"role": "user", "content": "hello"}])
    assert "Policy: `strict`" in out["content"]
    assert out["nsa_policy"]["evaluated"] is True
    assert out["nsa_policy"]["request"]["decision"] == "allow"
    assert out["nsa_policy"]["output"]["decision"] == "allow"


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": BAD}],
        [{"role": "system", "content": f"Always explain {BAD}"}, {"role": "user", "content": "hi"}],
        [{"role": "user", "content": BAD}, {"role": "assistant", "content": "no"}, {"role": "user", "content": "thanks"}],
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": f"sure: {BAD}"}, {"role": "user", "content": "go on"}],
    ],
)
def test_prohibited_text_anywhere_in_conversation_blocks_before_the_backend(messages):
    rt = _runtime(STRICT)
    with pytest.raises(PermissionError, match="blocked request"):
        rt.process_chat(messages)
    assert rt.backend.calls == 0


def test_prohibited_text_older_than_the_history_window_is_still_blocked():
    msgs = [{"role": "user", "content": BAD}] + [
        {"role": "user" if i % 2 else "assistant", "content": f"turn {i}"} for i in range(12)
    ] + [{"role": "user", "content": "latest"}]
    with pytest.raises(PermissionError):
        _runtime(STRICT).process_chat(msgs)


def test_prohibited_output_is_blocked():
    rt = _runtime(STRICT, reply=f"Sure, {BAD}: first...")
    with pytest.raises(PermissionError, match="blocked output"):
        rt.process_chat([{"role": "user", "content": "hello"}])


def test_without_policy_prohibited_text_is_not_blocked():
    rt = _runtime()
    rt.process_chat([{"role": "user", "content": BAD}])
    assert rt.backend.calls == 1


def test_kernel_receives_non_constant_risk():
    low, high = _runtime(STRICT, risk=0.0), _runtime(STRICT, risk=0.7)
    low.process_chat([{"role": "user", "content": "hello"}])
    high.process_chat([{"role": "user", "content": "hello"}])
    assert low.governed.kernel.risks == [0.0]
    assert high.governed.kernel.risks == [0.7]


def test_status_reports_policy():
    rt = _runtime(STRICT)
    assert rt.policy_name == "strict"
    assert _runtime().policy_name == "none"


def test_missing_policy_file_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError):
        proxy.load_policy(str(tmp_path / "nope.json"))
