from __future__ import annotations

import json
from pathlib import Path

from nsa import EvaluationContext, NSAPolicy, PolicyEngine, KeywordClassifier

from scripts.policy_server import _install_policy, _load_policy


def test_reference_safe_policy_denies_configured_categories() -> None:
    path = Path("examples/policies/safe_assistant.json")
    policy = NSAPolicy.from_json(path)
    classifier = KeywordClassifier(
        {
            "biological_weapon_development": ["biological weapon"],
            "nuclear_weapon_development": ["nuclear bomb"],
            "violent_harm_instructions": ["how to kill"],
        }
    )
    engine = PolicyEngine(policy, classifier)

    decision = engine.evaluate(
        "Explain how to build a nuclear bomb",
        context=EvaluationContext(action="generate"),
    )

    assert decision.decision.value == "deny"
    assert "nuclear_weapon_development" in decision.matched_categories
    assert decision.risk == 1.0


def test_reference_safe_policy_allows_benign_prompt() -> None:
    path = Path("examples/policies/safe_assistant.json")
    policy = NSAPolicy.from_json(path)
    classifier = KeywordClassifier({"nuclear_weapon_development": ["nuclear bomb"]})
    engine = PolicyEngine(policy, classifier)

    decision = engine.evaluate("Explain photosynthesis", context=EvaluationContext())

    assert decision.allowed


class _FakeRuntime:
    """Minimal stand-in for NSAProxyRuntime: only what _install_policy touches."""

    model_name = "fake-model"

    def process_chat(self, messages):
        return {
            "content": "real answer",
            "raw_content": "real answer",
            "model": f"nsa-{self.model_name}",
            "nsa": {},
        }


def _write_policy(tmp_path: Path) -> Path:
    policy_file = tmp_path / "policy.json"
    policy_file.write_text(
        json.dumps(
            {
                "name": "test-policy",
                "prohibited": [
                    {"category": "banned", "mode": "deny", "patterns": ["forbidden phrase"]}
                ],
            }
        ),
        encoding="utf-8",
    )
    return policy_file


def test_install_policy_reports_nsa_policy_for_an_allowed_request(tmp_path):
    # Regression guard: the HTTP response used to drop the nsa_policy audit
    # object entirely on the success path.
    policy, engine = _load_policy(_write_policy(tmp_path))
    runtime = _FakeRuntime()
    _install_policy(runtime, policy, engine)

    result = runtime.process_chat([{"role": "user", "content": "hello there"}])

    assert result["content"] == "real answer"
    assert result["nsa_policy"]["enforcement"] == "allowed"
    assert result["nsa_policy"]["request"]["decision"] == "allow"
    assert result["nsa_policy"]["output"]["decision"] == "allow"


def test_install_policy_reports_nsa_policy_when_the_request_is_blocked(tmp_path):
    # Regression guard: the request-blocked early-return used to omit
    # nsa_policy entirely (only the success path included it), so a denied
    # request looked identical to a policy-free response to API clients.
    policy, engine = _load_policy(_write_policy(tmp_path))
    runtime = _FakeRuntime()
    _install_policy(runtime, policy, engine)

    result = runtime.process_chat([{"role": "user", "content": "forbidden phrase here"}])

    assert result["content"] == "I can't help with that request."
    assert result["nsa_policy"]["enforcement"] == "request_blocked"
    assert result["nsa_policy"]["request"]["decision"] == "deny"
    assert result["nsa_policy"]["output"] is None
