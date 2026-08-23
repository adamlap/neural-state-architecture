from __future__ import annotations

from pathlib import Path

from nsa import EvaluationContext, NSAPolicy, PolicyEngine, KeywordClassifier


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
