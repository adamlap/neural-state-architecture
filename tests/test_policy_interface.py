from nsa.adapters import PolicyViolation, protect_model
from nsa.decision import Decision
from nsa.enforcement import EvaluationContext, KeywordClassifier, PolicyEngine
from nsa.policy import NSAPolicy


def make_engine():
    policy = NSAPolicy.from_mapping({
        "name": "reference-safe",
        "prohibited": ["restricted_harm_category"],
        "protected_data": ["credentials"],
        "restricted_actions": ["filesystem_write"],
        "require_approval": ["external_side_effect"],
    })
    classifier = KeywordClassifier({"restricted_harm_category": ["restricted-demo-marker"]})
    return PolicyEngine(policy, classifier)


def test_policy_denies_prohibited_semantic_category():
    decision = make_engine().evaluate("contains restricted-demo-marker")
    assert decision.decision == Decision.DENY
    assert "restricted_harm_category" in decision.matched_categories
    assert decision.hard_constraints_triggered


def test_policy_requires_approval_for_configured_action():
    engine = make_engine()
    decision = engine.evaluate("ordinary request", context=EvaluationContext(action="external_side_effect"))
    assert decision.decision == Decision.REQUIRE_APPROVAL


def test_policy_denies_unauthorized_capability():
    engine = make_engine()
    decision = engine.evaluate("ordinary request", context=EvaluationContext(action="generate", capabilities=frozenset({"filesystem_write"})))
    assert decision.decision == Decision.DENY
    assert "filesystem_write" in decision.required_capabilities


def test_protected_model_checks_request_and_output():
    engine = make_engine()
    model = protect_model(lambda prompt: "safe response", engine)
    assert model.generate("ordinary request") == "safe response"


def test_protected_model_fails_closed():
    engine = make_engine()
    model = protect_model(lambda prompt: "should not run", engine)
    try:
        model.generate("restricted-demo-marker")
    except PolicyViolation as exc:
        assert exc.decision.decision == Decision.DENY
    else:
        raise AssertionError("expected PolicyViolation")
