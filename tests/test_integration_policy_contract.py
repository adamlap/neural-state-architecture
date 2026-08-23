from nsa.policy import NSAPolicy, PolicyEngine, SecurityDecision


def test_policy_is_single_source_of_truth():
    policy = NSAPolicy.from_dict(
        {
            "name": "integration-test",
            "rules": [
                {
                    "category": "dangerous_request",
                    "mode": "deny",
                    "patterns": ["forbidden operation"],
                }
            ],
        }
    )
    engine = PolicyEngine(policy)

    decision = engine.evaluate("Please explain a forbidden operation")

    assert decision.decision == SecurityDecision.DENY
    assert "dangerous_request" in decision.matched_categories


def test_safe_request_is_allowed():
    policy = NSAPolicy.from_dict(
        {
            "name": "integration-test",
            "rules": [
                {
                    "category": "dangerous_request",
                    "mode": "deny",
                    "patterns": ["forbidden operation"],
                }
            ],
        }
    )

    decision = PolicyEngine(policy).evaluate("Explain how a compiler works")

    assert decision.decision == SecurityDecision.ALLOW
