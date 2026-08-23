from nsa.policy import NSAPolicy
from nsa.enforcement import PolicyEngine
from nsa.decision import Decision


def _policy():
    return NSAPolicy.from_mapping(
        {
            "name": "integration-test",
            "prohibited": [
                {
                    "category": "dangerous_request",
                    "mode": "deny",
                    "patterns": ["forbidden operation"],
                }
            ],
        }
    )


def test_policy_is_single_source_of_truth():
    decision = PolicyEngine(_policy()).evaluate("Please explain a forbidden operation")
    assert decision.decision == Decision.DENY
    assert "dangerous_request" in decision.matched_categories


def test_safe_request_is_allowed():
    decision = PolicyEngine(_policy()).evaluate("Explain how a compiler works")
    assert decision.decision == Decision.ALLOW
