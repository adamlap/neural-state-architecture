import pytest

from nsa.decision import Decision
from nsa.enforcement import PolicyEngine
from nsa.policy import NSAPolicy


def _policy(**extra):
    return NSAPolicy.from_mapping({
        "name": "t",
        "prohibited": [
            {"category": "soft", "mode": "escalate", "patterns": ["borderline"]},
            {"category": "hard", "mode": "deny", "patterns": ["forbidden"]},
        ],
        **extra,
    })


def test_deny_dominates_an_earlier_escalate():
    """Regression: the first matched rule used to win, so an escalate masked a deny."""
    engine = PolicyEngine(_policy())
    decision = engine.evaluate("this is borderline and also forbidden")
    assert decision.decision == Decision.DENY
    assert set(decision.matched_categories) == {"soft", "hard"}


def test_escalate_still_applies_when_nothing_stricter_matches():
    assert PolicyEngine(_policy()).evaluate("merely borderline").decision == Decision.ESCALATE


class _FixedClassifier:
    def __init__(self, *categories):
        self.categories = categories

    def classify(self, text):
        return self.categories


@pytest.mark.parametrize("unknown, expected", [("deny", Decision.DENY), ("escalate", Decision.ESCALATE), ("allow", Decision.ALLOW)])
def test_unknown_categories_follow_unknown_policy(unknown, expected):
    engine = PolicyEngine(_policy(unknown_policy=unknown), _FixedClassifier("novel_category"))
    assert engine.evaluate("anything").decision == expected


def test_known_deny_beats_unknown_allow():
    engine = PolicyEngine(_policy(unknown_policy="allow"), _FixedClassifier("novel_category", "hard"))
    assert engine.evaluate("x").decision == Decision.DENY


@pytest.mark.parametrize("text", [
    "FORBIDDEN", "for\u200bbidden", "forbidden\u200d",
    "\uff46\uff4f\uff52\uff42\uff49\uff44\uff44\uff45\uff4e",  # full-width letters
    "a  forbidden\tthing",
])
def test_classifier_normalises_case_width_and_invisible_characters(text):
    engine = PolicyEngine(_policy())
    assert engine.evaluate(text).decision == Decision.DENY


def test_multiword_patterns_match_across_whitespace_variants():
    policy = NSAPolicy.from_mapping({"prohibited": [{"category": "k", "patterns": ["how to kill"]}]})
    engine = PolicyEngine(policy)
    for text in ("how  to\tkill", "How\nTo Kill", "HOW TO KILL"):
        assert engine.evaluate(text).decision == Decision.DENY


@pytest.mark.parametrize("bad", [
    {"protected_data": "credentials"},          # a bare string would become a set of characters
    {"restricted_actions": "filesystem_write"},
    {"require_approval": 7},
    {"prohibited": {"category": "x"}},          # not a list: rules would be silently dropped
    {"prohibited": [42]},
    {"prohibited": [{"mode": "deny"}]},         # missing category
    {"prohibited": [{"category": "x", "patterns": "abc"}]},
])
def test_malformed_policy_fails_loudly_instead_of_silently_weakening(bad):
    with pytest.raises(ValueError):
        NSAPolicy.from_mapping(bad)


def test_unknown_policy_keys_warn():
    with pytest.warns(UserWarning, match="protected_datas"):
        NSAPolicy.from_mapping({"protected_datas": ["x"]})


def test_empty_yaml_or_json_document_is_rejected(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("null")
    with pytest.raises(ValueError):
        NSAPolicy.from_json(path)


def test_shipped_example_policies_still_load_without_warnings():
    import warnings
    from pathlib import Path
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for path in list(Path("examples/policies").glob("*.json")) + list(Path("policies").glob("*.json")):
            NSAPolicy.from_json(path)
