import pytest

from nsa.normative import NormativeClass, NormativeState, NormativeAssessment
from nsa.normative_policy import NormativeAction, NormativePolicy
from nsa.semantic import ReferenceSemanticClassifier


def test_normative_state_is_bounded_and_uncertain():
    state = NormativeState({"harm": 0.8}, 0.9)
    assert state.dominant is NormativeClass.HARMFUL
    assert not NormativeAssessment(state).uncertain


def test_low_confidence_escalates():
    assessment = NormativeAssessment(NormativeState({"harm": 0.0}, 0.4))
    assert NormativePolicy().evaluate(assessment) is NormativeAction.ESCALATE


def test_harmful_assessment_denies():
    assessment = NormativeAssessment(NormativeState({"harm": 0.9}, 0.95))
    assert NormativePolicy().evaluate(assessment) is NormativeAction.DENY


def test_sensitive_assessment_requires_approval():
    assessment = NormativeAssessment(NormativeState({"sensitivity": 0.8}, 0.95))
    assert NormativePolicy().evaluate(assessment) is NormativeAction.REQUIRE_APPROVAL


def test_reference_semantic_classifier_is_replaceable_boundary():
    classifier = ReferenceSemanticClassifier(
        [("violent_harm", ["dangerous harm"]), ("protected", ["private data"])]
    )
    result = classifier.classify("request about dangerous harm")
    assert "violent_harm" in result.categories
    assert result.normative.state.dominant is NormativeClass.HARMFUL


def test_invalid_normative_state_rejected():
    with pytest.raises(ValueError):
        NormativeState({"harm": 1.2}, 0.9)
