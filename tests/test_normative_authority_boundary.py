from nsa.normative import NormativeAssessment, NormativeState
from nsa.normative_policy import NormativeAction, NormativePolicy


def test_harmful_normative_state_cannot_grant_continue():
    assessment = NormativeAssessment(NormativeState({"harm": 1.0}, 1.0))
    action = NormativePolicy(deny_harmful=True).evaluate(assessment)
    assert action is NormativeAction.DENY


def test_uncertainty_defaults_to_escalation():
    assessment = NormativeAssessment(NormativeState({}, 0.0))
    assert NormativePolicy().evaluate(assessment) is NormativeAction.ESCALATE
