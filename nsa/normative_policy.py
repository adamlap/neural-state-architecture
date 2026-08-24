"""Normative-to-security composition without granting the model authority."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .normative import NormativeAssessment, NormativeClass


class NormativeAction(str, Enum):
    CONTINUE = "continue"
    DENY = "deny"
    ESCALATE = "escalate"
    REQUIRE_APPROVAL = "require_approval"


@dataclass(frozen=True)
class NormativePolicy:
    deny_harmful: bool = True
    escalate_uncertain: bool = True
    require_approval_for_sensitive: bool = True

    def evaluate(self, assessment: NormativeAssessment) -> NormativeAction:
        if self.escalate_uncertain and assessment.uncertain:
            return NormativeAction.ESCALATE
        if self.deny_harmful and assessment.state.dominant is NormativeClass.HARMFUL:
            return NormativeAction.DENY
        if (
            self.require_approval_for_sensitive
            and assessment.state.dominant is NormativeClass.SENSITIVE
        ):
            return NormativeAction.REQUIRE_APPROVAL
        return NormativeAction.CONTINUE
