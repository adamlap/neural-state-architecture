"""Policy evaluation and enforcement boundary.

The engine is intentionally model-agnostic. A semantic classifier can be learned,
remote, or deterministic. Its output is converted into an explicit SecurityDecision;
execution code should never infer authority from generated text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Protocol, Sequence

from nsa.core.state import CanonicalState
from nsa.decision import Decision, SecurityDecision
from nsa.policy import NSAPolicy


class PolicyClassifier(Protocol):
    def classify(self, text: str) -> Sequence[str]:
        """Return semantic policy categories detected in text."""


class KeywordClassifier:
    """Small deterministic reference classifier for demos and tests.

    Production systems should replace this with a trained semantic classifier.
    Patterns are supplied by the application and are deliberately not baked into
    NSA itself.
    """

    def __init__(self, patterns: dict[str, Sequence[str]]) -> None:
        self.patterns = {category: tuple(p.lower() for p in values) for category, values in patterns.items()}

    def classify(self, text: str) -> Sequence[str]:
        lowered = text.lower()
        return tuple(category for category, patterns in self.patterns.items() if any(p in lowered for p in patterns))


@dataclass(frozen=True)
class EvaluationContext:
    action: str = "generate"
    capabilities: FrozenSet[str] = frozenset()
    protected_data: FrozenSet[str] = frozenset()
    risk: float = 0.0
    uncertainty: float = 0.0


class PolicyEngine:
    """Evaluate semantic and capability state against an NSAPolicy."""

    def __init__(self, policy: NSAPolicy, classifier: Optional[PolicyClassifier] = None) -> None:
        self.policy = policy
        self.classifier = classifier or KeywordClassifier(policy.classifier_patterns())

    def evaluate(
        self,
        text: str,
        *,
        context: Optional[EvaluationContext] = None,
        state: Optional[CanonicalState] = None,
    ) -> SecurityDecision:
        ctx = context or EvaluationContext()
        categories = tuple(self.classifier.classify(text))

        matched = []
        hard = set()
        for category in categories:
            rule = self.policy.rule_for(category)
            if rule is None:
                continue
            matched.append(category)
            hard.add("policy:" + category)
            if rule.mode == "deny":
                return SecurityDecision(
                    Decision.DENY, self.policy.name, rule.reason or "prohibited semantic category",
                    tuple(matched), frozenset(hard), uncertainty=ctx.uncertainty, risk=max(ctx.risk, 1.0),
                )
            if rule.mode == "escalate":
                return SecurityDecision(
                    Decision.ESCALATE, self.policy.name, rule.reason or "policy requires review",
                    tuple(matched), frozenset(hard), uncertainty=max(ctx.uncertainty, 0.5), risk=max(ctx.risk, 0.5),
                )

        required = set(ctx.capabilities) & set(self.policy.restricted_actions)
        if required:
            authorizations = state.hard.authorizations if state is not None else frozenset()
            missing = required - set(authorizations)
            if missing:
                return SecurityDecision(
                    Decision.DENY, self.policy.name, "required capability is not authorized",
                    tuple(matched), frozenset(hard) | {"capability:" + x for x in missing},
                    frozenset(missing), risk=max(ctx.risk, 0.9), uncertainty=ctx.uncertainty,
                )

        protected = set(ctx.protected_data) & set(self.policy.protected_data)
        if protected:
            return SecurityDecision(
                Decision.DENY, self.policy.name, "protected data cannot cross this boundary",
                tuple(matched), frozenset(hard) | {"data:" + x for x in protected},
                risk=max(ctx.risk, 0.95), uncertainty=ctx.uncertainty,
            )

        if ctx.action in self.policy.require_approval:
            return SecurityDecision(
                Decision.REQUIRE_APPROVAL, self.policy.name, "action requires explicit approval",
                tuple(matched), frozenset(hard), frozenset(ctx.capabilities),
                risk=ctx.risk, uncertainty=ctx.uncertainty,
            )

        if ctx.uncertainty >= 0.8:
            outcome = Decision(self.policy.default_uncertainty)
            return SecurityDecision(outcome, self.policy.name, "high semantic uncertainty", tuple(matched), frozenset(hard), risk=ctx.risk, uncertainty=ctx.uncertainty)

        return SecurityDecision(
            Decision.ALLOW, self.policy.name, "no policy constraint matched",
            tuple(matched), frozenset(hard), frozenset(required), risk=ctx.risk, uncertainty=ctx.uncertainty,
        )

    def enforce(self, text: str, *, context: Optional[EvaluationContext] = None, state: Optional[CanonicalState] = None) -> SecurityDecision:
        """Evaluate without silently converting DENY into a model-generated refusal."""
        return self.evaluate(text, context=context, state=state)
