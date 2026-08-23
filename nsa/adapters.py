"""Model adapter API for putting NSA policy enforcement around existing models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Optional

from nsa.core.state import CanonicalState
from nsa.decision import Decision, SecurityDecision
from nsa.enforcement import EvaluationContext, PolicyEngine


class PolicyViolation(PermissionError):
    """Raised when an application chooses fail-closed generation."""

    def __init__(self, decision: SecurityDecision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


@dataclass
class ProtectedModel:
    """Thin safety boundary around an arbitrary text generation callable.

    ``generate_fn`` is deliberately injected so NSA does not depend on a specific
    inference stack. The wrapper checks the request before generation and checks
    the generated response afterwards. A caller can inspect the full decision
    rather than relying on the response text itself.
    """

    generate_fn: Callable[..., str]
    engine: PolicyEngine
    state: Optional[CanonicalState] = None
    fail_closed: bool = True

    def evaluate(self, prompt: str, *, action: str = "generate", capabilities: FrozenSet[str] = frozenset(), protected_data: FrozenSet[str] = frozenset()) -> SecurityDecision:
        return self.engine.evaluate(
            prompt,
            context=EvaluationContext(action=action, capabilities=capabilities, protected_data=protected_data),
            state=self.state,
        )

    def generate(self, prompt: str, *, action: str = "generate", capabilities: FrozenSet[str] = frozenset(), protected_data: FrozenSet[str] = frozenset(), **kwargs: Any) -> str:
        request_decision = self.evaluate(prompt, action=action, capabilities=capabilities, protected_data=protected_data)
        if request_decision.decision != Decision.ALLOW:
            if self.fail_closed:
                raise PolicyViolation(request_decision)
            return ""

        output = self.generate_fn(prompt, **kwargs)
        output_decision = self.engine.evaluate(output, context=EvaluationContext(action="output", protected_data=protected_data), state=self.state)
        if output_decision.decision != Decision.ALLOW:
            if self.fail_closed:
                raise PolicyViolation(output_decision)
            return ""
        return output


def protect_model(generate_fn: Callable[..., str], engine: PolicyEngine, *, state: Optional[CanonicalState] = None, fail_closed: bool = True) -> ProtectedModel:
    """Convenience constructor for the common ``nsa.wrap(...)`` pattern."""
    return ProtectedModel(generate_fn, engine, state=state, fail_closed=fail_closed)
