"""Unified declarative safety runtime for real NSA inference backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from nsa.core.capabilities import TrustTier
from nsa.decision import Decision, SecurityDecision
from nsa.enforcement import EvaluationContext, PolicyEngine
from nsa.policy import NSAPolicy, PolicyCompiler
from nsa.runtime.inference.base import InferenceBackend, LLMGenerationOutput
from nsa.runtime.inference.governed import NSAGovernedInference


@dataclass(frozen=True)
class PolicyRuntimeResult:
    """Structured result from policy-governed generation."""
    text: str
    request_decision: SecurityDecision
    output_decision: SecurityDecision
    generated: bool
    blocked_stage: Optional[str] = None

    @property
    def allowed(self) -> bool:
        return self.request_decision.allowed and self.output_decision.allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "generated": self.generated,
            "blocked_stage": self.blocked_stage,
            "allowed": self.allowed,
            "request_decision": self.request_decision.summary(),
            "output_decision": self.output_decision.summary(),
        }


class NSAPolicyRuntime:
    """Executable policy boundary around a real NSA inference backend."""

    def __init__(self, backend: InferenceBackend, policy: NSAPolicy, *, user_clearance: TrustTier = TrustTier.T1_INFO_GATHER, model_name: str = "unknown", classifier=None, refusal_text: str = "I can't help with that request under the active safety policy.") -> None:
        self.policy = policy
        self.engine: PolicyEngine = PolicyCompiler.compile(policy, classifier=classifier)
        self.governed = NSAGovernedInference(backend, user_clearance, model_name)
        self.refusal_text = refusal_text

    def evaluate(self, text: str, *, action: str = "generate", capabilities=frozenset(), protected_data=frozenset(), risk: float = 0.0, uncertainty: float = 0.0) -> SecurityDecision:
        return self.engine.evaluate(text, context=EvaluationContext(action=action, capabilities=frozenset(capabilities), protected_data=frozenset(protected_data), risk=risk, uncertainty=uncertainty), state=self.governed.state)

    def generate(self, prompt: str, *, max_tokens: int = 512, temperature: float = 0.7, system_prompt: Optional[str] = None, context: Optional[EvaluationContext] = None) -> PolicyRuntimeResult:
        ctx = context or EvaluationContext(action="generate")
        request_decision = self.engine.evaluate(prompt, context=ctx, state=self.governed.state)
        if request_decision.decision is not Decision.ALLOW:
            return PolicyRuntimeResult(self.refusal_text, request_decision, request_decision, False, "request")

        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        output: LLMGenerationOutput = self.governed.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)
        output_ctx = EvaluationContext(action=ctx.action, capabilities=ctx.capabilities, protected_data=ctx.protected_data, risk=ctx.risk, uncertainty=max(ctx.uncertainty, 1.0 - output.confidence_estimate))
        output_decision = self.engine.evaluate(output.text, context=output_ctx, state=self.governed.state)
        if output_decision.decision is not Decision.ALLOW:
            return PolicyRuntimeResult(self.refusal_text, request_decision, output_decision, True, "output")
        return PolicyRuntimeResult(output.text, request_decision, output_decision, True)

    def status(self) -> Dict[str, Any]:
        return {"policy": self.policy.to_mapping(), "policy_engine": "deterministic_reference_classifier", "model": self.governed.model_name, "governed_inference": self.governed.status()}


__all__ = ["NSAPolicyRuntime", "PolicyRuntimeResult"]
