"""End-to-End Governance and Capability Execution Harness (Phase C).

Unifies:
1. Declarative NSAPolicy
2. Deterministic or Neural PolicyEngine
3. Real LLM / Ollama Backend
4. Normative State Transition Engine (nu_{t+1})
5. CapabilityGate Reference Monitor

Guarantees:
- Model output NEVER directly exercises a capability.
- SecurityDecision must ALLOW before CapabilityGate executes any tool.
- DENY guarantees zero tool invocations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from nsa.capabilities.gate import CapabilityAccessDenied, CapabilityGate
from nsa.capabilities.model import Capability, CapabilityAuthority
from nsa.core.state import CanonicalState, HardState
from nsa.decision import Decision, SecurityDecision
from nsa.enforcement import EvaluationContext, PolicyEngine
from nsa.normative.engine import NormativeTransitionEngine
from nsa.normative.state import NormativeState
from nsa.policy import NSAPolicy
from nsa.runtime.inference.base import InferenceBackend

logger = logging.getLogger("GovernedHarness")


@dataclass(frozen=True)
class GovernedExecutionResult:
    """Result of an end-to-end governed model turn and capability execution."""

    prompt: str
    raw_model_response: str
    decision: SecurityDecision
    normative_state: NormativeState
    capability_executed: bool
    tool_name: Optional[str]
    tool_result: Optional[Any]
    denial_reason: Optional[str] = None


class GovernedExecutionHarness:
    """Full-loop reference monitor integrating LLM generation with strict capability gates."""

    def __init__(
        self,
        policy: NSAPolicy,
        backend: InferenceBackend,
        authority: Optional[CapabilityAuthority] = None,
        initial_normative: Optional[NormativeState] = None,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self.policy_engine = PolicyEngine(policy)
        self.authority = authority
        self.gate = CapabilityGate(authority=authority)
        self.transition_engine = NormativeTransitionEngine()
        self.normative_state = initial_normative or NormativeState(values={"harm": 0.0, "sensitivity": 0.1}, confidence=1.0)
        self._tools: Dict[str, Callable[..., Any]] = {}

    def register_tool(self, tool_name: str, fn: Callable[..., Any]) -> None:
        """Register an authorized tool implementation with the harness."""
        self._tools[tool_name] = fn

    def run_turn(
        self,
        prompt: str,
        *,
        requested_tool: Optional[str] = None,
        tool_args: Optional[Sequence[Any]] = None,
        tool_kwargs: Optional[Mapping[str, Any]] = None,
        state: Optional[CanonicalState] = None,
        caller: str = "agent_system",
    ) -> GovernedExecutionResult:
        """Process one conversational/agent turn with full governance checks."""
        # 1. Generate model response from backend
        model_output = self.backend.generate(prompt, max_tokens=256)
        output_text = model_output.text.strip()

        # 2. Evaluate semantic policy against model output & prompt
        eval_text = f"{prompt}\n{output_text}"
        ctx = EvaluationContext(
            action=requested_tool or "generate",
            capabilities=frozenset([requested_tool]) if requested_tool else frozenset(),
        )
        decision = self.policy_engine.evaluate(eval_text, context=ctx, state=state)

        # 3. Advance normative state trajectory
        observed_harm = 0.9 if decision.decision == Decision.DENY else 0.05
        observed_sens = 0.8 if "sensitive" in decision.matched_categories else 0.1
        self.normative_state = self.transition_engine.step(
            current_nu=self.normative_state,
            input_text=eval_text,
            memory_context={},
            sigma_h=state.hard if state else HardState(),
            observed_signals={"harm": observed_harm, "sensitivity": observed_sens},
            observed_confidence=0.9,
        )

        # 4. If a tool was requested, pass through CapabilityGate
        if requested_tool is not None:
            if requested_tool not in self._tools:
                raise ValueError(f"Unknown tool requested: {requested_tool}")
            
            tool_fn = self._tools[requested_tool]
            args = tool_args or ()
            kwargs = tool_kwargs or {}

            try:
                result = self.gate.require(
                    decision,
                    requested_tool,
                    tool_fn,
                    *args,
                    caller=caller,
                    **kwargs,
                )
                return GovernedExecutionResult(
                    prompt=prompt,
                    raw_model_response=output_text,
                    decision=decision,
                    normative_state=self.normative_state,
                    capability_executed=True,
                    tool_name=requested_tool,
                    tool_result=result,
                )
            except CapabilityAccessDenied as exc:
                return GovernedExecutionResult(
                    prompt=prompt,
                    raw_model_response=output_text,
                    decision=decision,
                    normative_state=self.normative_state,
                    capability_executed=False,
                    tool_name=requested_tool,
                    tool_result=None,
                    denial_reason=str(exc),
                )

        # Text-only response
        return GovernedExecutionResult(
            prompt=prompt,
            raw_model_response=output_text,
            decision=decision,
            normative_state=self.normative_state,
            capability_executed=False,
            tool_name=None,
            tool_result=None,
        )


__all__ = [
    "GovernedExecutionResult",
    "GovernedExecutionHarness",
]
