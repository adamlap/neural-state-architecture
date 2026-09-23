"""Closed-loop counterfactual simulation and speculative safety gating.

Allows cognition to fork speculative branches of CanonicalState, simulate
consequences of candidate actions, verify them against safety invariants,
and prune hazardous or unauthorized actions before live execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Callable, Mapping, Optional, Sequence

from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState, SoftState


@dataclass(frozen=True)
class CounterfactualBranch:
    """The result of speculatively simulating a candidate action."""
    action_id: str
    simulated_state: CanonicalState
    simulated_risk: float
    simulated_utility: float
    safe: bool
    rejection_reason: str = ""


@dataclass(frozen=True)
class CounterfactualEvaluation:
    """Outcome of evaluating multiple speculative action branches."""
    branches: tuple[CounterfactualBranch, ...]
    recommended_action: Optional[CounterfactualBranch]
    rejection_count: int
    evaluated_count: int
    duration_ms: float
    timestamp: float = field(default_factory=monotonic)


class CounterfactualSimulator:
    """Anticipatory simulation engine gating live execution behind safety invariants."""

    def __init__(
        self,
        max_acceptable_risk: float = 0.70,
        risk_escalation_limit: float = 0.40,
        safety_kernel: Any = None,
        capability_evaluator: Any = None,
        system_one: Any = None,
    ) -> None:
        self.max_acceptable_risk = max_acceptable_risk
        self.risk_escalation_limit = risk_escalation_limit
        self.safety_kernel = safety_kernel
        self.capability_evaluator = capability_evaluator
        self.system_one = system_one

    def simulate_branch(
        self,
        current_state: CanonicalState,
        candidate: ActionCandidate,
        speculative_dynamics: Optional[Callable[[CanonicalState, ActionCandidate], CanonicalState]] = None,
    ) -> CounterfactualBranch:
        """Fork state and speculatively simulate one candidate action."""
        # Step 1: Simulate next state under the candidate action
        if speculative_dynamics is not None:
            projected_state = speculative_dynamics(current_state, candidate)
        else:
            # Default conservative dynamics: project risk and utility
            new_risk = min(1.0, current_state.soft.risk + candidate.risk)
            new_uncertainty = max(0.0, current_state.soft.uncertainty - (candidate.expected_utility * 0.1))
            new_soft = SoftState(
                uncertainty=new_uncertainty,
                risk=new_risk,
                confidence=max(0.0, 1.0 - new_risk),
                resource_pressure=current_state.soft.resource_pressure,
            )
            projected_state = current_state.observe(
                uncertainty=new_soft.uncertainty,
                risk=new_soft.risk,
                confidence=new_soft.confidence,
                resource_pressure=new_soft.resource_pressure,
            )

        # Step 2: Risk and divergence checks
        risk_delta = projected_state.soft.risk - current_state.soft.risk
        if projected_state.soft.risk > self.max_acceptable_risk:
            return CounterfactualBranch(
                action_id=candidate.action_id,
                simulated_state=projected_state,
                simulated_risk=projected_state.soft.risk,
                simulated_utility=candidate.expected_utility,
                safe=False,
                rejection_reason=f"simulated risk {projected_state.soft.risk:.2f} exceeds threshold {self.max_acceptable_risk:.2f}",
            )

        if risk_delta > self.risk_escalation_limit:
            return CounterfactualBranch(
                action_id=candidate.action_id,
                simulated_state=projected_state,
                simulated_risk=projected_state.soft.risk,
                simulated_utility=candidate.expected_utility,
                safe=False,
                rejection_reason=f"risk escalation delta {risk_delta:.2f} exceeds limit {self.risk_escalation_limit:.2f}",
            )

        # Step 3: Hard Safety Kernel Verification (if provided)
        if self.safety_kernel is not None:
            verdict = self.safety_kernel.evaluate(projected_state)
            if not getattr(verdict, "passed", True):
                return CounterfactualBranch(
                    action_id=candidate.action_id,
                    simulated_state=projected_state,
                    simulated_risk=projected_state.soft.risk,
                    simulated_utility=candidate.expected_utility,
                    safe=False,
                    rejection_reason=f"safety kernel invariant failed: {getattr(verdict, 'reason', 'unknown')}",
                )

        # Step 4: Capability Authorization Verification
        missing = [cap for cap in candidate.required_capabilities if not current_state.hard.has_permission(cap)]
        if missing:
            return CounterfactualBranch(
                action_id=candidate.action_id,
                simulated_state=projected_state,
                simulated_risk=projected_state.soft.risk,
                simulated_utility=candidate.expected_utility,
                safe=False,
                rejection_reason=f"missing required capabilities: {missing}",
            )

        return CounterfactualBranch(
            action_id=candidate.action_id,
            simulated_state=projected_state,
            simulated_risk=projected_state.soft.risk,
            simulated_utility=candidate.expected_utility,
            safe=True,
            rejection_reason="",
        )

    def evaluate_candidates(
        self,
        current_state: CanonicalState,
        candidates: Sequence[ActionCandidate],
        speculative_dynamics: Optional[Callable[[CanonicalState, ActionCandidate], CanonicalState]] = None,
    ) -> CounterfactualEvaluation:
        """Simulate and evaluate multiple candidate branches to recommend the optimal safe action."""
        start_time = monotonic()
        branches: list[CounterfactualBranch] = []
        rejection_count = 0

        # Optional fast System 1 pre-filtering
        eval_candidates = candidates
        if self.system_one is not None and hasattr(self.system_one, "fast_prune_candidates"):
            eval_candidates = self.system_one.fast_prune_candidates(
                current_state, candidates, max_acceptable_risk=self.max_acceptable_risk
            )
            rejection_count += (len(candidates) - len(eval_candidates))

        for candidate in eval_candidates:
            branch = self.simulate_branch(current_state, candidate, speculative_dynamics=speculative_dynamics)
            branches.append(branch)
            if not branch.safe:
                rejection_count += 1

        # Select highest net utility among safe branches: Net = Utility - Risk
        safe_branches = [b for b in branches if b.safe]
        recommended = max(safe_branches, key=lambda b: b.simulated_utility - b.simulated_risk, default=None)

        duration_ms = (monotonic() - start_time) * 1000
        return CounterfactualEvaluation(
            branches=tuple(branches),
            recommended_action=recommended,
            rejection_count=rejection_count,
            evaluated_count=len(candidates),
            duration_ms=round(duration_ms, 2),
        )