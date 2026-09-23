"""System One fast typed decision engine inspired by Jev (TypeSafe AI).

Provides single-pass, sub-inference typed decision making, RLCD-style
calibrated probability estimation for SoftState (uncertainty, risk),
and sub-10ms counterfactual branch pruning without autoregressive token loops.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping, Optional, Sequence

from nsa.cognition.interfaces import ActionCandidate
from nsa.core.state import CanonicalState, SoftState


@dataclass(frozen=True)
class TypedDecisionSchema:
    """Schema defining valid discrete choices, constraints, and confidence gates."""
    name: str
    choices: tuple[str, ...]
    descriptions: Mapping[str, str] = field(default_factory=dict)
    min_confidence: float = 0.50
    risk_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.choices:
            raise ValueError("TypedDecisionSchema must define at least one choice.")


@dataclass(frozen=True)
class CalibratedDecision:
    """Outcome of a System 1 decision with RLCD-calibrated confidence and epistemic entropy."""
    schema_name: str
    selected_choice: str
    probabilities: Mapping[str, float]
    calibrated_confidence: float
    epistemic_uncertainty: float
    risk_score: float
    latency_ms: float
    passed_gate: bool = True
    reason: str = ""


class SystemOneDecisionEngine:
    """Fast, single-pass typed decision engine.
    
    Operates without conversational LLM overhead, providing:
    1. Single-pass typed state evaluation.
    2. RLCD-calibrated probability distributions.
    3. Epistemic uncertainty (Shannon entropy) directly populating SoftState.
    4. Rapid parallel candidate scoring (<10ms) for counterfactual branch pruning.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        calibration_slope: float = 1.15,
        calibration_intercept: float = -0.05,
    ) -> None:
        self.temperature = max(1e-4, temperature)
        # Platt/logistic scaling parameters for RLCD probability calibration
        self.calibration_slope = calibration_slope
        self.calibration_intercept = calibration_intercept

    def calibrate_probability(self, raw_p: float) -> float:
        """Apply empirical calibration scaling to map raw confidence to true empirical accuracy."""
        raw_p = max(1e-6, min(1.0 - 1e-6, raw_p))
        # Log-odds transform followed by linear calibration
        logit = math.log(raw_p / (1.0 - raw_p))
        calibrated_logit = self.calibration_slope * logit + self.calibration_intercept
        calibrated_p = 1.0 / (1.0 + math.exp(-calibrated_logit))
        return max(0.0, min(1.0, calibrated_p))

    def compute_epistemic_uncertainty(self, probs: Sequence[float]) -> float:
        """Compute normalized Shannon entropy H in [0.0, 1.0] representing epistemic uncertainty."""
        k = len(probs)
        if k <= 1:
            return 0.0
        entropy = -sum(p * math.log(p) for p in probs if p > 1e-9)
        max_entropy = math.log(k)
        return max(0.0, min(1.0, entropy / max_entropy))

    def evaluate_schema(
        self,
        schema: TypedDecisionSchema,
        features: Mapping[str, float],
        priors: Optional[Mapping[str, float]] = None,
    ) -> CalibratedDecision:
        """Evaluate a typed schema over input features in a single forward pass."""
        start = monotonic()
        choices = schema.choices
        priors = priors or {}

        # Compute raw logits from features and priors
        logits: dict[str, float] = {}
        for choice in choices:
            score = features.get(choice, 0.0)
            prior = priors.get(choice, 1.0 / len(choices))
            prior_logit = math.log(max(1e-6, prior))
            logits[choice] = (score / self.temperature) + prior_logit

        # Softmax over choices
        max_l = max(logits.values())
        exp_logits = {c: math.exp(l - max_l) for c, l in logits.items()}
        sum_exp = sum(exp_logits.values())
        raw_probs = {c: val / sum_exp for c, val in exp_logits.items()}

        # Top choice
        best_choice = max(raw_probs, key=raw_probs.get)
        raw_confidence = raw_probs[best_choice]
        calibrated_conf = self.calibrate_probability(raw_confidence)

        # Epistemic uncertainty from normalized entropy
        epistemic_u = self.compute_epistemic_uncertainty(list(raw_probs.values()))

        # Associated risk score
        risk = schema.risk_weights.get(best_choice, 0.0)

        latency = (monotonic() - start) * 1000.0
        passed = calibrated_conf >= schema.min_confidence

        return CalibratedDecision(
            schema_name=schema.name,
            selected_choice=best_choice,
            probabilities=raw_probs,
            calibrated_confidence=round(calibrated_conf, 4),
            epistemic_uncertainty=round(epistemic_u, 4),
            risk_score=round(risk, 4),
            latency_ms=round(latency, 3),
            passed_gate=passed,
            reason="" if passed else f"calibrated confidence {calibrated_conf:.2f} < gate {schema.min_confidence:.2f}",
        )

    def evaluate_candidate(
        self,
        current_state: CanonicalState,
        candidate: ActionCandidate,
    ) -> CalibratedDecision:
        """Evaluate an ActionCandidate using System 1 fast path."""
        start = monotonic()
        # Compute intuitive alignment features between current state goals/semantics and candidate
        goal_match = 0.8 if any(candidate.action_id.lower() in g.lower() for g in current_state.goals.goals) else 0.4
        utility_signal = candidate.expected_utility
        risk_penalty = candidate.risk * 1.2
        confidence_signal = getattr(candidate, "confidence", max(0.0, 1.0 - candidate.risk))

        score = (utility_signal * 1.5) + (goal_match * 1.0) + (confidence_signal * 0.8) - risk_penalty

        # Binary schema: accept vs reject
        schema = TypedDecisionSchema(
            name=f"eval_{candidate.action_id}",
            choices=("execute", "reject"),
            risk_weights={"execute": candidate.risk, "reject": 0.0},
            min_confidence=0.50,
        )

        features = {"execute": score, "reject": 1.0}
        priors = {"execute": 0.6, "reject": 0.4}
        decision = self.evaluate_schema(schema, features, priors)
        return decision

    def fast_prune_candidates(
        self,
        current_state: CanonicalState,
        candidates: Sequence[ActionCandidate],
        max_acceptable_risk: float = 0.70,
    ) -> tuple[ActionCandidate, ...]:
        """Sub-10ms pre-filtering: prune high-risk, low-confidence, or un-executable branches."""
        pruned: list[ActionCandidate] = []
        for candidate in candidates:
            # Capability check
            missing = [cap for cap in candidate.required_capabilities if not current_state.hard.has_permission(cap)]
            if missing:
                continue

            # Hard risk boundary
            if candidate.risk > max_acceptable_risk:
                continue

            # System 1 evaluation
            decision = self.evaluate_candidate(current_state, candidate)
            if decision.selected_choice == "execute" and decision.passed_gate:
                pruned.append(candidate)

        return tuple(pruned)

    def update_soft_state(
        self,
        current_state: CanonicalState,
        decision: CalibratedDecision,
    ) -> SoftState:
        """Update CanonicalState.soft with mathematically calibrated epistemic uncertainty and risk."""
        return SoftState(
            uncertainty=decision.epistemic_uncertainty,
            risk=decision.risk_score,
            confidence=decision.calibrated_confidence,
            resource_pressure=current_state.soft.resource_pressure,
        )
