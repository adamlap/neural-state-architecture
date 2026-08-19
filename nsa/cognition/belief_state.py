"""
nsa/cognition/belief_state.py
=============================
NSA 5.1 Belief-State Cognitive Dynamics & Active Information Gain Engine.

Maintains a formal discrete belief state B_t over competing world hypotheses:
    B_t = { (w_1, p_1), (w_2, p_2), ..., (w_n, p_n) }

Selects actions to maximize Expected Utility + Information Gain subject to ISK safety:
    a* = argmax_{a in A_legal} [ E_B[U(a)] + beta * I(W; O_{t+1} | a) - lambda * Risk(a) ]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class WorldHypothesis:
    hypothesis_id: str
    description: str
    probability: float
    compatible_observations: List[str]
    required_recovery_action: str


@dataclass
class BeliefState:
    hypotheses: List[WorldHypothesis]
    entropy: float = 0.0

    def __post_init__(self) -> None:
        self.normalize()

    def normalize(self) -> None:
        total_p = sum(h.probability for h in self.hypotheses)
        if total_p > 0:
            for h in self.hypotheses:
                h.probability /= total_p
        # Compute Shannon entropy H(W)
        self.entropy = -sum(
            h.probability * math.log2(h.probability) for h in self.hypotheses if h.probability > 0
        )

    def update_with_observation(self, observation: str) -> None:
        """Bayesian update of belief state given new observation evidence."""
        for h in self.hypotheses:
            if observation in h.compatible_observations:
                h.probability *= 0.90
            else:
                h.probability *= 0.10
        self.normalize()


class InformationGainSelector:
    """Selects actions maximizing information gain and task utility within legal constraints."""

    @classmethod
    def calculate_information_gain(
        cls,
        current_belief: BeliefState,
        action_name: str,
        discriminating_actions: Dict[str, str],
    ) -> float:
        """Estimate mutual information I(W; O | a)."""
        if action_name in discriminating_actions:
            # Action safely discriminates hypotheses, reducing entropy by up to H(W)
            return float(current_belief.entropy * 0.85)
        return 0.05

    @classmethod
    def score_action(
        cls,
        action_name: str,
        expected_utility: float,
        risk_level: float,
        info_gain: float,
        beta_info: float = 1.2,
        lambda_risk: float = 1.0,
    ) -> float:
        return expected_utility + (beta_info * info_gain) - (lambda_risk * risk_level)
