"""
nsa/formal/non_transference.py
==============================
NSA 4.0 General Authority Non-Transference Algebra.

Formally specifies and verifies the non-transference property:
    X_i -/-> X_j

Where intelligence dimensions cannot implicitly confer authority upon one another:
    Confidence (eps)      -/-> Authority (sigma_h)
    Capability / Scale    -/-> Authority (sigma_h)
    Utility (U)           -/-> Authority (sigma_h)
    Goal / Intent (g)     -/-> Authority (sigma_h)
    Prediction (Omega_hat)-/-> Authority (sigma_h)
    Authority (sigma_h)   -/-> Truth / Grounded Justification (eps)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class CognitiveDimension(enum.Enum):
    CONFIDENCE = "confidence"          # Epistemic justification / confidence eps
    CAPABILITY = "capability"          # Model size, reasoning FLOPs, parameter count
    UTILITY = "utility"                # Expected task reward / utility U
    GOAL = "goal"                      # Teleological intent / priority g
    PREDICTION = "prediction"          # Counterfactual prediction Omega_hat
    AUTHORITY = "authority"            # Operational clearance / permission sigma_h
    TRUTH = "truth"                    # Grounded factual validity


@dataclass(frozen=True)
class DimensionTransferAttempt:
    source_dimension: CognitiveDimension
    target_dimension: CognitiveDimension
    claimed_value: float
    is_externally_authorized: bool = False


class AuthorityNonTransferenceEngine:
    """Evaluates cross-dimensional state flows and blocks unauthorized transference."""

    # Forbidden implicit transfer pairs without explicit external cryptographic proof
    FORBIDDEN_TRANSFERS: Set[Tuple[CognitiveDimension, CognitiveDimension]] = {
        (CognitiveDimension.CONFIDENCE, CognitiveDimension.AUTHORITY),
        (CognitiveDimension.CAPABILITY, CognitiveDimension.AUTHORITY),
        (CognitiveDimension.UTILITY, CognitiveDimension.AUTHORITY),
        (CognitiveDimension.GOAL, CognitiveDimension.AUTHORITY),
        (CognitiveDimension.PREDICTION, CognitiveDimension.AUTHORITY),
        (CognitiveDimension.AUTHORITY, CognitiveDimension.TRUTH),
    }

    @classmethod
    def evaluate_transfer(cls, transfer: DimensionTransferAttempt) -> Tuple[bool, str]:
        """Verify whether cross-dimensional flow is permissible."""
        pair = (transfer.source_dimension, transfer.target_dimension)

        if pair in cls.FORBIDDEN_TRANSFERS:
            if not transfer.is_externally_authorized:
                return (
                    False,
                    f"Non-Transference Violation: Dimension '{transfer.source_dimension.value}' cannot confer '{transfer.target_dimension.value}' without external cryptographic authorization.",
                )

        return True, "Dimension flow conforms to non-transference algebra."
