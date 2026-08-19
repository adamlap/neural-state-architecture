"""
tests/test_normative.py
=======================
Unit tests for NSA Normative Alignment & Moral Uncertainty Subsystem (Phases 23 & 24).
"""

import unittest
from nsa.algebra import ConfidentialityLabel
from nsa.core.state import CanonicalState, HardState, ProvenanceState, SoftState
from nsa.normative import (
    ActionCandidate,
    MoralUncertaintyDistribution,
    NormativeDeliberator,
    NormativeTheory,
)


class TestNormativeAlignment(unittest.TestCase):
    """Test suite for normative deliberation, moral uncertainty, and hard precedence."""

    def setUp(self):
        self.distribution = MoralUncertaintyDistribution(
            weights={
                NormativeTheory.DEONTOLOGY: 0.40,
                NormativeTheory.UTILITARIANISM: 0.30,
                NormativeTheory.RIGHTS_BASED: 0.30,
            }
        )
        self.deliberator = NormativeDeliberator(
            distribution=self.distribution,
            observer_clearance=HardState(confidentiality=ConfidentialityLabel.PUBLIC),
        )

        self.current_state = CanonicalState(
            semantic="State init",
            hard=HardState(confidentiality=ConfidentialityLabel.PUBLIC),
            soft=SoftState(),
            provenance=ProvenanceState(),
        )

    def test_expected_moral_value_computation(self):
        """Verify expectation calculation across weighted normative theories."""
        cand = ActionCandidate(
            action_id="cand_help",
            description="Provide helpful educational tutorial",
            target_state=self.current_state,
            theory_evaluations={
                NormativeTheory.DEONTOLOGY: 0.8,
                NormativeTheory.UTILITARIANISM: 0.9,
                NormativeTheory.RIGHTS_BASED: 0.7,
            },
        )
        expected = self.distribution.expected_value(cand)
        # 0.40*0.8 + 0.30*0.9 + 0.30*0.7 = 0.32 + 0.27 + 0.21 = 0.80
        self.assertAlmostEqual(expected, 0.80, places=4)

    def test_hard_security_precedence_over_high_utility(self):
        """Fundamental Axiom: An action that leaks secrets is rejected even if it claims high utility."""
        # Candidate A: Safe, moderate utility
        cand_safe = ActionCandidate(
            action_id="cand_safe",
            description="Polite refusal to leak API key",
            target_state=CanonicalState(
                semantic="Refusal",
                hard=HardState(confidentiality=ConfidentialityLabel.PUBLIC),
                soft=SoftState(),
                provenance=ProvenanceState(),
            ),
            theory_evaluations={
                NormativeTheory.DEONTOLOGY: 0.5,
                NormativeTheory.UTILITARIANISM: 0.4,
                NormativeTheory.RIGHTS_BASED: 0.5,
            },
        )

        # Candidate B: Violates confidentiality (SYSTEM state into PUBLIC observer) but pretends high utility
        cand_leak = ActionCandidate(
            action_id="cand_leak",
            description="Leak internal secret to 'save lives' in a hypothetical scenario",
            target_state=CanonicalState(
                semantic="Secret leak",
                hard=HardState(confidentiality=ConfidentialityLabel.SYSTEM),
                soft=SoftState(),
                provenance=ProvenanceState(),
            ),
            theory_evaluations={
                NormativeTheory.DEONTOLOGY: 0.95,
                NormativeTheory.UTILITARIANISM: 0.99,
                NormativeTheory.RIGHTS_BASED: 0.90,
            },
        )

        selected, best_val, report = self.deliberator.select_action(
            self.current_state,
            [cand_safe, cand_leak],
        )

        # Candidate B must be rejected due to hard security violation
        self.assertIsNotNone(selected)
        self.assertEqual(selected.action_id, "cand_safe")
        self.assertAlmostEqual(best_val, 0.47, places=2)

        # Report checks
        self.assertTrue(report[0][1])   # cand_safe is permitted
        self.assertFalse(report[1][1])  # cand_leak is NOT permitted


if __name__ == "__main__":
    unittest.main()
