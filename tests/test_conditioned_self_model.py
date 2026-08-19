"""
tests/test_conditioned_self_model.py
====================================
Unit tests for Conditioned Predictive Self-Model & Counterfactual Simulation (Phases 18 & 19).
"""

import unittest
import torch
from nsa.self_model import (
    ConditionedPredictiveSelfModel,
    CounterfactualInternalSimulator,
    SimulationResult,
)


class TestConditionedSelfModel(unittest.TestCase):
    """Test suite for conditioned transition prediction, uncertainty, and counterfactual simulation."""

    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 32
        self.state_dim = 8
        self.model = ConditionedPredictiveSelfModel(
            d_model=self.d_model,
            state_dim=self.state_dim,
            action_dim=self.state_dim,
        )
        self.simulator = CounterfactualInternalSimulator(
            self_model=self.model,
            uncertainty_penalty=0.5,
        )

    def test_conditioned_prediction_and_hard_security_immutability(self):
        """Verify conditioned self-model outputs predicted delta, uncertainty, quality, and preserves hard security."""
        meaning = torch.randn(2, 4, self.d_model)
        state = torch.randn(2, 4, self.state_dim)
        state[..., 0] = 3.0  # Set discrete security coordinate
        action = torch.randn(2, 4, self.state_dim)

        out = self.model(meaning, state, action)

        self.assertIn("predicted_delta", out)
        self.assertIn("predicted_state", out)
        self.assertIn("uncertainty", out)
        self.assertIn("predicted_quality", out)

        # Invariant: Security coordinate 0 delta must be EXACTLY 0.0
        self.assertEqual(float(out["predicted_delta"][..., 0].abs().max().item()), 0.0)
        self.assertTrue(torch.equal(out["predicted_state"][..., 0], state[..., 0]))

        # Bounds checks
        self.assertTrue(torch.all(out["uncertainty"] >= 0.0) and torch.all(out["uncertainty"] <= 1.0))
        self.assertTrue(torch.all(out["predicted_quality"] >= -1.0) and torch.all(out["predicted_quality"] <= 1.0))

    def test_counterfactual_simulation_prunes_illegal_candidates(self):
        """Verify simulator strictly filters illegal actions and chooses highest scoring legal action."""
        meaning = torch.randn(1, 1, self.d_model)
        current_state = torch.randn(1, 1, self.state_dim)

        cand_legal_1 = ("action_1", torch.randn(1, 1, self.state_dim), True)
        cand_illegal = ("action_leak", torch.randn(1, 1, self.state_dim), False)
        cand_legal_2 = ("action_2", torch.randn(1, 1, self.state_dim), True)

        best_res, all_res = self.simulator.evaluate_candidates(
            meaning,
            current_state,
            [cand_legal_1, cand_illegal, cand_legal_2],
        )

        self.assertIsNotNone(best_res)
        self.assertTrue(best_res.is_legal)
        self.assertNotEqual(best_res.action_id, "action_leak")

        # Verify illegal action received -inf
        illegal_res = [r for r in all_res if r.action_id == "action_leak"][0]
        self.assertEqual(illegal_res.score, float("-inf"))
        self.assertFalse(illegal_res.is_legal)


if __name__ == "__main__":
    unittest.main()
