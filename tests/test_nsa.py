"""
tests/test_nsa.py
=================
Unit test suite for Neural State Architecture (NSA) components.
"""

import unittest

from nsa.algebra import (
    StateLabel,
    StateLattice,
    ConservationLaw,
    DEFAULT_LATTICE,
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestStateAlgebra(unittest.TestCase):
    """Test suite for state algebra, lattice operations, and conservation laws."""

    def test_state_label_ordering(self):
        """Verify state label numerical hierarchy."""
        self.assertTrue(StateLabel.SYSTEM > StateLabel.PRIVATE)
        self.assertTrue(StateLabel.PRIVATE > StateLabel.CONFIDENTIAL)
        self.assertTrue(StateLabel.CONFIDENTIAL > StateLabel.TRUSTED)
        self.assertTrue(StateLabel.TRUSTED > StateLabel.PUBLIC)
        self.assertTrue(StateLabel.PUBLIC > StateLabel.UNTRUSTED)

    def test_lattice_meet_and_join(self):
        """Verify lattice meet (infimum) and join (supremum) operations."""
        lattice = DEFAULT_LATTICE
        # Meet (greatest lower bound / greatest permission overlap)
        self.assertEqual(lattice.meet(StateLabel.PRIVATE, StateLabel.PUBLIC), StateLabel.PUBLIC)
        # Join (least upper bound / least restrictive upper bound)
        self.assertEqual(lattice.join(StateLabel.PUBLIC, StateLabel.PRIVATE), StateLabel.PRIVATE)

    def test_conservation_law_monotone(self):
        """Verify monotone conservation laws in state lattice."""
        lattice = DEFAULT_LATTICE
        # Monotone transition (equal or higher sensitivity level / upward)
        self.assertTrue(lattice.is_allowed(StateLabel.PUBLIC, StateLabel.PRIVATE))
        self.assertTrue(lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PRIVATE))
        # Forbidden transition (declassification downward without explicit gate)
        self.assertFalse(lattice.is_allowed(StateLabel.PRIVATE, StateLabel.PUBLIC))

    def test_custom_law_override(self):
        """Verify explicit conservation law creation and violation check."""
        law = ConservationLaw(
            from_label=StateLabel.PRIVATE,
            to_label=StateLabel.PUBLIC,
            allowed=False,
            penalty_weight=2.5
        )
        self.assertTrue(law.is_violated(StateLabel.PRIVATE, StateLabel.PUBLIC))
        self.assertFalse(law.is_violated(StateLabel.PUBLIC, StateLabel.PRIVATE))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for neural state primitives tests")
class TestStatePrimitives(unittest.TestCase):
    """Test suite for state vectors and transition operators."""

    def test_state_vector_creation(self):
        """Verify state vector initialization."""
        from nsa.state import StateVector
        sv = StateVector(state_dim=8, mode="discrete", init_label=StateLabel.PRIVATE)
        self.assertEqual(sv.state_dim, 8)
        self.assertEqual(sv.most_likely_label(), StateLabel.PRIVATE)

    def test_transition_operator(self):
        """Verify state transition operator matrix dimensions."""
        from nsa.state import StateTransitionOperator
        op = StateTransitionOperator(state_dim=8)
        self.assertEqual(op.state_dim, 8)
        self.assertEqual(op.V.shape, (8, 8))


@unittest.skipUnless(HAS_TORCH, "PyTorch required for utility function tests")
class TestUtils(unittest.TestCase):
    """Test suite for utility functions."""

    def test_count_parameters(self):
        """Test parameter counting utility."""
        from nsa.utils import count_parameters

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)  # 100 weights + 10 biases = 110

        model = DummyModel()
        counts = count_parameters(model)
        self.assertEqual(counts["total"], 110)
        self.assertEqual(counts["trainable"], 110)


@unittest.skipUnless(HAS_TORCH, "PyTorch required for causal LM tests")
class TestNSACausalLM(unittest.TestCase):
    """Test suite for NSACausalLM forward pass and logits shape."""

    def test_causal_lm_forward(self):
        """Verify NSACausalLM forward output tensor shapes."""
        from nsa.layers import NSACausalLM
        model = NSACausalLM(vocab_size=100, d_model=32, state_dim=4, num_layers=2, num_heads=4)
        tokens = torch.randint(0, 100, (2, 16))
        states = torch.randn(2, 16, 4)

        logits, x_out, state_out = model(tokens, state_init=states)
        self.assertEqual(logits.shape, (2, 16, 100))
        self.assertEqual(x_out.shape, (2, 16, 32))
        self.assertEqual(state_out.shape, (2, 16, 4))


if __name__ == "__main__":
    unittest.main()
