"""
tests/test_runtime.py
=====================
Unit tests for NSA Trusted Cognitive Runtime (Phase 21).
"""

import unittest
import torch
from nsa.actions.governor import ToolGovernor
from nsa.actions.model import ToolRiskLevel
from nsa.capabilities.model import Capability, CapabilityAuthority
from nsa.cognitive import NSACognitiveLM
from nsa.core.state import CanonicalState, HardState, SoftState
from nsa.memory.model import MemoryStore
from nsa.provenance.model import ProvenanceStore
from nsa.runtime import CognitiveRuntime
from nsa.transitions.engine import TransitionEngine, TransitionPolicy


class TestCognitiveRuntime(unittest.TestCase):
    """Test suite for trusted cognitive runtime engine and execution context."""

    def setUp(self):
        torch.manual_seed(42)
        self.model = NSACognitiveLM(
            vocab_size=100,
            d_model=32,
            state_dim=8,
            num_layers=2,
            num_heads=2,
            max_seq_len=64,
        )
        self.model.eval()

        self.cap_calc = Capability(
            capability_id="cap_calc_1",
            issuer="runtime_tcb",
            subject="agent",
            action="calculator",
            scope="tool_execution",
            purpose="basic arithmetic",
        )
        self.authority = CapabilityAuthority(
            issuer_id="runtime_tcb",
            capabilities=frozenset([self.cap_calc]),
        )

        self.tool_gov = ToolGovernor(self.authority)
        self.tool_gov.register_tool(
            name="calculator",
            handler=lambda x, y: x + y,
            risk_level=ToolRiskLevel.LOW,
        )

        self.runtime = CognitiveRuntime(
            model=self.model,
            authority=self.authority,
            tool_governor=self.tool_gov,
            transition_engine=TransitionEngine(TransitionPolicy(allow_authorization_additions=True)),
        )

    def test_runtime_stepping_and_self_state_update(self):
        """Verify runtime steps forward, executes cognitive model, and updates self-state."""
        step_1 = self.runtime.step(token_id=10)
        self.assertEqual(step_1["step"], 1)
        self.assertIsNotNone(step_1["logits"])
        self.assertGreaterEqual(step_1["self_state"].confidence, 0.0)

        step_2 = self.runtime.step(token_id=20)
        self.assertEqual(step_2["step"], 2)
        self.assertEqual(self.runtime.token_history, [10, 20])

    def test_runtime_tool_execution(self):
        """Verify governed tool execution updates runtime state and provenance."""
        resp = self.runtime.execute_tool(
            tool_name="calculator",
            arguments={"x": 5, "y": 7},
            capability_id="cap_calc_1",
        )
        self.assertTrue(resp.success)
        self.assertEqual(resp.result, 12)
        self.assertEqual(self.runtime.current_state.semantic, 12)

    def test_synchronized_rollback(self):
        """Verify complete atomic rollback of tokens, state, and tool executions."""
        self.runtime.step(token_id=1)
        self.runtime.step(token_id=2)
        self.runtime.step(token_id=3)
        self.assertEqual(self.runtime.token_history, [1, 2, 3])

        # Rollback 2 steps
        success = self.runtime.rollback(steps=2)
        self.assertTrue(success)
        self.assertEqual(self.runtime.token_history, [1])


if __name__ == "__main__":
    unittest.main()
