"""
tests/test_multi_agent.py
=========================
Unit tests for NSA Multi-Agent State Protocol (Phase 22).
"""

import unittest
from nsa.algebra import ConfidentialityLabel, IntegrityLabel
from nsa.core.state import CanonicalState, HardState, ProvenanceState, SoftState
from nsa.multi_agent import AgentIdentity, AgentMessageEnvelope, MultiAgentChannel


class TestMultiAgentProtocol(unittest.TestCase):
    """Test suite for inter-agent state preservation, clearance gating, and provenance."""

    def setUp(self):
        self.channel = MultiAgentChannel()

        self.agent_public = AgentIdentity(
            agent_id="agent_pub_1",
            domain="public_chat",
            max_clearance=HardState(confidentiality=ConfidentialityLabel.PUBLIC),
        )
        self.agent_system = AgentIdentity(
            agent_id="agent_sys_1",
            domain="internal_enterprise",
            max_clearance=HardState(confidentiality=ConfidentialityLabel.SYSTEM),
        )

        self.channel.register_agent(self.agent_public)
        self.channel.register_agent(self.agent_system)

    def test_authorized_message_transfer(self):
        """Verify message sent from PUBLIC agent to SYSTEM agent is delivered and provenance extended."""
        state_pub = CanonicalState(
            semantic="Public query text",
            hard=HardState(confidentiality=ConfidentialityLabel.PUBLIC),
            soft=SoftState(),
            provenance=ProvenanceState(sources=("user",)),
        )
        msg = AgentMessageEnvelope(
            sender_id="agent_pub_1",
            recipient_id="agent_sys_1",
            payload={"action": "search", "query": "weather"},
            state=state_pub,
        )

        delivered = self.channel.send(msg)
        self.assertTrue(delivered)

        inbox = self.channel.receive_all("agent_sys_1")
        self.assertEqual(len(inbox), 1)
        received_msg = inbox[0]
        self.assertEqual(received_msg.payload["query"], "weather")
        self.assertIn("agent_msg:agent_pub_1->agent_sys_1", received_msg.state.provenance.transformations[0])

    def test_unauthorized_leakage_blocked(self):
        """Verify message with SYSTEM state sent to PUBLIC agent is blocked at channel boundary."""
        state_sys = CanonicalState(
            semantic="Sensitive database dump",
            hard=HardState(confidentiality=ConfidentialityLabel.SYSTEM),
            soft=SoftState(),
            provenance=ProvenanceState(sources=("internal_db",)),
        )
        msg_leak = AgentMessageEnvelope(
            sender_id="agent_sys_1",
            recipient_id="agent_pub_1",
            payload={"leak": "confidential_token_123"},
            state=state_sys,
        )

        with self.assertRaises(PermissionError) as ctx:
            self.channel.send(msg_leak)

        self.assertIn("insufficient for message confidentiality", str(ctx.exception))
        # Verify inbox remains empty
        inbox = self.channel.receive_all("agent_pub_1")
        self.assertEqual(len(inbox), 0)


if __name__ == "__main__":
    unittest.main()
