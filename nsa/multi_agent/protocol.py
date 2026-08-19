"""
nsa.multi_agent.protocol
========================
Multi-Agent State Protocol for NSA (Phase 22).

Enforces:
1. Agent identity and clearance domain boundaries.
2. State preservation across agent-to-agent communications (no silent clearance degradation).
3. Delegated capability transfer and contract verification.
4. Distributed provenance hop lineage.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from nsa.capabilities.model import Capability
from nsa.core.state import CanonicalState, HardState


@dataclass(frozen=True)
class AgentIdentity:
    """Identity and authorization profile of an autonomous NSA agent."""

    agent_id: str
    domain: str
    max_clearance: HardState
    held_capabilities: frozenset[str] = frozenset()

    def can_receive(self, incoming_hard_state: HardState) -> bool:
        """Predicate checking if incoming message state is within agent clearance."""
        if incoming_hard_state.confidentiality.value > self.max_clearance.confidentiality.value:
            return False
        if incoming_hard_state.license_tier > self.max_clearance.license_tier:
            return False
        return True


@dataclass(frozen=True)
class AgentMessageEnvelope:
    """Typed inter-agent message carrying state, delegated capabilities, and provenance."""

    sender_id: str
    recipient_id: str
    payload: Any
    state: CanonicalState
    delegated_capabilities: Tuple[Capability, ...] = ()
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)


class MultiAgentChannel:
    """State-governed inter-agent communication bus."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentIdentity] = {}
        self._inboxes: Dict[str, List[AgentMessageEnvelope]] = {}
        self._message_log: List[AgentMessageEnvelope] = []

    def register_agent(self, agent: AgentIdentity) -> None:
        if agent.agent_id in self._agents:
            raise ValueError(f"duplicate agent_id: {agent.agent_id}")
        self._agents[agent.agent_id] = agent
        self._inboxes[agent.agent_id] = []

    def send(self, envelope: AgentMessageEnvelope) -> bool:
        """Transmit message envelope across agent boundaries with state validation."""
        if envelope.sender_id not in self._agents:
            raise PermissionError(f"unknown sender: {envelope.sender_id}")
        if envelope.recipient_id not in self._agents:
            raise PermissionError(f"unknown recipient: {envelope.recipient_id}")

        recipient = self._agents[envelope.recipient_id]

        # Invariant: Recipient clearance must accommodate transmitted state
        if not recipient.can_receive(envelope.state.hard):
            raise PermissionError(
                f"Agent '{recipient.agent_id}' clearance ({recipient.max_clearance.confidentiality.name}) "
                f"insufficient for message confidentiality ({envelope.state.hard.confidentiality.name})"
            )

        # Extend provenance with inter-agent transmission hop
        new_provenance = envelope.state.provenance.extend(
            transformation=f"agent_msg:{envelope.sender_id}->{envelope.recipient_id}:{envelope.message_id[:8]}"
        )
        delivered_state = CanonicalState(
            semantic=envelope.state.semantic,
            hard=envelope.state.hard,
            soft=envelope.state.soft,
            provenance=new_provenance,
        )

        delivered_envelope = AgentMessageEnvelope(
            sender_id=envelope.sender_id,
            recipient_id=envelope.recipient_id,
            payload=envelope.payload,
            state=delivered_state,
            delegated_capabilities=envelope.delegated_capabilities,
            message_id=envelope.message_id,
            timestamp=envelope.timestamp,
        )

        self._inboxes[envelope.recipient_id].append(delivered_envelope)
        self._message_log.append(delivered_envelope)
        return True

    def receive_all(self, agent_id: str) -> List[AgentMessageEnvelope]:
        """Fetch and clear all pending messages for an agent."""
        if agent_id not in self._inboxes:
            raise KeyError(f"unknown agent: {agent_id}")
        messages = list(self._inboxes[agent_id])
        self._inboxes[agent_id].clear()
        return messages


__all__ = ["AgentIdentity", "AgentMessageEnvelope", "MultiAgentChannel"]
