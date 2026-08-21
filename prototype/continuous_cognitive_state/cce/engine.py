"""Continuous Cognitive Engine runtime.

The engine owns the live cognitive loop. It never mutates nsa/; NSA remains an
external trusted policy substrate. Model inference, state evolution, and
actuation are separate interfaces so real integrations can be supplied by a
deployment without replacing them with simulations.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Protocol

from ..action import ActionProposal, GovernanceDecision
from ..dynamics import DynamicalState, evolve_state
from ..self_model import SelfModel
from ..state import CognitiveState
from .governor import CCEGovernor


class Reasoner(Protocol):
    def reason(self, state: CognitiveState, event: str | None) -> str: ...


class ProposalGenerator(Protocol):
    def propose(self, state: CognitiveState, reasoning: str) -> ActionProposal | None: ...


class Actuator(Protocol):
    async def execute(self, proposal: ActionProposal) -> object: ...


@dataclass
class CCEConfig:
    tick_hz: float = 1.0
    max_event_queue: int = 256
    state_dt: float = 0.1


@dataclass
class CCEEvent:
    payload: str
    timestamp: float = field(default_factory=time.time)
    source: str = "external"


class ContinuousCognitiveEngine:
    """Live asynchronous cognitive loop with NSA governance at the action boundary."""

    def __init__(
        self,
        *,
        state: CognitiveState,
        reasoner: Reasoner,
        governor: CCEGovernor,
        proposal_generator: ProposalGenerator,
        actuator: Actuator | None = None,
        self_model: SelfModel | None = None,
        config: CCEConfig | None = None,
    ) -> None:
        self.state = state
        self.reasoner = reasoner
        self.governor = governor
        self.proposal_generator = proposal_generator
        self.actuator = actuator
        self.self_model = self_model or SelfModel()
        self.config = config or CCEConfig()
        self.events: asyncio.Queue[CCEEvent] = asyncio.Queue(maxsize=self.config.max_event_queue)
        self.running = False

    async def ingest(self, payload: str, *, source: str = "external") -> None:
        """Inject a live sensory/input event without stopping the internal clock."""
        await self.events.put(CCEEvent(payload=payload, source=source))

    def _evolve(self, drive: float) -> None:
        """Advance the dynamical state even when drive == 0."""
        self.state.cognitive_context = evolve_state(
            self.state.cognitive_context,
            drive=drive,
            dt=self.config.state_dt,
        )

    async def tick(self) -> GovernanceDecision | None:
        self.state.tick_once()
        event: CCEEvent | None = None
        if not self.events.empty():
            event = self.events.get_nowait()

        prediction = self.self_model.predict(self.state)
        self._evolve(1.0 if event else 0.0)
        event_text = event.payload if event else None
        reasoning = self.reasoner.reason(self.state, event_text)
        self.state.last_output = reasoning
        self.self_model.update(self.state, prediction)

        proposal = self.proposal_generator.propose(self.state, reasoning)
        if proposal is None:
            return None

        source = self.governor.default_source_state()
        target = self.governor.target_from_hard_state(
            hard=self.state_to_hard_state(proposal),
            confidence=proposal.confidence,
            provenance=frozenset({proposal.provenance}),
        )
        decision = self.governor.evaluate(proposal, source, target)
        if decision.allowed and self.actuator is not None:
            await self.actuator.execute(proposal)
        return decision

    @staticmethod
    def state_to_hard_state(proposal: ActionProposal):
        from nsa.algebra import HardStateVector
        return HardStateVector(
            authorizations=frozenset({proposal.capability}),
            license_tier=0,
        )

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        """Run continuously until cancelled or an optional stop event is set."""
        self.running = True
        period = 1.0 / max(self.config.tick_hz, 1e-6)
        try:
            while stop is None or not stop.is_set():
                started = time.monotonic()
                await self.tick()
                elapsed = time.monotonic() - started
                await asyncio.sleep(max(0.0, period - elapsed))
        finally:
            self.running = False
