"""Continuous Cognitive Engine runtime.

CCE is an isolated runtime layer. It consumes NSA's public algebra primitives
but never edits the core NSA implementation. Model inference, state evolution,
and actuation are separate live interfaces.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol

from .action import ActionProposal, GovernanceDecision
from .governor import CCEGovernor
from .state import CCEState


class Reasoner(Protocol):
    def reason(self, state: CCEState, event: str | None) -> str: ...


class ProposalGenerator(Protocol):
    def propose(self, state: CCEState, reasoning: str) -> ActionProposal | None: ...


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
    """Live asynchronous cognitive loop with NSA governance at its action boundary."""

    def __init__(self, *, state: CCEState, reasoner: Reasoner, governor: CCEGovernor,
                 proposal_generator: ProposalGenerator, actuator: Actuator | None = None,
                 config: CCEConfig | None = None) -> None:
        self.state = state
        self.reasoner = reasoner
        self.governor = governor
        self.proposal_generator = proposal_generator
        self.actuator = actuator
        self.config = config or CCEConfig()
        self.events: asyncio.Queue[CCEEvent] = asyncio.Queue(maxsize=self.config.max_event_queue)
        self.running = False

    async def ingest(self, payload: str, *, source: str = "external") -> None:
        await self.events.put(CCEEvent(payload=payload, source=source))

    async def tick(self) -> GovernanceDecision | None:
        self.state.tick()
        event = self.events.get_nowait() if not self.events.empty() else None
        if event is not None:
            self.state.observe(event.payload)
        # Autonomous dynamics continue even with no event.
        self.state.dynamic.tick(drive=1.0 if event else 0.0, dt=self.config.state_dt)
        event_text = event.payload if event else None
        reasoning = self.reasoner.reason(self.state, event_text)
        self.state.last_reasoning = reasoning
        proposal = self.proposal_generator.propose(self.state, reasoning)
        if proposal is None:
            return None

        decision = self.governor.evaluate(
            proposal,
            self.governor.source_state(self.state),
            self.governor.target_state(proposal),
        )
        if decision.allowed:
            if self.actuator is None:
                return GovernanceDecision("HOLD", "NSA allowed proposal but no real actuator is attached", proposal)
            await self.actuator.execute(proposal)
        return decision

    async def run(self, *, stop: asyncio.Event | None = None) -> None:
        self.running = True
        period = 1.0 / max(self.config.tick_hz, 1e-6)
        try:
            while stop is None or not stop.is_set():
                started = time.monotonic()
                await self.tick()
                await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))
        finally:
            self.running = False
