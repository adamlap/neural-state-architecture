"""Minimal state-aware runtime facade; backend adapters can plug into this boundary."""
from dataclasses import dataclass, field
from typing import Any, Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str, *, state: dict[str, Any] | None = None) -> str: ...


@dataclass
class AgentResult:
    text: str
    state: dict[str, Any]
    trace: list[dict[str, Any]] = field(default_factory=list)


class NSA:
    """Small public facade for running an LLM with explicit state."""

    def __init__(self, backend: ModelBackend, *, initial_state: dict[str, Any] | None = None):
        self.backend = backend
        self.state: dict[str, Any] = dict(initial_state or {})
        self.trace: list[dict[str, Any]] = []

    def step(self, prompt: str) -> AgentResult:
        text = self.backend.generate(prompt, state=self.state)
        event = {"prompt": prompt, "response": text}
        self.trace.append(event)
        return AgentResult(text=text, state=dict(self.state), trace=list(self.trace))

    def run(self, prompt: str) -> AgentResult:
        return self.step(prompt)
