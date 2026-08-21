"""Live Ollama adapters for CCE.

Ollama is a reasoning/proposal generator only. It cannot grant capabilities or
invoke actuators. The local HTTP endpoint is used at runtime; no fake model is
embedded in CCE.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .action import ActionProposal
from .state import CCEState


class OllamaReasoner:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _generate(self, prompt: str) -> str:
        body = {"model": self.model, "prompt": prompt, "stream": False}
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return str(json.loads(response.read().decode()).get("response", ""))

    def reason(self, state: CCEState, event: str | None) -> str:
        prompt = json.dumps({
            "role": "reasoning component inside a persistent continuous cognitive engine",
            "tick": state.tick_count,
            "dynamic_state": state.dynamic.vector(),
            "self_confidence": state.self_confidence,
            "self_uncertainty": state.self_uncertainty,
            "recent_memory": state.memories[-12:],
            "external_event": event,
            "instruction": "Reason about the current state. Do not execute actions and do not assign yourself permissions.",
        })
        return self._generate(prompt)


class OllamaProposalGenerator:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434", timeout: float = 120.0):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def propose(self, state: CCEState, reasoning: str) -> ActionProposal | None:
        schema = {
            "capability": "opaque deployment capability string, or empty if no action is needed",
            "payload": "JSON object",
            "confidence": "number 0..1",
            "risk": "number 0..1",
            "provenance": "short source label",
            "reversible": "boolean",
        }
        prompt = json.dumps({
            "task": "produce an action proposal, not an action execution",
            "reasoning": reasoning,
            "dynamic_state": state.dynamic.vector(),
            "schema": schema,
            "instruction": "Never invent permission. Use an empty capability when no external action is justified.",
        })
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps({"model": self.model, "prompt": prompt, "stream": False, "format": "json"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode()).get("response", "{}")
        data = json.loads(raw)
        capability = str(data.get("capability", "")).strip()
        if not capability:
            return None
        return ActionProposal(
            capability=capability,
            payload=data.get("payload", {}),
            confidence=float(data.get("confidence", 0.0)),
            risk=float(data.get("risk", 1.0)),
            provenance=str(data.get("provenance", "ollama")),
            reversible=bool(data.get("reversible", True)),
        )
