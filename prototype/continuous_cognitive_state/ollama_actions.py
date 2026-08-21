"""Dynamic Ollama reasoning/action-proposal adapter.

The model returns structured proposals; the adapter never executes them.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .action import ActionProposal
from .state import CognitiveState


class OllamaActionReasoner:
    def __init__(self, model: str, base_url: str = "http://127.0.0.1:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def propose(self, state: CognitiveState, context: str | None = None) -> ActionProposal | None:
        prompt = {
            "system": "You are a reasoning component. Produce one JSON action proposal or null. You do not have authority to execute actions. The capability name is a request only.",
            "state": {
                "tick": state.tick,
                "cognitive_context": state.cognitive_context,
                "self_state": state.self_state.__dict__,
                "goals": [g.__dict__ for g in state.goals if g.active],
                "recent_memory": [m.__dict__ for m in state.memories[-8:]],
            },
            "input": context,
            "schema": {"capability": "string", "payload": "object", "confidence": "number 0..1", "risk": "number 0..1", "provenance": "string", "reversible": "boolean"},
        }
        request = Request(f"{self.base_url}/api/generate", data=json.dumps({"model": self.model, "prompt": json.dumps(prompt), "format": "json", "stream": False}).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode())
        text = result.get("response", "null")
        if text.strip().lower() == "null":
            return None
        value = json.loads(text)
        if not isinstance(value, dict) or not isinstance(value.get("capability"), str):
            raise ValueError("Ollama returned an invalid action proposal")
        return ActionProposal(value["capability"], value.get("payload", {}), float(value.get("confidence", 0.0)), float(value.get("risk", 1.0)), str(value.get("provenance", "model")), bool(value.get("reversible", True)))
