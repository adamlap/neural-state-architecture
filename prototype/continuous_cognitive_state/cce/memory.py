"""Persistent typed memory for CCE."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from .state import CCEState


class JSONMemory:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self, state: CCEState) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        state.memories = data.get("memories", [])
        state.goals = data.get("goals", [])
        state.cognitive_context = data.get("cognitive_context", "")

    def save(self, state: CCEState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"memories": state.memories, "goals": state.goals, "cognitive_context": state.cognitive_context}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, default=asdict), encoding="utf-8")
