"""Adapters from NSA cognitive state to residency features."""
from __future__ import annotations
from typing import Any, Mapping

def cognitive_state_features(state: Any) -> dict[str, object]:
    """Convert read-only CCE state into bounded residency features."""
    if state is None:
        return {"uncertainty": 0.5, "tags": []}
    if isinstance(state, Mapping):
        uncertainty = float(state.get("uncertainty", 0.5))
        tags = state.get("tags", ())
        if not isinstance(tags, (list, tuple, set, frozenset)): tags = ()
        return {"uncertainty": max(0.0, min(1.0, uncertainty)), "tags": [str(x) for x in tags]}
    uncertainty = float(getattr(state, "uncertainty", 0.5))
    goal = getattr(state, "goal", ())
    working = getattr(state, "working", ())
    tags = []
    if goal: tags.append("goal-active")
    if working: tags.append("working-state")
    if uncertainty >= 0.75: tags.append("high-uncertainty")
    elif uncertainty <= 0.25: tags.append("low-uncertainty")
    return {"uncertainty": max(0.0, min(1.0, uncertainty)), "tags": tags}
