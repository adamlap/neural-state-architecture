"""Versioned deterministic codec for complete CanonicalState replay."""
from __future__ import annotations

import json
from typing import Any, Mapping

from nsa.algebra import ConfidentialityLabel, IntegrityLabel
from nsa.core.state import CanonicalState, GoalState, HardState, ProvenanceState, SemanticState, SoftState
from nsa.core.transition import state_digest

SCHEMA_VERSION = "nsa.canonical-state.v1"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try: return value.value
        except Exception: pass
    raise TypeError(f"semantic value is not deterministically JSON serializable: {type(value).__name__}")


def encode_state(state: CanonicalState) -> dict[str, Any]:
    semantic = _jsonable(state.semantic.value)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": {
            "semantic": semantic,
            "hard": {
                "confidentiality": state.hard.confidentiality.name,
                "integrity": state.hard.integrity.name,
                "authorizations": sorted(state.hard.authorizations),
                "license_tier": state.hard.license_tier,
            },
            "soft": {
                "uncertainty": state.soft.uncertainty, "risk": state.soft.risk,
                "confidence": state.soft.confidence, "resource_pressure": state.soft.resource_pressure,
            },
            "provenance": {
                "sources": list(state.provenance.sources), "transformations": list(state.provenance.transformations),
                "evidence_ids": list(state.provenance.evidence_ids), "trust_domain": state.provenance.trust_domain,
                "timestamp": state.provenance.timestamp,
            },
            "goals": {
                "goals": list(state.goals.goals), "active_goal": state.goals.active_goal, "priority": state.goals.priority,
            },
            "step": state.step,
        },
    }
    payload["state_digest"] = state_digest(state)
    return payload


def dumps_state(state: CanonicalState) -> str:
    return json.dumps(encode_state(state), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def decode_state(payload: Mapping[str, Any]) -> CanonicalState:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical state schema: {payload.get('schema_version')!r}")
    raw = payload["state"]
    state = CanonicalState(
        semantic=SemanticState(raw.get("semantic")),
        hard=HardState(
            confidentiality=ConfidentialityLabel[raw["hard"]["confidentiality"]],
            integrity=IntegrityLabel[raw["hard"]["integrity"]],
            authorizations=frozenset(raw["hard"].get("authorizations", [])),
            license_tier=int(raw["hard"].get("license_tier", 0)),
        ),
        soft=SoftState(**{k: float(v) for k, v in raw["soft"].items()}),
        provenance=ProvenanceState(
            sources=tuple(raw["provenance"].get("sources", [])),
            transformations=tuple(raw["provenance"].get("transformations", [])),
            evidence_ids=tuple(raw["provenance"].get("evidence_ids", [])),
            trust_domain=raw["provenance"].get("trust_domain"),
            timestamp=raw["provenance"].get("timestamp"),
        ),
        goals=GoalState(
            goals=tuple(raw["goals"].get("goals", [])),
            active_goal=raw["goals"].get("active_goal"),
            priority=float(raw["goals"].get("priority", 1.0)),
        ),
        step=int(raw["step"]),
    )
    expected = payload.get("state_digest")
    if expected is not None and expected != state_digest(state):
        raise ValueError("canonical state digest mismatch; journal corruption detected")
    return state


def round_trip(state: CanonicalState) -> CanonicalState:
    return decode_state(encode_state(state))


__all__ = ["SCHEMA_VERSION", "decode_state", "dumps_state", "encode_state", "round_trip"]
