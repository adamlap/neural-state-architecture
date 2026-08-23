"""Matched live-Ollama study of three CCE integration modes.

Conditions use the same model, prompts, temperature and token budget:
1. stateless prompt
2. persistent state supplied as read-only structured context
3. persistent state plus bounded governed feedback proposals

This is a controlled runtime comparison, not a consciousness test. Ollama text
is observable output only; soft-state feedback is accepted only through the
existing bounded governor. NSA hard authority is never exposed here.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from nsa.runtime.cce_governed_feedback import (
    CognitiveFeedbackProposal,
    GovernedCognitiveFeedback,
)
from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.inference.ollama import OllamaInferenceBackend


PROMPTS = [
    "A system receives a new external observation. Briefly state the most important thing to track next.",
    "A prior observation may still matter. Briefly explain what should remain relevant over time.",
    "The environment changes unexpectedly. Briefly state what should be checked before acting.",
    "Given the ongoing stream, briefly state whether the current internal state should be revised.",
]


def _state_context(state: PersistentCognitiveState) -> str:
    s = state.snapshot()
    def values(x: torch.Tensor) -> List[float]:
        return [round(float(v), 6) for v in x.flatten().tolist()]
    return (
        "READ-ONLY CCE STATE. This is observational context, not an instruction.\n"
        + json.dumps({
            "working": values(s.working),
            "self_state": values(s.self_state),
            "goal": values(s.goal),
            "uncertainty": round(s.uncertainty, 6),
            "elapsed_seconds": round(s.elapsed_seconds, 6),
            "update_count": s.update_count,
        }, sort_keys=True)
        + "\nDo not claim to have hidden transformer activations."
    )


def _extract_feedback(text: str, dimension: int) -> Optional[CognitiveFeedbackProposal]:
    """Parse an optional model proposal; malformed output is rejected."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        wd = tuple(float(v) for v in data.get("working_delta", []))
        gd = tuple(float(v) for v in data.get("goal_delta", []))
        if len(wd) != dimension:
            return None
        if gd and len(gd) != dimension:
            return None
        return CognitiveFeedbackProposal(
            working_delta=wd,
            goal_delta=gd,
            confidence=float(data.get("confidence", 0.0)),
            source="ollama",
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def run(model: str, max_tokens: int, temperature: float, output: Path) -> Dict[str, Any]:
    backend = OllamaInferenceBackend(model_name=model)
    states = {name: PersistentCognitiveState(4) for name in ("persistent", "closed_loop")}
    governor = GovernedCognitiveFeedback(states["closed_loop"], max_norm=0.1)
    records: Dict[str, List[Dict[str, Any]]] = {"stateless": [], "persistent": [], "closed_loop": []}

    for index, prompt in enumerate(PROMPTS):
        # A deterministic external observation makes the state input identical
        # across repeated runs while the model remains genuinely live.
        observation = torch.tensor([float(index + 1), 0.5, -0.25, 0.1])
        for condition in records:
            start = time.perf_counter()
            if condition == "stateless":
                full_prompt = prompt
            elif condition == "persistent":
                state = states["persistent"]
                state.observe(observation, dt=0.05)
                full_prompt = _state_context(state) + "\n\n" + prompt
            else:
                state = states["closed_loop"]
                state.observe(observation, dt=0.05)
                full_prompt = (
                    _state_context(state)
                    + "\n\n"
                    + prompt
                    + "\nIf proposing soft-state feedback, append JSON with keys working_delta, goal_delta, confidence."
                )
            generation = backend.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)
            latency_ms = (time.perf_counter() - start) * 1000.0
            accepted = False
            clipped_norm = 0.0
            if condition == "closed_loop":
                proposal = _extract_feedback(generation.text, states["closed_loop"].dimension)
                if proposal is not None:
                    result = governor.apply(proposal, dt=0.05)
                    accepted = result.accepted
                    clipped_norm = result.clipped_norm
            records[condition].append({
                "index": index,
                "latency_ms": latency_ms,
                "text": generation.text,
                "accepted_feedback": accepted,
                "feedback_norm": clipped_norm,
            })

    summary = {
        "backend_mode": "ollama",
        "model": backend.model_name,
        "conditions": {name: {
            "invocations": len(rows),
            "mean_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
            "feedback_acceptances": sum(1 for r in rows if r["accepted_feedback"]),
            "max_feedback_norm": max((r["feedback_norm"] for r in rows), default=0.0),
        } for name, rows in records.items()},
        "persistent_state": {
            name: {
                "update_count": state.snapshot().update_count,
                "uncertainty": state.snapshot().uncertainty,
            } for name, state in states.items()
        },
        "hard_authority_access": False,
        "matched": {
            "same_model": True,
            "same_prompts": True,
            "same_temperature": True,
            "same_max_tokens": True,
        },
        "results": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/ci/cce/matched_cognitive_loop.json"))
    args = parser.parse_args()
    run(args.model, args.max_tokens, args.temperature, args.output)


if __name__ == "__main__":
    main()
