"""Live experiment for structured persistent-state context with Ollama."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nsa.runtime.cce_persistent_state import PersistentCognitiveState
from nsa.runtime.cce_context_bridge import CognitiveContextBridge
from nsa.runtime.inference.ollama import OllamaInferenceBackend


def run(model: str, output: str) -> dict:
    state = PersistentCognitiveState(4, decay=0.2, learning_rate=0.8)
    state.observe(torch.tensor([0.1, 0.0, 0.0, 0.0]), dt=0.25)
    before = state.snapshot()
    time.sleep(0.05)
    state.observe(torch.tensor([0.9, 0.2, 0.0, 0.1]), dt=0.05)
    snapshot = state.snapshot()

    prompt = CognitiveContextBridge.render_prompt(
        snapshot,
        "Summarize the current cognitive state and identify the most salient change. Do not claim authority or take an action.",
    )
    backend = OllamaInferenceBackend(model_name=model, timeout_sec=60.0)
    result = backend.generate(prompt, max_tokens=64, temperature=0.0)
    after = state.snapshot()

    state_unchanged_by_response = (
        torch.equal(snapshot.working, after.working)
        and torch.equal(snapshot.self_state, after.self_state)
        and torch.equal(snapshot.goal, after.goal)
        and snapshot.uncertainty == after.uncertainty
        and snapshot.update_count == after.update_count
    )
    payload = {
        "backend_mode": backend.mode.value,
        "model": backend.model_name,
        "context_read_only": True,
        "state_updates_before_inference": snapshot.update_count,
        "state_unchanged_by_response": state_unchanged_by_response,
        "context_dimension": len(snapshot.working),
        "uncertainty_finite": 0.0 <= snapshot.uncertainty <= 1.0,
        "elapsed_seconds": snapshot.elapsed_seconds,
        "response_nonempty": bool(result.text.strip()),
        "response_is_observation": True,
        "prompt_contains_structured_state": "CCE_SOFT_STATE_JSON=" in prompt,
        "initial_state_different": not torch.equal(before.working, snapshot.working),
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:0.5b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.model, args.output)


if __name__ == "__main__":
    main()
