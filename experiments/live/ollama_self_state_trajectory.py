"""Collect real Ollama generation trajectories at the NSA runtime boundary.

This experiment is deliberately live-only: it requires a reachable Ollama
server and never falls back to the repository's mock backend. The collected
records contain explicit NSA self-state observations plus normalized action
features; they do not contain or claim transformer hidden activations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime
from nsa.self_model.trajectory import trajectory_step
from nsa.self_state.model import SelfState


def _load_prompts(path: str | None, prompts: list[str]) -> list[str]:
    if path:
        values = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
        prompts.extend(value for value in values if value)
    if not prompts:
        raise ValueError("provide at least one --prompt or --prompt-file")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file")
    parser.add_argument("--output", default="results/live-self-state-trajectory.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    prompts = _load_prompts(args.prompt_file, args.prompt)
    backend = OllamaInferenceBackend(model_name=args.model, mode="ollama")
    runtime = NSATypedRuntime(backend, goal_id="phase19-live-trajectory")
    state = SelfState()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, prompt in enumerate(prompts):
            generation = runtime.generate(
                prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            record = trajectory_step(
                before=state,
                generation=generation,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                model=backend.model_name,
            )
            state = SelfState(**{
                key: record.state_after[key]
                for key in (
                    "confidence", "uncertainty", "perceived_risk",
                    "capability_awareness", "resource_pressure",
                    "goal_progress", "state_prediction_error", "step",
                )
            })
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
            print(f"[{index + 1}/{len(prompts)}] step={state.step} model={backend.model_name}")


if __name__ == "__main__":
    main()
