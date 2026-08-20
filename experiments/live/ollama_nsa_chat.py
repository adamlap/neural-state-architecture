"""Live Ollama chat through the canonical NSA typed-state runtime.

This script uses the actual Ollama HTTP backend.  It does not simulate model
inference and does not claim hidden-state access.  NSA is the trusted runtime
control plane around the real model: canonical state is injected as read-only
context, generation is performed by Ollama, and the trusted runtime commits
the post-generation state observation.
"""

from __future__ import annotations

import argparse

from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.typed_runtime import NSATypedRuntime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--prompt")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    backend = OllamaInferenceBackend(model_name=args.model, mode="ollama")
    runtime = NSATypedRuntime(backend, goal_id="live-chat")

    if args.prompt:
        result = runtime.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(result.output.text)
        print("\nNSA state:")
        print(runtime.inspect())
        return

    print(f"NSA-wrapped Ollama model: {backend.model_name}")
    print("Type /state to inspect canonical state; /reset to reset session state; /quit to exit.")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt == "/quit":
            break
        if prompt == "/state":
            print(runtime.inspect())
            continue
        if prompt == "/reset":
            runtime.reset()
            print("NSA session state reset; runtime-owned authority preserved.")
            continue
        if not prompt:
            continue

        result = runtime.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\nNSA-Ollama> {result.output.text}")
        print(
            f"[step={result.state.state.temporal_state.step_index} "
            f"provenance={result.state.state.provenance_state.record_id}]"
        )


if __name__ == "__main__":
    main()
