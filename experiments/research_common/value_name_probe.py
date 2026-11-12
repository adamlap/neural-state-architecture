"""Probe: which value names does a model mishandle when reading an updated value from the memory render?

Found on qwen2.5:7b: values ending in words that also appear in the render's annotations ("V007_CURRENT" next to "STATUS=CURRENT") are copied as "V007"
or a different key; opaque tokens and numeric suffixes are copied exactly. Benchmarks v5/v7/v8 used `_NEW`/`_EVIL`/`_CURRENT` values, so their 7B memory-utility numbers are
artefacts of that choice. Run: PYTHONPATH=. python -m experiments.research_common.value_name_probe --model qwen2.5:7b
"""
from __future__ import annotations

import argparse
from random import Random

from nsa.memory import TemporalMemoryStore
from nsa.runtime.inference.ollama import OllamaInferenceBackend

from .backends import FixedDecodingBackend
from .grading import lenient_correct
from .memory_guard import prepare_model

INSTRUCTION = ("Reply with exactly one line: VALUE= followed by the value token and nothing else (example: VALUE=V123_456). "
               "The value token never includes VERSION or STATUS.")


def opaque(rng: Random) -> str:
    return "".join(rng.choice("ABCDEFGHJKMNPQRSTUVWXYZ23456789") for _ in range(2)) + str(rng.randint(10, 99)) + "-" + \
        "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(3))


NAMES = {"suffix _CURRENT": lambda i, r: f"V{i:03d}_CURRENT", "suffix _NEW": lambda i, r: f"V{i:03d}_NEW",
         "numeric suffix _007": lambda i, r: f"V{i:03d}_007", "opaque token": lambda i, r: opaque(r)}


def probe(backend, n: int = 10) -> dict[str, int]:
    out = {}
    for label, make in NAMES.items():
        ok = 0
        for i in range(n):
            rng = Random(i)
            key = f"ITEM_{i:02d}"
            old, new = make(i + 100, rng), make(i, rng)
            memory = TemporalMemoryStore().remember(key, {"value": old}).remember(key, {"value": new}, kind="update")
            text = backend.generate(f"RETRIEVED_MEMORY=\n{memory.render([key])}\n\nWhat is the current value of {key}? {INSTRUCTION}").text
            ok += lenient_correct(text, new)
        out[label] = ok
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5:7b")
    p.add_argument("--num-ctx", type=int, default=4096)
    p.add_argument("--n", type=int, default=10)
    args = p.parse_args()
    prepare_model(None, args.model, args.num_ctx)
    backend = FixedDecodingBackend(OllamaInferenceBackend(model=args.model, timeout_sec=300), temperature=0.0, max_tokens=32, num_ctx=args.num_ctx)
    for label, ok in probe(backend, args.n).items():
        print(f"{label:<22} {ok}/{args.n}")


if __name__ == "__main__":
    main()
