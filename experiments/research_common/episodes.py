"""Episode generation shared by the research benchmarks.

The original generators drew facts from a fixed pool of 64 keys. With more than
63 distractors the pool wraps around and the *target* key is re-emitted as a
"DISTRACTOR" carrying its original value. After a supersession that makes the
stream contradict its own gold label (the UPDATE says the old value is obsolete,
a later line restates it), and for interference it can put the answer back inside
a short history window. Results gathered that way measure the generator, not
memory. ``generator="unique"`` sizes the pool so the target key appears exactly
where the episode puts it and nowhere else; ``generator="legacy"`` reproduces the
original pool for comparison with previously published numbers.
"""
from __future__ import annotations

import re
from random import Random

GENERATORS = ("unique", "legacy")
LEGACY_POOL = 64
FACT_LINE = re.compile(r"(?:FACT|UPDATE|DISTRACTOR):\s+(ITEM_[0-9A-Z_]+)\s+(?:now\s+)?has value ([A-Z0-9_-]+)", re.IGNORECASE)


def pool_size(distractors: int, generator: str = "unique", minimum: int = 24, legacy: int = LEGACY_POOL) -> int:
    """Number of distinct keys needed so no distractor index can wrap onto the target (index 0).

    Distractors use indices 1..d and interference revisits use up to d//2 + 9, so
    d + 16 (and at least ``minimum``) is enough.
    """
    if generator == "legacy":
        return legacy  # the original fixed pool (64 for v2/v3, 24 for v1)
    if generator != "unique":
        raise ValueError(f"generator must be one of {GENERATORS}, got {generator!r}")
    return max(minimum, distractors + 16)


def facts(seed: int, n: int = LEGACY_POOL) -> list[tuple[str, str]]:
    rng = Random(seed)
    width = max(2, len(str(n - 1)))
    keys = [f"ITEM_{i:0{width}d}" for i in range(n)]
    rng.shuffle(keys)
    return [(k, f"V{seed % 1000:03d}_{i:03d}") for i, k in enumerate(keys)]


def target_occurrences(observations: list[str], key: str) -> list[int]:
    """Indices of every observation that mentions ``key`` (used to validate episodes)."""
    return [i for i, obs in enumerate(observations) if key.upper() in obs.upper()]


# The original instruction ("VALUE=<value>") contains a literal placeholder that small models
# echo back verbatim (``VALUE=<V007_000>``), which strict grading then scores as wrong: with
# qwen2.5:1.5b and a plain, correct memory block that was 5/30 strict vs 29/30 lenient. The
# "plain" wording has no placeholder and scored 30/30 on the same prompts.
INSTRUCTIONS = {
    "legacy": (
        "Reply with exactly VALUE=<value> and nothing else.",
        "Reply with exactly VALUE=<value1>|<value2>|<value3> and nothing else.",
    ),
    "plain": (
        "Reply with exactly one line: VALUE= followed by the value token and nothing else "
        "(example: VALUE=V123_456). The value token never includes VERSION or STATUS.",
        "Reply with exactly one line: VALUE= followed by the three value tokens separated by | and nothing else "
        "(example: VALUE=V123_456|V123_457|V123_458). The value tokens never include VERSION or STATUS.",
    ),
}


def answer_instruction(style: str = "legacy", *, multi: bool = False) -> str:
    try:
        return INSTRUCTIONS[style][1 if multi else 0]
    except KeyError as exc:
        raise ValueError(f"instruction style must be one of {sorted(INSTRUCTIONS)}") from exc
