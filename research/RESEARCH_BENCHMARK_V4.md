# Research Benchmark v4 — Temporal Semantics and Retrieval Attribution

> **Validity notice:** results from the original (pre-fix) run are unreliable — see [BENCHMARK_VALIDITY.md](BENCHMARK_VALIDITY.md). Use the corrected benchmark version.


## Purpose

v4 follows the v3 smoke result by separating two questions:

1. Did the memory subsystem retrieve the correct version?
2. Did the language model answer correctly from that retrieved context?

It also makes temporal semantics explicit in the model-facing memory representation.

## Temporal representation

Current-value retrieval renders entries as:

    - ITEM_XX: VALUE VERSION=2 STATUS=CURRENT

Historical retrieval renders superseded entries as:

    - ITEM_XX: OLD_VALUE VERSION=1 STATUS=SUPERSEDED CURRENT_VALUE=NEW_VALUE

This prevents the model from having to infer temporal meaning from an unlabeled key/value pair.

## Metrics

Each memory condition records:

- `memory_retrieval_correct` — whether the deterministic retrieval layer supplied the expected version/value.
- `correct` — whether the model produced the expected final answer.
- `retrieval_answer_gap` — the difference between retrieval correctness and answer correctness.

This distinguishes storage/retrieval failures from failures in the model's use of correct memory.

## Conditions and tasks

v4 retains the v3 conditions and tasks:

- Conditions: `raw`, `transcript`, `bounded`, `nsa_state`, `append_memory`, `temporal_memory`, `nsa_temporal_memory`.
- Tasks: `recall`, `interference`, `supersession_current`, `supersession_history`, `multi_current`.

## Smoke test

    make benchmark-research-v4 RESEARCH_V4_MODELS="qwen2.5:3b" RESEARCH_V4_SEEDS="7" RESEARCH_V4_DISTRACTORS="10 100" RESEARCH_V4_TRIALS=1

Recommended full run:

    make benchmark-research-v4 RESEARCH_V4_MODELS="qwen2.5:1.5b qwen2.5:3b qwen2.5:7b" RESEARCH_V4_SEEDS="7 17 37 73 137" RESEARCH_V4_DISTRACTORS="10 50 100 250 500 1000" RESEARCH_V4_TRIALS=5

Outputs:

- `results/research-v4/raw.jsonl`
- `results/research-v4/manifest.json`

## Hypotheses

- **H13:** temporal memory exposes explicit CURRENT versus SUPERSEDED semantics to the model.
- **H14:** memory retrieval correctness can be measured independently from final model-answer correctness.
- **H15:** historical retrieval remains correct when the previous value is explicitly labeled SUPERSEDED and the current value is provided as CURRENT_VALUE.
- **H16:** temporal memory retains its accuracy/latency advantage over cognitive-state-only context without requiring cognitive machinery.

## Scientific boundary

Memory extraction remains deterministic from structured FACT/UPDATE/DISTRACTOR observations. v4 therefore isolates temporal storage/retrieval semantics and does not establish autonomous LLM memory formation, consciousness, AGI, or general superiority.
