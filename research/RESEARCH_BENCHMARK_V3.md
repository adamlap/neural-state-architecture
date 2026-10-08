# Research Benchmark v3 — Temporal Canonical Memory

## Purpose

Benchmark temporal memory after v2 showed that append-only retrieval can find the right memory while still exposing stale superseded versions to a small model.

v3 evaluates explicit version chains, canonical current-value retrieval, and historical retrieval under long distraction sequences.

## Conditions

1. `raw` — question only.
2. `transcript` — complete observation transcript.
3. `bounded` — recent `history_limit` observations.
4. `nsa_state` — current NSARuntime state/cognitive state plus bounded history.
5. `append_memory` — v2-style append-only MemoryStore retrieval.
6. `temporal_memory` — TemporalMemoryStore canonical retrieval.
7. `nsa_temporal_memory` — TemporalMemoryStore retrieval through NSARuntime.

## Tasks

- `recall`
- `interference`
- `supersession_current`
- `supersession_history`
- `multi_current`

## Hypotheses

- **H9:** canonical temporal memory preserves current-value accuracy across long distraction sequences with bounded retrieved context.
- **H10:** canonical retrieval reduces stale-memory retrieval relative to append-only retrieval.
- **H11:** explicit historical retrieval recovers superseded values without contaminating current-value retrieval.
- **H12:** temporal memory benefits do not require cognitive-state machinery; compare `temporal_memory` with `nsa_temporal_memory`.

## Run

Smoke test:

    make benchmark-research-v3 RESEARCH_V3_MODELS="qwen2.5:3b" RESEARCH_V3_SEEDS="7" RESEARCH_V3_DISTRACTORS="10 100" RESEARCH_V3_TRIALS=1

Recommended:

    make benchmark-research-v3 RESEARCH_V3_MODELS="qwen2.5:1.5b qwen2.5:3b qwen2.5:7b" RESEARCH_V3_SEEDS="7 17 37 73 137" RESEARCH_V3_DISTRACTORS="10 50 100 250 500 1000" RESEARCH_V3_TRIALS=5

Outputs:

- `results/research-v3/raw.jsonl`
- `results/research-v3/manifest.json`

## Scientific boundary

Memory extraction is deterministic from the benchmark's structured FACT/UPDATE/DISTRACTOR observations. Therefore v3 isolates temporal storage/retrieval semantics. It does not establish autonomous LLM memory extraction, consciousness, AGI, or general superiority.
