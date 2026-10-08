# NSA Research Benchmark v2 — Selective Persistent Memory

## Purpose

Benchmark v1 showed that the current compact cognitive/state prompt is measurably better than NSA without state, but it loses long-range facts and is much slower than bounded context. v2 isolates the missing mechanism: **persistent selective retrieval**.

The benchmark uses the repository's existing typed MemoryStore. Memory extraction is deliberately deterministic from the benchmark's structured FACT/UPDATE observations. That makes the first experiment about storage/retrieval, not about whether a language model can correctly extract memories. Extraction quality should be tested separately.

## Hypotheses

- **H5 — Selective retention:** selective memory should retain task-relevant facts across long distractor sequences while using substantially less prompt context than a full transcript.
- **H6 — Memory vs cognition:** the retrieval benefit should persist without cognitive-state machinery; nsa_memory should be comparable to memory_no_cognitive if memory, rather than cognitive prompting, is responsible for the gain.
- **H7 — Supersession:** latest-value retrieval should preserve current values after explicit updates without increasing stale-memory retrieval as distractor count grows.
- **H8 — Efficiency:** accuracy gains must be reported jointly with prompt size, latency, retrieval-hit rate, and stale-memory retrieval rate.

## Conditions

1. raw — question only.
2. transcript — complete observation transcript.
3. bounded — recent history_limit observations only.
4. nsa_state — current NSARuntime state/cognitive context plus bounded history.
5. memory_no_cognitive — typed MemoryStore + deterministic structured extraction + selective retrieval, with no cognitive substrate.
6. nsa_memory — the same memory/retrieval path, with the NSARuntime cognitive substrate enabled.

The key comparisons are memory_no_cognitive vs bounded/transcript, then nsa_memory vs memory_no_cognitive.

## Stress design

Default distractor counts are 10, 50, 100, 250, 500, and 1000. The target fact remains identifiable while irrelevant observations grow. The benchmark also revisits distractor keys for the interference task.

For supersession, the sequence is FACT(A=old) -> UPDATE(A=current) -> distractors. The memory store remains append-only, so the benchmark explicitly checks whether latest matching memory wins rather than deleting history.

## Metrics

Every raw row records model, seed, task, distractor count, trial and condition; expected/predicted value and exact-match correctness; latency and prompt-size proxy; number of stored memories and retrieved memories; retrieval-hit rate; whether the expected value was present in retrieved memory; and stale-memory retrieval flag.

The manifest reports per-condition aggregates and per-task/distractor cells.

## Running

Smoke test:

    make benchmark-research-v2 RESEARCH_MODELS="qwen2.5:3b" RESEARCH_SEEDS="7" RESEARCH_DISTRACTORS="10 50" RESEARCH_TRIALS=1

Recommended first run:

    make benchmark-research-v2 RESEARCH_MODELS="qwen2.5:1.5b qwen2.5:3b qwen2.5:7b" RESEARCH_SEEDS="7 17 37 73 137" RESEARCH_DISTRACTORS="10 50 100 250 500 1000" RESEARCH_TRIALS=5

For a clean comparison with v1, keep RESEARCH_HISTORY_LIMIT=6.

Results are written to results/research-v2/manifest.json and results/research-v2/raw.jsonl.

## Interpretation boundary

A positive result supports selective persistent retrieval under the benchmark's structured-memory assumption. It does not establish autonomous memory formation, consciousness, AGI, or general superiority. A failure of the memory condition is also useful: it would indicate that simply adding a persistent store is insufficient and that retrieval/update policy needs work.
