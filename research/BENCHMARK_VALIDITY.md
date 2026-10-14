# Benchmark validity notes (v1–v4)

An audit of the research-v1…v4 benchmarks found problems that make the
**original published numbers unreliable**. The corrected versions are v1.2, v2.2,
v3.1 and v4.2. Results produced before the fixes are kept in `results/research-v1`
…`results/research-v4` for provenance only; do not cite them.

## Problems found and how each is fixed

| # | Problem | Effect | Fix |
|---|---------|--------|-----|
| 1 | Fact generator drew keys from a fixed pool of 64 (24 in v1). At d ≥ 64 the target key reappeared as a `DISTRACTOR` carrying its **old** value. | Episodes could be unanswerable or contradict the label; interference episodes in v1 leaked the answer. | `research_common.episodes`: unique-key pool `max(24, d+16)`. `--generator legacy` reproduces the old behaviour for comparison. Invariant tests assert the target key occurs the expected number of times. |
| 2 | NSA conditions ran at backend defaults (temperature 0.7, 256 tokens, variable `num_ctx`); direct conditions ran at temperature 0, 32 tokens. | Condition differences were confounded with decoding differences. | `FixedDecodingBackend`: temperature 0, 32 tokens, `num_ctx=32768` for every condition; `OllamaInferenceBackend.generate(num_ctx=…)`. |
| 3 | v1 control `nsa_no_state` disabled state **and history** (`include_state_in_prompt=False` hid both), so it was effectively "no context". | It could not isolate what the serialised state adds; the state-vs-control gap measured "any context vs none". | v1.1: control keeps the same history window and drops only the state text (`include_history_in_prompt=True`). v4.1 adds the matched `nsa_history` condition. |
| 4 | v2 `nsa_memory` was built with `include_state_in_prompt=False`, making its prompt byte-identical to `memory_no_cognitive` (108/108 identical responses). | H6 (cognitive state adds nothing / hurts) could not have failed. | v2.1 includes state in the prompt, so the two conditions actually differ. |
| 5 | Strict exact-match grading only; instruction `VALUE=<value>` contained a placeholder that small models echo (`VALUE=<V007_000>`). | A correct memory block scored 5/30 strict vs 29/30 lenient with qwen2.5:1.5b. | `--instruction plain` (no placeholder) is the default from v1.2 / v2.2 / v3.1 (opt-in) / v4.1, and **strict** and **lenient** accuracy are both recorded (v1.2, v2.2, v4.x). v1.1 and v2.1 (legacy instruction) are superseded: in v2.1 `memory_no_cognitive` scored 0 % strict with 100 % correct retrieval. |
| 6 | `stale_memory_retrieved` hard-coded `False` for temporal memory. | "Temporal memory eliminates stale retrieval" was true by construction. | Measured from `TemporalMemoryStore` conflict metadata (`reasserts_version`, `overrides_update`). |
| 7 | Prompt length was reconstructed after the call with `agent._prompt()`. | Could differ from what the model received. | Prompt, token counts, truncation and timeouts are recorded from the backend call. |
| 8 | Partial v3/v4 runs, no manifest; resume ignored configuration. | Silent mixing of incompatible episodes. | `config.json` sidecar guards `--resume`; manifest always written; `--summarize-only`. |
| 9 | Counts only; no uncertainty. | Small differences read as findings. | v4.x reports Wilson 95 % CIs, exact McNemar on paired episodes and Holm correction. v1–v3 still report point estimates only. |

## What the benchmarks still do **not** show

* Memory extraction is a deterministic keyed lookup oracle. The benchmarks test
  storage/retrieval semantics and prompt format, **not** an LLM deciding what to
  remember.
* Sample sizes are small (see each manifest's `n`); intervals are wide.
* Results are for small local models (qwen2.5 1.5b/3b). They do not support
  claims about larger models.
* Prompt shape matters: a good result for a retrieval prompt says little about the
  serialised-state prompt, and vice versa. `nsa_history` exists to separate them.
