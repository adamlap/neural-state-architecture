# Research benchmark results

| Directory | Status |
|-----------|--------|
| `research-v1`, `research-v2`, `research-v3` | **Legacy — do not cite.** Produced before the validity fixes in [`research/BENCHMARK_VALIDITY.md`](../research/BENCHMARK_VALIDITY.md) (key-pool wraparound, unequal decoding, invalid v1 control, identical v2 memory conditions, hard-coded stale metric). |
| `research-v4` (untracked, if present) | **Legacy.** Partial 4.0 run, no manifest. |
| `research-v4.1` | Corrected benchmark, qwen2.5:1.5b. See below. |

v1.1, v2.1 and v3.1 code is fixed and tested but has **not** been re-run live; they have no corrected results yet.

## research-v4.1 (qwen2.5:1.5b)

720 episodes: 3 seeds × 2 trials × distractors {10, 250, 1000} × 5 tasks × 8 conditions
(n = 90 per condition). Temperature 0, 32 tokens, `num_ctx` 32768, unique-key generator,
`plain` instruction. Re-run with `make benchmark-research-v4 RESEARCH_V4_MODELS=qwen2.5:1.5b RESEARCH_V4_DISTRACTORS="10 250 1000"`;
rebuild the manifest from `raw.jsonl` with `--summarize-only`. No timeouts, no truncation.
The run was interrupted once by an Ollama HTTP 500 and resumed from the 546 completed episodes.

| Condition | Strict | Lenient | Lenient 95 % CI |
|-----------|-------:|--------:|-----------------|
| raw | 0.00 | 0.00 | 0.00–0.04 |
| transcript | 0.91 | 0.91 | 0.83–0.95 |
| bounded | 0.01 | 0.01 | 0.00–0.06 |
| nsa_history (control) | 0.01 | 0.01 | 0.00–0.06 |
| nsa_state | 0.00 | 0.00 | 0.00–0.04 |
| append_memory | 1.00 | 1.00 | 0.96–1.00 |
| temporal_memory | 0.77 | 0.80 | 0.71–0.87 |
| nsa_temporal_memory | 0.68 | 0.68 | 0.58–0.77 |

Retrieval itself was correct in 100 % of memory-condition episodes, so every memory failure is the
model misusing correct context.

### What this does and does not show

* **State text adds nothing here.** `nsa_state` vs `nsa_history` (same window, no state): 0 vs 1 discordant pairs, p = 1.0.
  Both are ~0 % because a 6-observation window almost never contains the target fact. The serialised
  state does not carry the facts, so it cannot help with these tasks. This is a property of the state
  serialisation, not evidence about state in general.
* **Retrieved memory beats the transcript at long range.** Transcript accuracy falls with length
  (100 % at d=10, 93 % at 250, 80 % at 1000; prompts reach ~31.6k tokens) while memory prompts stay ~100 tokens and
  ~1 s versus ~103 s. `append_memory` vs transcript: 8 vs 0 discordant, Holm p = 0.03.
* **Temporal rendering is worse than append-only for this model.** `temporal_memory` < `append_memory`
  (0 vs 18 discordant, Holm p < 0.001). All 18 extra failures are `supersession_history`: the model answers the
  `CURRENT_VALUE` shown in the row instead of the superseded value. Current-value tasks are 100 %.
  Earlier ad-hoc checks suggested qwen2.5:3b handles this format; that is untested here.
* **The NSA wrapper hurts memory prompts.** `nsa_temporal_memory` < `temporal_memory` (Holm p = 0.005); on
  `supersession_current` it drops to 0.44 because the model copies a distractor value from the
  history window the wrapper adds.
* **Stale retrieval.** `append_memory` retrieves a superseded value in 20 % of episodes (by construction: it returns
  both versions) yet scores 100 %; `temporal_memory` measured 0 %. Only one model and a keyed-lookup oracle
  extractor were tested; none of this speaks to LLM-driven memory formation or larger models.
