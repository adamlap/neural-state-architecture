# Research benchmark results

| Directory | Status |
|-----------|--------|
| `research-v1`, `research-v2`, `research-v3` | **Legacy — do not cite.** Produced before the validity fixes in [`research/BENCHMARK_VALIDITY.md`](../research/BENCHMARK_VALIDITY.md) (key-pool wraparound, unequal decoding, invalid v1 control, identical v2 memory conditions, hard-coded stale metric). |
| `research-v4` (untracked, if present) | **Legacy.** Partial 4.0 run, no manifest. |
| `research-v4.1` | Corrected benchmark (qwen2.5:1.5b). Superseded by 4.2 for the temporal and NSA-memory conditions; its history-query result is what motivated the 4.2 render change. |
| `research-v4.2`, `research-v4.2-3b` | Current. See below. |

v1.1, v2.1 and v3.1 live runs (qwen2.5:1.5b, reduced grids) are listed at the end of this file once finished.

## research-v4.2

Two runs, same code: `research-v4.2` (qwen2.5:1.5b, 3 seeds × 2 trials × distractors {10, 250, 1000}, n = 90 per
condition) and `research-v4.2-3b` (qwen2.5:3b, 3 seeds × 1 trial × {10, 250}, n = 30 per condition).
Temperature 0, 32 tokens, `num_ctx` 32768, unique-key generator, `plain` instruction. No timeouts or truncation.

`research-v4.2` reuses 540 records (raw, transcript, bounded, nsa_history, nsa_state, append_memory) from
`research-v4.1`, because their prompts are unchanged and the episodes are identical (`manifest.json → carried_over`,
`carried_from` on each row). The temporal and NSA-memory conditions were re-run.

Lenient accuracy (strict in brackets where it differs by more than 0.03):

| Condition | 1.5b (n=90) | 3b (n=30) |
|-----------|-------------|-----------|
| raw | 0.00 | 0.00 |
| transcript | 0.91 | 0.93 |
| bounded | 0.01 | 0.03 |
| nsa_history (control) | 0.01 | 0.00 |
| nsa_state | 0.00 | 0.00 |
| append_memory | 1.00 | 1.00 |
| temporal_memory | 0.99 (0.97) | 0.90 |
| nsa_temporal_memory | 0.87 | 0.97 (0.80) |
| nsa_temporal_memory_nohist | 1.00 | 1.00 |

Memory retrieval was correct in 100 % of memory-condition episodes in both runs.

### What changed from 4.1, and what it shows

* **History queries fixed.** Removing `CURRENT_VALUE=` from superseded rows took `temporal_memory` on
  `supersession_history` from 0/18 to 18/18 on 1.5b (3b: 6/6). In a controlled check on the same prompts the
  variant with the field scored 0/18 on 1.5b and 18/18 on 3b; without it 18/18 on both.
* **Recent observations compete with retrieved memory.** `nsa_temporal_memory` (history + state + memory) vs
  `nsa_temporal_memory_nohist` (state + memory): 1.5b 0.87 vs 1.00, driven by `supersession_current`
  (0.39 vs 1.00; the model copies a distractor value from the window). Holm p = 0.006 for NSA-memory vs plain
  temporal memory on 1.5b. On 3b the difference is within noise. This is why `RuntimeConfig.include_history_with_memory`
  exists; its default is still `True`.
* **State text still adds nothing** (`nsa_state` vs `nsa_history`: 0 vs 1 discordant pairs, p = 1.0 on 1.5b; 0 vs 0 on 3b).
  Same caveat as before: the serialised state does not carry the facts.
* **Memory beats the transcript at long range on 1.5b.** Transcript accuracy falls with length (100 % at d=10,
  93 % at 250, 80 % at 1000; prompts up to ~31.6k tokens, ~103 s) while memory prompts are ~100 tokens and ~1 s.
  `append_memory` vs transcript: 8 vs 0 discordant, Holm p = 0.04. On 3b (d ≤ 250) the transcript is not
  distinguishable from memory.
* **Format bleed on 3b.** `temporal_memory` scores 3/6 on `supersession_current` because 3b appends the version number to
  the value (`VALUE=V017_CURRENT_2`). The retrieved value is right; the strict and lenient graders both mark it wrong
  (`_<digits>` is indistinguishable from a legitimate value suffix, so the grader is not loosened). That is a 3b
  output-format fault, not a retrieval fault.
* **Run-to-run variation exists at temperature 0.** One `temporal_memory` episode (seed 17, d=10) scored differently
  between the 4.1 and 4.2 runs on an identical prompt (`VALUE=V017_CURRENT_VERSION=2` vs `VALUE=2 V017_CURRENT`). Expect
  about ±1 episode of noise per cell.

### Limits

One or two small models, 30–90 episodes per condition, a keyed-lookup oracle for memory extraction, and no test of
LLM-driven memory formation or larger models. `append_memory` returns superseded values 20 % of the time (stale rate) yet
scores 100 % here; with a model less able to pick the later of two rows that would not hold.
