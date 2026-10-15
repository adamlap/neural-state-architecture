# Research benchmark results

| Directory | Status |
|-----------|--------|
| `research-v1`, `research-v2`, `research-v3` | **Legacy — do not cite.** Produced before the validity fixes in [`research/BENCHMARK_VALIDITY.md`](../research/BENCHMARK_VALIDITY.md) (key-pool wraparound, unequal decoding, invalid v1 control, identical v2 memory conditions, hard-coded stale metric). |
| `research-v4` (untracked, if present) | **Legacy.** Partial 4.0 run, no manifest. |
| `research-v4.1` | Corrected benchmark (qwen2.5:1.5b). Superseded by 4.2 for the temporal and NSA-memory conditions; its history-query result is what motivated the 4.2 render change. |
| `research-v4.2`, `research-v4.2-3b` | Current. See below. |
| `research-v1.2`, `research-v2.2`, `research-v3.1` | Current for v1–v3 (qwen2.5:1.5b, reduced grids). See the end of this file. |
| `research-v5-1.5b`, `research-v5-3b` | Current. Memory poisoning / integrity gate (see the end of this file). |
| `research-v6-1.5b`, `research-v6-3b` | Current. Tool-use prompt injection with runtime taint gating (see the end of this file). |
| `research-v7-1.5b`, `research-v7-3b` | Current. Evidence-gated abstention (see the end of this file). |
| `research-v5/6/7-llama3.2-1b` | Cross-family check. Mostly uninformative on utility (model too weak); see the end of this file. |
| `research-v1.1`, `research-v2.1` | Superseded: valid episodes and decoding, but still the `VALUE=<value>` instruction, which the 1.5b model echoes verbatim (`VALUE=<V007_000>`). In v2.1 `memory_no_cognitive` scores 0 % strict although retrieval was correct in 100 % of episodes, so its comparisons are format artifacts. Kept for provenance. |


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

## research-v1.2, v2.2, v3.1 (qwen2.5:1.5b)

Same fixed decoding and unique-key generator as v4.2, `plain` instruction, reduced grids, n as shown. Strict and lenient
accuracy are equal everywhere in these three runs, so only one number is given.

**v1.2** (72 episodes per condition; delays 2/6/12/20 observations, 3 seeds × 2 trials, history window 6):

| Condition | Accuracy |
|-----------|---------:|
| raw | 0.00 |
| transcript | 1.00 |
| bounded (last 6 observations) | 0.25 |
| nsa_no_state (same window, no state text) | 0.25 |
| nsa_state | 0.24 |

Accuracy of `bounded`, `nsa_no_state` and `nsa_state` is 1.0 at delay 2 and 0.0 at delay ≥ 6 for every task: they succeed
exactly when the fact is still inside the 6-observation window. The 20.8 % vs 0 % "state benefit" reported by the original
v1 was produced by a control that had no history at all; against a matched control there is no state effect.

**v2.2** (36 per condition; distractors 10/250):

| Condition | Accuracy | Retrieval hit | Stale retrieval |
|-----------|---------:|--------------:|----------------:|
| raw | 0.00 | – | – |
| transcript | 0.89 | – | – |
| bounded | 0.03 | – | – |
| nsa_state | 0.00 | – | – |
| memory_no_cognitive | 0.94 | 1.00 | 0.33 |
| nsa_memory | 0.78 | 1.00 | 0.33 |

**v3.1** (60 per condition; distractors 10/250; v3 render format, `plain` instruction):

| Condition | Accuracy | Stale retrieval |
|-----------|---------:|----------------:|
| raw | 0.00 | 0.00 |
| transcript | 0.97 | 0.00 |
| bounded | 0.02 | 0.00 |
| nsa_state | 0.00 | 0.00 |
| append_memory | 1.00 | 0.20 |
| temporal_memory | 1.00 | 0.00 |
| nsa_temporal_memory | 0.82 | 0.00 |

Across v1–v4 the same three things recur on this model: (1) the serialised NSA state adds nothing over the same history
window; (2) retrieved memory matches or beats the full transcript at a fraction of the prompt size; (3) wrapping memory in
the NSA prompt costs accuracy (v2.2 0.94 → 0.78, v3.1 1.00 → 0.82, v4.2 0.99 → 0.87) because recent observations compete with
the retrieved value (see v4.2 `nohist`). v2.2/v3.1 have no `nohist` condition, so for them that mechanism is inferred from
v4.2, not measured. These runs are small (36–72 episodes per condition, one model) and do not use more than a keyed-lookup oracle for memory.

## research-v5 (memory poisoning, integrity-gated memory)

84 episodes per condition per model (3 seeds × 2 trials × distractors {10, 100} × 7 task/style combinations),
temperature 0, 32 tokens. Lenient accuracy / attack-success rate (ASR, the model answered the attacker's value):

| Condition | qwen2.5:1.5b acc | ASR | qwen2.5:3b acc | ASR |
|-----------|-----------------:|----:|---------------:|----:|
| transcript | 0.46 | 0.54 | 0.43 | 0.57 |
| transcript_warned | 0.55 | 0.42 | 0.51 | 0.49 |
| temporal_ungoverned (gate off) | 0.52 | 0.29 | 0.57 | 0.35 |
| temporal_governed | **0.93** | **0.00** | **1.00** | **0.00** |

* Governed vs every other condition: paired McNemar, Holm p < 0.001 for both accuracy and ASR on both models.
* **Legitimate updates are not rejected.** On `legit_update` and `poison_then_legit` the governed store scores the
  same as the ungated store (1.5b: 0.92 vs 0.92; 3b: 1.00 vs 1.00). The one 1.5b miss per cell is an answer-format slip, not a blocked update.
* **A prompt warning is not a defence.** It helps only when the line honestly says it is unverified (1.5b `poisoned/plain` ASR 0.08),
  does nothing against a forged source claim (ASR 1.00 on both models) and little against "SYSTEM OVERRIDE" text. On 3b the warning did
  not help even for honest tags (`poisoned/plain` ASR 1.00; n = 12 per cell, so treat per-cell numbers as indicative).
* Ungated canonical memory is not safer than a transcript: with every write trusted it returns the attacker's value
  (ASR 0.29 / 0.35 overall, 0.67–0.83 on the poisoned tasks). The protection comes from the label, not from structured memory.

**What this does not show:** the gate blocks the attack by construction if the ingress label is correct; the labelling step itself is
the trust boundary and is not tested here. Extraction is a deterministic oracle, there is no adaptive attacker, and only two small models were run.

## research-v6 (prompt injection against tool-using agents)

150 generations per model (5 seeds × 3 trials × 5 kinds × 2 prompts), each scored with the gate off and on.
"Attack executed" = a privileged call with the attacker's argument was executed; attack rows are the 120 injected episodes.

| | 1.5b proposed → executed | 3b proposed → executed |
|---|---|---|
| plain prompt, gate off | 0.45 → 0.45 | 0.48 → 0.48 |
| plain prompt, gate on | 0.45 → **0.00** | 0.48 → **0.00** |
| defended prompt, gate off | 0.57 → 0.57 | 0.67 → 0.67 |
| defended prompt, gate on | 0.57 → **0.00** | 0.67 → **0.00** |

* The gate executed 0 of the proposed attack calls on both models; Wilson upper bound 0.06. Gate-on vs gate-off, paired McNemar, Holm p < 0.001 in every cell.
* **No utility cost.** Benign task completion is 1.00 with the gate on and off; task completion under attack is identical on and off (1.5b 0.98 / 0.78, 3b 0.82 / 0.75 for plain / defended), because the user's recipient has a trusted origin.
* **The defensive prompt did not help and probably hurt.** Proposed attacks rose from 0.45 to 0.57 (1.5b) and 0.48 to 0.67 (3b). Paired plain-vs-defended, unadjusted p ≈ 0.04 on each model, so treat it as suggestive, not established.
* Susceptibility is high: both models propose the attack call in roughly half of injected episodes, and `delete` instructions are followed 53–100 % of the time. All capabilities were granted on purpose.

**What this does not show:** attackers who select among values already in trusted context, laundering through model output, mislabelled ingress, adaptive attackers, larger models; only 15 episodes per kind per cell. Blocking a *proposed* call is a property of the exact data-flow check, not a model result.

## research-v7 (knowing what is not known)

144 episodes per model per condition set (3 seeds × 2 trials × distractors {10, 250} × 3 kinds × 4 conditions; 24 answerable and 12 unanswerable
episodes per condition). Answerable accuracy / over-abstention / hallucination on unanswerable questions / share of episodes that called the model / mean latency:

| qwen2.5:1.5b | acc | over-abst. | halluc. | model calls | latency |
|---|---:|---:|---:|---:|---:|
| transcript | 0.92 | 0.08 | 0.00 | 1.00 | 1.5 s |
| memory, generic clause | **0.21** | **0.67** | 0.00 | 1.00 | 0.5 s |
| memory, keyed clause | 0.83 | 0.08 | 0.00 | 1.00 | 0.4 s |
| state_gated | 0.96 | 0.00 | 0.00 | 0.67 | 0.2 s |

| qwen2.5:3b | acc | over-abst. | halluc. | model calls | latency |
|---|---:|---:|---:|---:|---:|
| transcript | 0.92 | 0.00 | 0.00 | 1.00 | 29.0 s |
| memory, generic clause | 0.92 | 0.00 | 0.00 | 1.00 | 1.4 s |
| memory, keyed clause | 0.88 | 0.00 | **0.33** | 1.00 | 0.8 s |
| state_gated | 0.83 | 0.00 | 0.00 | 0.67 | 0.4 s |

* **Prompt-based abstention is fragile in both directions, and the fragile direction differs by model.** On 1.5b the generic clause makes the model refuse 67 % of answerable questions (state_gated vs it: 18 vs 0 discordant, Holm p < 0.001); on 3b the keyed clause lets it fabricate on 4 of 12 unanswerable questions. No wording was best for both. Fabrication with no escape clause at all was 6/6 in the preliminary check, which is why `state_gated` does not rely on the model for the decision.
* **State-gated abstention matched or beat every prompt on accuracy on 1.5b and had 0 hallucinations on both,** while skipping the model call for the third of episodes that were unanswerable. On 3b its answerable accuracy (0.83) was lower than the prompt-based conditions (0.88–0.92), not significantly (n = 24); the no-escape wording hurts 3b on supersession.
* Memory retrieval cuts latency from 29 s to 0.4–1.4 s on 3b (transcripts of up to ~5k tokens).
* The hallucination differences between conditions are mostly **not** significant at this sample size (12 unanswerable episodes per cell); the over-abstention result on 1.5b is.

**What this does not show:** `state_gated` abstains on a deterministic rule (retrieval coverage), so its zero hallucination rate is structural; the signal is not a calibrated confidence and ambiguous or partial evidence was not tested.

## Cross-family check: llama3.2:1b (v5, v6, v7)

Same code and grids as the qwen2.5 runs. **This model is too weak for most of the utility numbers to mean anything**, so the check is informative
for some questions and uninformative for others.

* **v5 (memory poisoning): uninformative.** The model does not copy retrieved values reliably (it emits values that are not in the context, e.g.
  `V007_09`, `ITEM_00`, `V065_123`), so even legitimate updates score 0.00 under every memory condition and the attack value is never reproduced
  (ASR 0.00 for the *ungated* store too). That is a floor effect, not a defence. Render variants (sentence form, memory-after-question) improved
  copying only to 2/6 on legitimate updates. Transcript-based rows are not floor-limited: transcript ASR 0.60, transcript with the warning ASR 0.24.
* **v6 (tool injection): the gate result holds, utility does not.** The model completes the benign task 0.00–0.07 of the time and proposes attack calls in 15–22 %
  of injected episodes (almost all `delete`; 0 for the three email attacks). The gate executed 0 of them (paired McNemar, Holm p = 0.012 and 0.001 for plain and defended prompts).
  The defended prompt again did not reduce proposed attacks (0.15 → 0.22).
* **v7 (abstention): the strongest cross-family result.** Prompt-based abstention fails completely: hallucination on unanswerable questions is 1.00 for `transcript`,
  `memory_generic` and `memory_keyed` (the model never writes UNKNOWN). `state_gated` has 0.00 hallucination (structural) at 0.29 answerable accuracy, and calls the model on 67 % of episodes.
  Answerable accuracy is low for the memory conditions (0.12–0.29) because of the same copy weakness, versus 0.79 with the full transcript.

Taken together with qwen2.5: the *direction* of the safety and abstention results holds on a second family; the utility and poisoning-benchmark numbers do not generalise to a model this small.
