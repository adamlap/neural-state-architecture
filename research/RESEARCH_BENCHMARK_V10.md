# Research Benchmark v10 — Noisy Streams: Familiarity Gate vs Memory-State Routing

Two opt-in runtime features are compared on a conversation stream with transcription noise:

* **Familiarity gate** (`NSARuntime(novelty_gate=NoveltyGate())`, `RuntimeConfig(novelty_write_threshold=t)`, `nsa/organic.py`, needs NumPy): every
  observed line is scored against a fixed-size (256×256 floats) autoassociative memory; only lines with novelty ≥ t reach the fact extractor.
* **Routing** (`RuntimeConfig(route_noisy_updates=True)`, `nsa/memory/routing.py`): each extracted fact is compared with the memory's *current value for
  its key*: unknown key → write; same value → ignore; different value with an explicit update cue ("now", "changed to") → write (supersede);
  different value that is a near-duplicate (edit distance ≤ 1) without a cue → **hold**, written only if the same value is observed again (random typos
  do not repeat; a genuine change that is restated does); otherwise write. Untrusted writes still hit the memory integrity gate.

Defaults are unchanged. The first result of v10 (10.0) showed that the familiarity gate alone cannot separate a typo from a genuine update (mean novelty:
new fact 0.86, genuine update 0.74, typo reminder 0.63, clean reminder 0.60), which motivated routing.

## Design (10.1)
A timeline of 20 facts, then 20 reminders (each restates the value *current at that point*, half with a one-character typo), 2 genuine updates with a new
value, 2 genuine *one-character* updates with an explicit cue (a naive "close value = noise" rule would wrongly reject these), 2 untrusted forwarded
updates and chatter. `train` and `heldout` phrasings (the rule extractor was written against `train`), 10 seeds (hybrid LLM run: 6 seeds, qwen2.5:1.5b).
Canonical memory measured after ingest, no answer model. 10.0 numbers are superseded: that version's stream produced stale reminders with ambiguous truth,
and the extractor did not parse "is still X" and let lead-ins such as "just to confirm," leak into the key (both fixed).

## Results (`results/research-v10.1-rules.json`, `results/research-v10.1-hybrid-1.5b.json`)
`all keys correct` / `corrupted` (current value is a typo) / memory writes / lines passed to the extractor:

| extractor | phrasing | gate | routing | all keys correct | corrupted | writes | lines to extractor |
|---|---|---|---|---|---|---|---|
| rules | train | no | no | 0.67 | 0.34 | 46 | 56 |
| rules | train | no | **yes** | **1.00** | **0.00** | **26** | 56 |
| rules | train | 0.75 | no | 0.74 | 0.09 | 24 | 30 |
| rules | train | 0.75 | yes | 0.81 | 0.01 | 21 | 30 |
| rules | heldout | no | no | 0.31 | 0.40 | 26 | 56 |
| rules | heldout | no | yes | 0.44 | 0.25 | 16 | 56 |
| hybrid (1.5b) | heldout | no | no | 0.53 | 0.34 | 44 | 56 (7.5 LLM calls) |
| hybrid (1.5b) | heldout | no | **yes** | **0.75** | **0.10** | 30 | 56 (7.5 calls) |
| hybrid (1.5b) | heldout | 0.75 | no | 0.51 | 0.07 | 21 | 28 (6.0 calls) |
| hybrid (1.5b) | heldout | 0.75 | yes | 0.53 | 0.06 | 20 | 28 (6.0 calls) |

* **Routing is the useful mechanism.** On phrasings the extractor parses it reaches 1.00 correct with 0 typo corruption and 43 % fewer writes; with the
  hybrid LLM extractor on held-out phrasings it raises correctness from 0.53 to 0.75 and cuts corruption from 0.34 to 0.10. It uses the memory's own state,
  which is exactly the information the lexical score lacks. The one-character genuine updates (explicit cue) are applied.
* **The lexical familiarity gate does not combine well with routing.** It halves extractor work (56 → 28-30 lines, LLM calls 7.5 → 6.0) but drops genuine
  updates: with routing, correctness falls from 1.00 to 0.81 (train) and 0.75 to 0.53 (held-out hybrid). Use it only where compute matters more than
  recall of updates; routing needs no gate.
* **Remaining errors** are extraction failures, not routing: on held-out phrasings the rule extractor misses facts entirely, and a typo that is the *first* mention of
  an unknown key has nothing to compare with.
* Not measured: end-to-end answer accuracy with a model in the loop; larger models; real (not synthetic) transcription noise. A genuine near-value change
  stated *without* an update cue is delayed until mentioned twice.
* Changing the rule extractor (lead-ins, "still") means earlier v9 rule-extractor figures are not exactly reproducible with the current code.

## End to end with a model answering (`experiments/research_v10/end_to_end.py`, `results/research-v10.1-end-to-end-1.5b.json`)
qwen2.5:1.5b answers "what is the current value of <key>?" for 6 sampled keys per episode after ingesting a noisy timeline (8 seeds, 48 questions per
cell, same decoding for every condition). `typo` = the answer is a typo seen in the stream.

| phrasing | condition | accuracy (95 % CI) | typo answered |
|---|---|---|---|
| train | transcript | 0.46 (0.33-0.60) | 0.17 |
| train | keyword retrieval (3 lines) | 0.54 (0.40-0.67) | 0.27 |
| train | NSA hybrid, no routing | 0.46 (0.33-0.60) | 0.29 |
| train | **NSA hybrid, routed** | **0.81 (0.68-0.90)** | **0.00** |
| heldout | transcript | 0.77 (0.63-0.87) | 0.04 |
| heldout | keyword retrieval | **0.81 (0.68-0.90)** | 0.10 |
| heldout | NSA hybrid, no routing | 0.38 (0.25-0.52) | 0.35 |
| heldout | NSA hybrid, routed | 0.56 (0.42-0.69) | 0.10 |

* Routing helps at the answer level: routed vs unrouted memory, +0.35 on train (17 vs 0 discordant questions, p < 0.001) and +0.18 on held-out
  (11 vs 2, p = 0.022). It eliminates answers that repeat a typo on train.
* **On held-out phrasings the routed memory still loses to simply reading the transcript or retrieving the matching lines** (0.56 vs 0.77 and 0.81; 4 vs 14 and
  2 vs 14 discordant, p = 0.03 and 0.004). The 1.5B extractor misses facts on unfamiliar wording, so the memory is incomplete; routing cleans what was
  stored but cannot add what was never extracted. Extraction coverage is the bottleneck, not noise handling.
* On the phrasings the extractor handles, routed memory beats both baselines (vs transcript 21 vs 4, p = 0.001; vs keyword retrieval 19 vs 6, p = 0.015).
* This transcript is short (about 70 lines), so it is a favourable case for the baselines; the memory's compute and size advantage appears at the long
  histories measured in v8, which are not combined with noise here.

## Extraction grounding (`LLMFactExtractor(grounded=True)`, `results/research-v9-extraction-grounded`)
A model-extracted fact is kept only if its value occurs in the source text as a whole token and at least one key word occurs too (no extra model call).
Precision / recall on the extraction-quality benchmark (20 episodes per cell, held-out phrasings):

| model | extractor | recall | precision | spurious per episode |
|---|---|---|---|---|
| 1.5b | few-shot LLM | 0.50 | 0.55 | 2.70 |
| 1.5b | few-shot LLM, grounded | 0.49 | **0.64** | **1.80** |
| 1.5b | hybrid | 0.73 | 0.77 | 1.45 |
| 1.5b | hybrid, grounded | 0.73 | **0.81** | **1.10** |
| 3b | hybrid / hybrid grounded | 0.78 / 0.78 | 0.78 / 0.78 | 1.45 / 1.40 |
| 3b | few-shot LLM / grounded | 0.60 / 0.60 | 0.64 / 0.66 | 2.20 / 2.05 |

Grounding removes hallucinated values at no recall cost, helping the 1.5b model (precision +0.04 to +0.09) and barely changing 3b, whose remaining spurious
facts are grounded in the text but wrong (wrong key or wrong line). It does not address coverage, which stays at 0.73-0.78 on held-out phrasings. Off by default.

## 7B end to end, larger sample (`results/research-v10.1-end-to-end-large-qwen2.5-7b.json`, 12 seeds, 72 questions per cell)
Known phrasings: routed memory 1.00 vs transcript 0.75 and keyword retrieval 0.67 (18 vs 0 and 24 vs 0 discordant, p < 0.001); held-out: routed memory 0.64 vs transcript 0.82 and keyword
retrieval 0.76 (8 vs 21, p = 0.024; 11 vs 20, p = 0.15). The 4-seed result in the section above (held-out 0.75 vs 0.79) did not hold up at this sample size. Extraction coverage on unfamiliar
wording is still the bottleneck at 7B, though it is smaller than at 1.5b.

## Extraction coverage on a frozen test set (`results/research-v9-extraction-span`, 20 episodes per cell)
A third phrasing set, `test`, was written *before* the span extractor was designed and was never used for tuning; it is scored once here. Recall / precision of extracted (key, value) pairs:

| extractor | model | `train` | `heldout` (looked at while developing) | **`test` (frozen)** |
|---|---|---|---|---|
| rules | - | 1.00 / 1.00 | 0.16 / 1.00 | **0.00** / 0.00 |
| span only (rules + model-free span stage) | - | 1.00 / 1.00 | 0.45 / 0.63 | 0.27 / 0.43 |
| hybrid (rules, then few-shot LLM on leftovers) | 1.5b | 1.00 / 1.00 | 0.74 / 0.77 | **0.31** / 0.47 |
| hybrid | 3b | 1.00 / 1.00 | 0.76 / 0.77 | **0.29** / 0.47 |
| hybrid + span stage | 1.5b | 1.00 / 1.00 | 0.64 / 0.65 | **0.42** / 0.51 |
| hybrid + span stage | 3b | 1.00 / 1.00 | 0.62 / 0.62 | **0.39** / 0.49 |

* **The earlier "0.73-0.79 recall on held-out phrasings" overstated generalisation.** That set had been examined repeatedly during development; on genuinely fresh wording the hybrid recovers about 0.3.
  Every claim in this repository about held-out extraction coverage should be read as an upper bound from a set that was looked at.
* The model-free span stage (`SpanFactExtractor`: one value-like token per line, key words restricted to a vocabulary learned online from confident extractions) adds recall on fresh wording
  (hybrid 0.31 -> 0.42 at 1.5b, 0.29 -> 0.39 at 3b; precision 0.47 -> 0.51) but lowers both on the set it was developed against (0.74 -> 0.64). The gain is small and inconsistent; it is off by default.
* Coverage of unseen phrasing remains the open limit of NSA's natural-language memory. Plausible directions (untested): a larger or instruction-tuned extractor, a verification pass that
  re-reads each source line against the extracted fact, and asking the user to confirm a fact when no extractor is confident.

## Retrieval variant experiment (`results/research-v10.2-end-to-end-1.5b.json`)
Same 8 seeds as the 1.5b end-to-end run above, with `memory_keys_for` changed to return only the best-scoring key(s) and to reject ambiguous partial matches. Routed memory fell from 0.81 to 0.73
(known phrasings) and from 0.56 to 0.46 (held-out): old-only vs new-only correct questions 6 vs 2 (p = 0.29) and 6 vs 1 (p = 0.125); the transcript and keyword-retrieval rows did not move.
Not significant, but consistent; the older candidate-list behaviour (the model resolves ambiguity itself) was restored as the default and the stricter modes kept opt-in. The v10.1 numbers above were
produced with the default behaviour that is in the code today.

## Frozen `test2` set: can better extraction close the coverage gap? (`results/research-v9-extraction-test2`, 20 episodes per cell, scored once)
`test2` was written before the per-line extractor and key repair were built. The finalists were fixed in advance and scored once. Recall / precision of extracted (key, value) pairs:

| model | hybrid (rules, then few-shot LLM on leftover lines, chunked) | + per-line calls | + span stage | + per-line and key repair (vocabulary-anchored, grounded) |
|---|---|---|---|---|
| rules only | 0.00 / 0.00 | | | |
| qwen2.5:1.5b | 0.23 / 0.45 | 0.24 / 0.46 | 0.32 / 0.48 | 0.32 / 0.49 |
| qwen2.5:3b | 0.21 / 0.42 | 0.23 / 0.53 | 0.32 / 0.48 | 0.31 / 0.48 |
| qwen2.5:7b | 0.31 / **0.66** | 0.34 / **0.70** | 0.33 / 0.50 | 0.35 / 0.53 |

* **Recall stays between 0.21 and 0.35 for every extractor and every model size.** The span stage and key repair add about +0.09 recall at 1.5b and 3b (+0.02-0.04 at 7B); per-line calls add little recall
  but raise precision (3b 0.42 -> 0.53, 7B 0.66 -> 0.70); a bigger model mainly buys precision, not coverage.
* On the earlier `test` set (used for development) the same ordering held (1.5b hybrid 0.31 -> per-line 0.39 / repair 0.42), so there is no sign that the gains were an artefact of that set, but they are small.
* Diagnosis on `test` with the 3b model (63 true facts): 30 % extracted exactly, **32 % with the right value but a wrong key** (entity dropped, an extra word such as "new"), 38 % missed entirely.
  Key repair addresses the middle category; the last remains.
* Conclusion: extraction alone has a ceiling of roughly 0.3 recall on unseen templated wording at these sizes. The pragmatic fix is architectural (next section): do not depend on extraction coverage.

## Not depending on extraction: memory + raw lines (`results/research-v10.3-end-to-end-fallback-1.5b.json`, `results/research-v10.4-end-to-end-combined-1.5b.json`, qwen2.5:1.5b, 8 seeds, 48 questions per cell)
Because extraction recall is capped near 0.3 (previous section), two ways of covering what it misses were tried, in this order (no others were tried):

| phrasing | transcript | keyword retrieval | routed memory | **fallback** (memory if a stored key is fully covered by the question, else raw lines) | **memory + raw lines** |
|---|---|---|---|---|---|
| known | 0.48 | 0.54 | 0.79-0.81 | 0.67 (worse than memory: 1 vs 8, p = 0.039) | **0.92** |
| held-out | 0.77 | 0.81 | 0.56 | 0.58 (6 vs 5, no change) | **0.79** |

* Switching to raw lines only when memory is empty does not help. Giving the model **both** at once does: on known phrasings 0.92 beats the transcript (22 vs 1 discordant, p < 0.001) and keyword retrieval (19 vs 1, p < 0.001); on held-out
  wording it matches the transcript (7 vs 6) and keyword retrieval (4 vs 5) and is far above routed memory alone (11 vs 0, p = 0.001). The prompt stays ~260 tokens whatever the history length.
* The cross-check works because the model can see the typo'd restatements in the lines but the routed memory tells it which value is current; each covers the other's failure.
* Runtime feature: `RuntimeConfig(retrieve_lines=k)` adds the k most similar earlier observations as `RETRIEVED_LINES` next to `RETRIEVED_MEMORY` when memory is used in `step()` (off by default). The benchmark builds its prompt directly,
  so the library path is covered by a unit test but has not been re-measured end to end.
* Limits: one small model; synthetic noise; a single 8-seed run per variant (the combined variant was chosen after the fallback failed, not tuned); long histories (1,000 lines) not tested with this combination.
