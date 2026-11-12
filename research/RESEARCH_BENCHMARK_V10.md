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
