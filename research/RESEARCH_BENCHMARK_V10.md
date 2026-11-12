# Research Benchmark v10 — Selective Storage With a Familiarity Gate

`NSARuntime(..., novelty_gate=NoveltyGate())` with `RuntimeConfig(novelty_write_threshold=t)` scores every observed line against a fixed-size
familiarity memory (`nsa/organic.py`, 256×256 floats) and passes only lines with novelty ≥ t to the fact extractor. The gate learns every line.
Without a threshold it only measures (`last_novelty`, `gate_stats`). Default behaviour is unchanged. Needs NumPy.

## Design
v9 conversations plus 20 reminders restating earlier facts (half with a one-character typo in the value), 4 genuine updates, 2 untrusted forwarded
updates and chatter (`train` and `heldout` phrasings, 10 seeds). Rule extractor, canonical memory measured after ingest (no answer model).

## Result (`results/research-v10-rules.json`)
| phrasing | threshold | all keys correct | corrupted by a typo | memory writes | lines to extractor |
|---|---|---|---|---|---|
| train | none | 0.82 | 0.17 | 40 | 56 |
| train | 0.60 | 0.81 | 0.17 | 34 | 42 |
| train | 0.75 | 0.82 | **0.03** | **23** | **30** |
| heldout | none | 0.31 | 0.12 | 19 | 56 |
| heldout | 0.75 | 0.24 | 0.04 | 8 | 28 |

* The gate halves the work (lines passed to the extractor 56 → 30, writes 40 → 23) and removes most typo corruption (0.17 → 0.03).
* **It does not improve overall correctness** (0.82 → 0.82; held-out 0.31 → 0.24), because the same lexical score also drops some genuine updates.
  Mean novelty by line type (train): new fact 0.86, genuine update 0.74, typo'd reminder 0.63, clean reminder 0.60. A typo'd and a clean reminder are
  indistinguishable to the gate, and a genuine update is only partly separable from a reminder: "same line, different value" is what both a typo and a
  real change look like at the character level.
* Telling them apart needs semantics (is the new value a plausible change?), not familiarity. A rule such as "treat a value within edit distance 1 as
  noise" would be wrong for legitimate close values (ports 5433 → 5434).

## Status
The integration is a **selective-storage / compute-saving knob with a measured accuracy cost on heldout phrasings**, not an accuracy improvement.
Not measured: LLM extraction calls saved (the hybrid extractor makes one call per chunk, so savings need whole chunks to be skipped) and end-to-end
answer accuracy with a model in the loop. Recommended next step: combine the familiarity score with the memory's own state (does the line restate the
*current* value for a known key?) to route a line to ignore / confirm / supersede, rather than thresholding familiarity alone.
