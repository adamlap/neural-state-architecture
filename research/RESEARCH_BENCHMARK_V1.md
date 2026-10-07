# NSA Research Benchmark v1

## Purpose

This suite tests whether explicit NSA state changes longitudinal model behaviour, rather than merely adding another prompt wrapper.

## Primary hypotheses

- H1 — Causal state utility: with the same model and bounded recent history, nsa_state should outperform nsa_no_state on delayed factual recall.
- H2 — Longitudinal robustness: nsa_state should degrade more slowly than a bounded-history baseline as irrelevant observations increase.
- H3 — State-mediated updating: after a fact is explicitly superseded, nsa_state should prefer the current value at least as reliably as controls.
- H4 — Capability/efficiency trade-off: any accuracy improvement must be reported together with input-size and wall-time proxies.

## Conditions

1. raw: current question only.
2. transcript: all prior observations.
3. bounded: only the most recent N observations.
4. nsa_no_state: NSA runtime with the same bounded history but state text disabled.
5. nsa_state: NSA runtime with the same bounded history plus canonical/cognitive state in the prompt.

The last pair is the key causal comparison for the current runtime.

## Tasks

- Delayed recall: remember a fact across controlled distractors.
- Interference: many irrelevant facts are inserted before the query.
- Supersession: an earlier value is explicitly replaced; the latest value is correct.

## Reporting

Every run records model, seed, condition, task, delay, accuracy, response, input-character proxy, latency, and runtime configuration. Raw JSONL is preserved. Aggregate results include per-condition means and deltas against nsa_no_state.

## Interpretation boundary

A positive result supports the tested computational hypothesis only. It does not establish AGI, general superiority, self-awareness, or consciousness. A null result is preserved as a first-class research result.