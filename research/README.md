# NSA Research Package

This directory is the researcher-facing entry point for the Neural State Architecture (NSA) project.

## Start here

1. `NSA_RESEARCH_BRIEF.md` — architecture, hypothesis and current evidence.
2. `NSA_6_4_LIVE_RESULTS.md` — first live Ollama replication analysis.
3. `CLAIMS_AND_EVIDENCE.md` — claim-by-claim evidence boundary.
4. `REPRODUCIBILITY.md` — reproduction workflow and reporting requirements.
5. `ARCHITECTURE_OVERVIEW.md` — technical map for researchers and implementers.

## Canonical evidence

The machine-readable NSA 6.4 live run is preserved under `results/nsa64/ollama-quick/`. The manifest records model/backend, development and held-out seeds, experiment grid, compute-accounting policy, run records, invariant/audit status and raw artifact locations.

The first live run used `qwen2.5:3b` through Ollama. It is a **quick replication**, not the full multi-model publication matrix. It demonstrates real-model execution of the protocol; it does not establish final generalization or superiority claims.

## Research status

Current evidence supports narrower statements: the NSA substrate can execute explicit state, epistemic and governance loops around a real local model; the protocol preserves held-out/adversarial evaluation structure; and the first live run shows strong performance in easier cells with zero observed monitored violations while performance degrades as hypothesis complexity increases.

Negative results are retained deliberately.

## Before publication claims

Run multiple independent model families, the full predeclared difficulty/noise grid, larger trial counts, matched compute accounting, confidence intervals/effect sizes, held-out adversarial evaluation, and raw trajectory preservation.