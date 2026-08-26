# NSA Research Package

This directory is the researcher-facing entry point for the Neural State Architecture (NSA) project.

## Start here

1. [`NSA_RESEARCH_BRIEF.md`](NSA_RESEARCH_BRIEF.md) — concise description of the architecture, hypothesis and current evidence.
2. [`NSA_6_4_LIVE_RESULTS.md`](NSA_6_4_LIVE_RESULTS.md) — analysis of the first live Ollama replication run.
3. [`CLAIMS_AND_EVIDENCE.md`](CLAIMS_AND_EVIDENCE.md) — claim-by-claim evidence boundary and what is not yet established.
4. [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — exact reproduction workflow and reporting requirements.
5. [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) — technical map for researchers and implementers.

## Canonical evidence

The machine-readable NSA 6.4 live run is preserved under:

```text
results/nsa64/ollama-quick/
```

The manifest records the model/backend, development and held-out seeds, experiment grid, compute-accounting policy, every run record, invariant/audit status and raw artifact locations.

The current live run used `qwen2.5:3b` through Ollama. It is a **quick replication**, not the full multi-model publication matrix. It should therefore be treated as evidence that the protocol and substrate execute against a real local model, not as the final generalization claim.

## Research status

The strongest current evidence supports three narrower statements:

- the NSA 6.3 full substrate achieved high governed task completion in the procedural blind-world benchmark while recording zero observed governance violations in the monitored arm;
- the NSA 6.4 protocol preserves this evaluation structure across development/held-out splits and adaptive adversarial stress;
- the first live Ollama replication produced successful task completion in easier cells and preserved zero observed violations, while performance degraded substantially as hypothesis complexity increased.

The third point is important: the live result is **not uniformly strong**. The difficulty degradation is itself a result and is retained rather than hidden.

## What remains before publication claims

The next empirical release should run multiple independent model families, the full predeclared difficulty/noise grid, substantially larger trial counts, fixed compute accounting across all arms, confidence intervals/effect sizes, and held-out adversarial evaluation. Only then should cross-model generalization or adoption claims be made.
