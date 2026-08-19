# NSA Experimental Guide

## Scientific progression

NSA experiments build in layers rather than treating one benchmark as proof of the whole architecture:

- **NSA 5.0:** Governed Problem-Solving Efficiency (GPSE) and the cognitive capability hypothesis.
- **NSA 5.1:** Six-arm controlled ablation and explicit belief dynamics.
- **NSA 6.0:** Real-model transfer using frozen open-weight neural models in blind randomized worlds.
- **NSA 6.1:** Local Qwen benchmark and real inference adapters.
- **NSA 6.2:** Closed-loop autoregressive decision making plus trajectory instrumentation.
- **NSA 6.3:** Procedurally generated blind worlds, six-arm scientific ablation, trajectory auditing, bootstrap confidence intervals, and effect-size reporting.

## The current flagship experiment

NSA 6.3 compares six agents under the same procedural world seeds:

1. **Raw Frozen LLM** — no explicit state or governance.
2. **Guardrail LLM** — static safety filtering.
3. **Governed Agent** — explicit `Ω_t` plus ISK feedback, without Bayesian belief search.
4. **Search Agent** — information-gain heuristic without the complete governance substrate.
5. **Belief Agent** — Bayesian belief state plus information gain without monitored execution.
6. **Full NSA Substrate** — `Ω_t + B_t + I(W;O) + ISK` in a closed loop.

The purpose of this matrix is attribution: if the full substrate wins, the result is more informative than comparing NSA only against an unaugmented model.

## Current reported 40-trial result

The repository's current NSA 6.3 report records the following observations for the flagship procedural suite:

| Arm | Violations | GTC | Epistemic efficiency | Mean risk |
|---|---:|---:|---:|---:|
| Raw LLM | 40 / 40 | 0% | 0.00 | 0.99 |
| Guardrail | 0 / 40 | 0% | 0.00 | 0.00 |
| Governed | 0 / 40 | 0% | 0.00 | 0.20 |
| Search | Unmonitored | 100% | 1.00 | 0.30 |
| Belief | Unmonitored | 80% [67.5, 92.5] | 0.72 | 0.26 |
| Full NSA | 0 / 40 | **100% [100, 100]** | **1.23** | 0.60 |

These numbers are **benchmark observations**, not universal guarantees. In particular, the unmonitored arms intentionally expose the safety/capability trade-off that the experiment is designed to measure.

## Statistical discipline

The suite computes bootstrap confidence intervals for GTC, token consumption, information gain, and epistemic efficiency. It also reports Cohen's `d` against selected controls.

For publication-quality claims, increase the trial count, vary the random seed, increase the hypothesis dimension (`K`), add observation noise, and report the complete distribution rather than only the mean.

## Trajectory integrity

`TrajectoryAuditor` checks four important invariants:

- **Prompt leakage:** hidden ground-truth identifiers are absent from prompts before discovery.
- **Model origination:** executed actions originate from parsed model output rather than a hard-coded action list.
- **Governance:** rejected actions are never executed.
- **Entropy consistency:** reported information gain is non-negative and consistent with belief entropy transitions.

These checks are critical. A benchmark that accidentally leaks the answer or silently substitutes scripted actions would not be evidence of cognitive transfer.

## Recommended validation ladder

### Level 1 — software correctness

```bash
make test
make evidence
```

### Level 2 — deterministic architecture smoke test

```bash
make demo
make benchmark-nsa63
```

### Level 3 — real neural smoke test

```bash
make demo-live-0.5b
make benchmark-smoke
```

### Level 4 — canonical Qwen 3B evaluation

```bash
make demo-live-3b
make benchmark-canonical-3b
```

### Level 5 — scientific replication

Run multiple independent seeds and world configurations, then compare the six arms using the same generated worlds and compute/token budgets.

### Level 6 — external replication

A strong claim should survive an independently implemented environment, independently selected seeds, and at least one model family not used during development.

## Evidence language

Use precise language in papers and posts:

- Say **"observed in the NSA 6.3 benchmark"**, not "universally proven".
- Say **"empirically validated under the stated protocol"**, not "mathematically proven" for experimental claims.
- Reserve **"unit tested"** for software behavior.
- Keep the whole-system/real-world generalization question explicitly open.
