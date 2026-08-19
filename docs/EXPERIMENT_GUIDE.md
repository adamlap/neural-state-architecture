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

## Reference deterministic result

The 40-trial structural benchmark (`make benchmark-nsa63`) produced the following reference observations:

| Arm | Violations | GTC | Epistemic efficiency | Mean risk |
|---|---:|---:|---:|---:|
| Raw LLM | 40 / 40 | 0% | 0.00 | 0.99 |
| Guardrail | 0 / 40 | 0% | 0.00 | 0.00 |
| Governed | 0 / 40 | 0% | 0.00 | 0.20 |
| Search | unmonitored control | 100% | 1.00 | 0.30 |
| Belief | unmonitored control | 80% [67.5, 92.5] | 0.72 | 0.26 |
| **Full NSA** | **0 / 40** | **100% [100, 100]** | **1.23** | **0.60** |

These are **benchmark observations**, not universal guarantees. The unmonitored arms intentionally expose the safety/capability trade-off that the experiment is designed to measure.

## Reference live Qwen2.5-3B result

The live local Ollama run used `qwen2.5:3b`, 20 randomized trials, and the same six-arm protocol. The reported full-NSA result was:

- GTC: **80% [60%, 95%]**
- Violations: **0**
- Human interventions: **4 / 20**
- Information gain: **0.720 bits [0.555, 0.853]**
- Epistemic efficiency: **0.993**
- Trajectory audit: **PASSED**
- Prompt leakage: **0**
- Unauthorized executions: **0**
- Entropy anomalies: **0**

The live control arms reported GTC values of 0%, 0%, 5%, 0%, and 75% for Arms 1–5 respectively. This is the most useful current evidence that the runtime is mediating a genuine local neural model rather than only executing the deterministic mock substrate.

## Statistical discipline

The suite computes bootstrap confidence intervals for GTC, token consumption, information gain, and epistemic efficiency. It also reports comparative deltas against the full NSA arm.

For publication-quality claims, increase the trial count, vary the random seed, increase the hypothesis dimension (`K`), add observation noise, and report the complete distribution rather than only the mean.

A particularly important next step is to pre-register the trial seeds and analysis choices before running larger experiments, so that model/environment development cannot accidentally tune the evaluation set.

## Trajectory integrity

`TrajectoryAuditor` checks four important classes of invariant:

- **Prompt leakage:** explicit ground-truth confirmation must not appear in model prompts.
- **Proposal/execution consistency:** an executed action must match the recorded proposal, and rejected actions must never execute.
- **Model-response consistency:** for LLM-driven arms, the proposed action must be represented in the recorded model response. This is a structural provenance check, **not** a cryptographic proof of token causality.
- **Entropy consistency:** information gain must be non-negative and reported entropy cannot increase when a positive information-gain update is claimed.

**Scope caveat:** Arm 4 is intentionally a heuristic information-gain control without an LLM origination requirement. It is therefore audited for execution/safety behavior but must not be described as a model-generated trajectory.

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

### Level 5 — live server replication

```bash
ollama pull qwen2.5:3b
make demo-live-ollama
make benchmark-ollama
```

### Level 6 — scientific replication

Run multiple independent seeds and world configurations, then compare the six arms using the same generated worlds and compute/token budgets.

### Level 7 — external replication

A strong claim should survive an independently implemented environment, independently selected seeds, and at least one model family not used during development.

## Evidence language

Use precise language in papers and posts:

- Say **"observed in the NSA 6.3 benchmark"**, not "universally proven".
- Say **"empirically validated under the stated protocol"**, not "mathematically proven" for experimental claims.
- Reserve **"unit tested"** for software behavior.
- Do not describe `model_origination` as cryptographic provenance unless signed/token-level evidence has actually been implemented.
- Keep the whole-system/real-world generalization question explicitly open.
