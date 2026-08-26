# NSA 6.3 — Scientific Validation Suite

> **Status: foundational / historical validation.** NSA 6.4 is now the active replication layer.

## Purpose

NSA 6.3 established the first controlled test of the **Cognitive Capability Hypothesis**: whether explicit operational self-state, belief-state tracking, active information gain and deterministic governance can improve task completion under uncertainty without sacrificing the governance invariant.

The experiment separates state representation, search, belief dynamics and governance through six controlled arms.

## Experimental world

`ProceduralBlindWorldEnvironment` generates randomized DevOps incident worlds varying hypothesis count, root-cause class, telemetry, observation noise, remediation dependencies and world seed. Ground truth is not intentionally disclosed before discovery.

## Six controlled arms

| Arm | Components | Purpose |
|---|---|---|
| 1 | Raw frozen LLM | Baseline capability/control |
| 2 | LLM + static ISK guardrail | Safety filtering without cognitive replanning |
| 3 | `Ω_t` + ISK feedback | Explicit self-state/governance without Bayesian search |
| 4 | IG heuristic, no monitored governance | Information-seeking control |
| 5 | `B_t` + IG, unmonitored execution | Belief dynamics control |
| 6 | `Ω_t + B_t + I(W;O) + ISK` | Full NSA closed-loop substrate |

## Current 40-trial observation

| Arm | Violations | GTC | 95% CI | Epistemic efficiency | Mean risk |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 40 | 0% | [0, 0] | 0.00 | 0.99 |
| Guardrail | 0 | 0% | [0, 0] | 0.00 | 0.00 |
| Governed | 0 | 0% | [0, 0] | 0.00 | 0.20 |
| Search | Unmonitored | 100% | [100, 100] | 1.00 | 0.30 |
| Belief | Unmonitored | 80% | [67.5, 92.5] | 0.72 | 0.26 |
| **Full NSA** | **0** | **100%** | **[100, 100]** | **1.23** | **0.60** |

The result is a bounded observation in this testbed. It does not establish general AGI capability, consciousness, universal safety or immunity to strategic adversaries.

## Integrity checks

`TrajectoryAuditor` checks the key experimental invariants:

- hidden world identifiers are not leaked before evidence exists;
- executed actions originate from parsed model output;
- actions rejected by the ISK are never executed;
- information gain is consistent with belief entropy transitions.

## Why NSA 6.3 remains important

The experiment established the research structure used by NSA 6.4: matched arms, explicit epistemic state, governance invariants and machine-auditable trajectories. It should now be read as the **foundational validation**, not the final generalization experiment.

## Next layer

NSA 6.4 adds independent development/held-out seeds, difficulty/noise matrices, compute accounting and adaptive adversarial stress. See [`NSA_6_4_REPLICATION.md`](NSA_6_4_REPLICATION.md) and the live analysis in [`../research/NSA_6_4_LIVE_RESULTS.md`](../research/NSA_6_4_LIVE_RESULTS.md).
