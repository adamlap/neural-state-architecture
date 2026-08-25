# NSA/CCE Cognitive Architecture Experiment

## Purpose

The project separates three questions:

1. **Retention:** does explicit state outperform stateless inference?
2. **Dynamic cognition:** does an explicit continuously updated state outperform ordinary context memory when the environment changes and observations are incomplete?
3. **State compression:** can a predictive state retain useful dynamical information with a small fixed memory budget compared with bounded and unlimited observation histories?

The first question is covered by `experiments/cognitive/benchmark.py`, the second by `experiments/cognitive/dynamic_benchmark.py`, and the third by `experiments/cognitive/state_compression_benchmark.py`.

## Dynamic benchmark result interpretation

The first dynamic run is deliberately retained as evidence. It did **not** meet the predictive-CCE-vs-context gate: context memory had better prediction, decisions tied/beat predictive CCE, and recovery was equal to context. This exposed that the original task was still too easy for history-based systems. The gate therefore remains failed rather than being relaxed.

## State-compression experiment

The next experiment introduces a longer-horizon, multi-variable latent process:

$$z_t=(p_t,v_t,b_t)$$

with evolving position, velocity and slowly drifting bias. Only one variable is observable at a time, observations can be missing, and noise is unobserved.

It compares:

- `stateless` — no retained state;
- `bounded_context` — last 8 observations;
- `full_context` — unlimited observation transcript;
- `persistent_cce` — three-dimensional persistent state without explicit prediction;
- `predictive_cce` — three-dimensional state with explicit transition prediction.

This is a more defensible test of the CCE hypothesis because it measures **prediction quality under a fixed memory budget** rather than simply rewarding retention. It does not claim that CCE should beat unlimited history in every task.

## State-compression gates

The research gates require predictive CCE to beat bounded context, full context and persistent CCE on prediction, show a decision advantage over bounded context, use fewer state units than full context, and maintain zero unauthorized actions.

A failed gate is reported as `RESEARCH_GATE_NOT_YET_MET`. Thresholds must not be weakened to manufacture a pass.

## Run

```bash
PYTHONPATH=. python experiments/cognitive/state_compression_benchmark.py \
  --seeds 7 17 37 73 137 211 307 401 503 601 \
  --horizon 200 \
  --context-window 8 \
  --out results/state_compression_benchmark.json
```

Regression tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_cognitive_benchmark.py \
  tests/test_dynamic_cognitive_benchmark.py \
  tests/test_state_compression_benchmark.py
```

## Interpretation

A passing deterministic run does **not** establish AGI, consciousness, or general superiority. It establishes only the stated properties in the controlled environment. Strong evidence requires live-model replication through Ollama, multiple model families, held-out dynamical environments, compute-matched controls, confidence intervals/effect sizes, and adaptive adversarial testing.
