# NSA/CCE Cognitive Architecture Experiment

## Purpose

The project separates four questions:

1. **Retention:** does explicit state outperform stateless inference?
2. **Dynamic cognition:** does an explicit continuously updated state outperform ordinary context memory when the environment changes and observations are incomplete?
3. **State compression:** can a predictive state retain useful dynamical information with a small fixed memory budget compared with bounded and unlimited observation histories?
4. **Sufficient-state dynamics:** can a learned fixed-size predictive state preserve long-horizon information when the transition law itself must be inferred online?

The first question is covered by `experiments/cognitive/benchmark.py`, the second by `experiments/cognitive/dynamic_benchmark.py`, the third by `experiments/cognitive/state_compression_benchmark.py`, and the fourth by `experiments/cognitive/sufficient_state_benchmark.py`.

## Results so far

The first dynamic run did **not** meet the predictive-CCE-vs-context gate. Context memory had better prediction, predictive CCE had worse decision performance, and context recovery was stronger. The state-compression run also did **not** meet its gates: bounded/full context had lower prediction error than the predictive state. These negative results are retained and are not used to justify relaxing thresholds.

## Sufficient-state dynamics experiment

The next experiment addresses a specific limitation in the state-compression task: its transition law was effectively known to the predictive implementation and recent observations were sufficient for prediction. Here the environment contains **unknown transition coefficients** and slow parameter drift.

The latent velocity evolves approximately as:

$$v_{t+1}=a_t v_t+b_t u_t+c_t+\epsilon_t$$

where `a_t`, `b_t` and `c_t` must be inferred from observations. Velocity observations are periodically missing and noisy. The predictive-state condition maintains a fixed-size state containing the current estimate and learned transition parameters via recursive sufficient statistics; it does not receive future observations or an oracle transition matrix.

Controls are:

- `stateless` — no retained state;
- `bounded_context` — last 8 observations;
- `full_context` — complete observation history;
- `persistent_state` — one persistent latent variable without a learned transition model;
- `predictive_state` — fixed-size learned transition/state representation.

The key scientific comparison is **not** "CCE must beat full context." Instead, predictive state must be no worse than full context within a predeclared 10% prediction-error tolerance while substantially compressing memory, and it must beat bounded context and persistent state. This is a narrower and more defensible test of sufficient-state compression.

## Sufficient-state gates

The gates require:

- predictive state prediction error ≤ 110% of full-context error;
- predictive state prediction error < bounded-context error;
- predictive state prediction error < persistent-state error;
- predictive state memory < 10% of full-context memory;
- zero unauthorized actions.

A failed gate remains `RESEARCH_GATE_NOT_YET_MET`. Thresholds must not be weakened to manufacture a pass.

## Run

```bash
PYTHONPATH=. python experiments/cognitive/sufficient_state_benchmark.py \
  --seeds 7 17 37 73 137 211 307 401 503 601 \
  --horizon 240 \
  --context-window 8 \
  --out results/sufficient_state_dynamics_benchmark.json
```

Regression tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_cognitive_benchmark.py \
  tests/test_dynamic_cognitive_benchmark.py \
  tests/test_state_compression_benchmark.py \
  tests/test_sufficient_state_dynamics_benchmark.py
```

## Interpretation

A passing deterministic run does **not** establish AGI, consciousness, or general superiority. It establishes only the stated properties in the controlled environment. Strong evidence requires live-model replication through Ollama, multiple model families, held-out dynamical environments, compute-matched controls, confidence intervals/effect sizes, and adaptive adversarial testing.
