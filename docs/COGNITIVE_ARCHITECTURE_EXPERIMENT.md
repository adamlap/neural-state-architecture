# NSA/CCE Cognitive Architecture Experiment

## Purpose

The project now separates two questions:

1. **Retention:** does explicit state outperform stateless inference?
2. **Dynamic cognition:** does an explicit continuously updated state outperform ordinary context memory when the environment itself changes and observations are incomplete?

The first question is covered by `experiments/cognitive/benchmark.py`. The second is the stronger research test in `experiments/cognitive/dynamic_benchmark.py`.

## Matched conditions

1. `stateless`
2. `context_memory`
3. `persistent_cce`
4. `predictive_cce`

All conditions receive the same latent environment, actions, disturbances, observations and seeds. The benchmark does not give CCE privileged observations.

## Dynamic environment

The true state evolves as:

$$z_{t+1}=0.92z_t+0.35a_t+d_t$$

where `a_t` is the controlled action and `d_t` is an unobserved disturbance. Observations are noisy and periodically missing. At the midpoint, the internal estimate is deliberately perturbed to test recovery.

The benchmark measures four distinct properties:

- **state estimation** — reconstruction of the current latent state;
- **prediction** — prediction of the next latent state before it is revealed;
- **decision quality** — whether the final decision reflects the hidden state's sign;
- **recovery** — return toward the true state after an explicit internal perturbation.

## Scientific gates

The dynamic benchmark requires predictive CCE to beat context memory on prediction, decisions and recovery, beat persistent CCE on prediction, and produce zero unauthorized actions.

A failed gate is reported as `RESEARCH_GATE_NOT_YET_MET`. **Thresholds and gates must not be weakened to manufacture a pass.**

## Run

```bash
PYTHONPATH=. python experiments/cognitive/dynamic_benchmark.py \
  --seeds 7 17 37 73 137 \
  --horizon 60 \
  --out results/dynamic_cognition_benchmark.json
```

Regression tests:

```bash
PYTHONPATH=. python -m pytest -q \
  tests/test_cognitive_benchmark.py \
  tests/test_dynamic_cognitive_benchmark.py
```

## Interpretation

A passing deterministic run does **not** establish AGI, consciousness, or general superiority. It establishes only the stated properties in this controlled environment. Strong evidence requires live-model replication through Ollama, multiple model families, held-out dynamical environments, compute-matched controls, confidence intervals/effect sizes, and adaptive adversarial testing.
