# NSA Explicit Self-State Experiment

This is the first runnable experiment for the hypothesis that an explicit computational self-state can improve useful cognition.

## Models

### Baseline

$$m_{t+1}=F(m_t,x_t)$$

A recurrent evidence accumulator with a confidence head.

### NSA explicit-state model

$$
(m_{t+1},S_{t+1})=F(m_t,x_t,S_t)
$$

The model maintains a seven-dimensional state:

1. confidence
2. uncertainty
3. perceived risk
4. capability awareness
5. resource pressure
6. goal progress
7. state prediction error

The state is predicted from the internal representation and fed back into the next recurrent computation.

## Task

The model observes a sequence of noisy measurements of a hidden binary hypothesis. Consistent evidence should increase confidence; contradictory/noisy evidence should reduce useful confidence.

Training and evaluation use synthetic data so the experiment is deterministic, cheap and reproducible.

## Metrics

- accuracy
- Brier score
- calibration error (ECE)
- selective accuracy at confidence >= 0.7
- coverage
- mean confidence
- shifted-distribution performance

## Causal test

The explicit-state model supports:

```text
state_scale=1.0  -> normal NSA computation
state_scale=0.0  -> explicit state path ablated
```

The same weights and observations are used. A performance change is therefore evidence that the explicit state pathway is causally participating in the computation, although it is not by itself evidence for consciousness.

## Run

From the repository root:

```bash
PYTHONPATH=. python experiments/self_state/run.py --steps 800 --seed 7
```

For a quick smoke test:

```bash
PYTHONPATH=. python experiments/self_state/run.py --steps 50 --seed 7
```

## Interpretation

This is **not** a proof that NSA makes AI more intelligent. It is the first falsifiable instrument for testing that hypothesis.

A strong result requires repeated seeds, matched compute/parameter budgets, independent tasks and causal state interventions. A negative result is equally valuable because it tells us the explicit state substrate is not useful in this form.
