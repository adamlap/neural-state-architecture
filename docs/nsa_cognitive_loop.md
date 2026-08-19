# NSA Cognitive State Loop

This document defines the first end-to-end research loop for the framework.

$$
X_t \rightarrow S_t \rightarrow M_t \rightarrow A_t \rightarrow X_{t+1} \rightarrow \hat S_{t+1} \rightarrow S_{t+1}
$$

Where:

- $X_t$: observed input/world state;
- $S_t$: canonical + self-state representation;
- $M_t$: semantic/model computation;
- $A_t$: selected action;
- $X_{t+1}$: resulting observation;
- $\hat S_{t+1}$: predicted next self-state;
- $S_{t+1}$: observed next self-state.

## Metacognitive error

Define:

$$
E_t=d(\hat S_{t+1},S_{t+1})
$$

where $d$ is a bounded state distance.

The hypothesis is that $E_t$ can become a useful control signal for reasoning, verification and action selection.

## Important distinction

The loop does **not** imply consciousness. It establishes a measurable computational self-model:

$$
\text{system predicts its own state}\\
\text{system observes its actual state}\\
\text{system uses the discrepancy}
$$

Whether richer forms of self-modeling have consequences for intelligence is an empirical question.

## Safety invariant

Self-state and self-prediction remain advisory signals:

$$
S^{self}\not\Rightarrow Authority
$$

Any privileged action must still pass the state-flow graph and capability authority.

## Research progression

1. hand-defined state observations;
2. learned state estimators;
3. learned self-state predictors;
4. causal state interventions;
5. metacognitive routing/verification;
6. predictive self-models;
7. long-horizon self-model evaluation.

Each stage should be evaluated independently before advancing.
