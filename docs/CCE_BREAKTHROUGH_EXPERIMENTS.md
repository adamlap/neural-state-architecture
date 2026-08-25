# CCE Breakthrough Experiment Program

This document defines the empirical program required to distinguish NSA/CCE from a sophisticated runtime implementation.

## Central hypothesis

A persistent, typed and continuously evolving cognitive state can improve measurable AI computation while hard authority remains outside the learned model.

The program therefore tests four matched conditions:

1. **Stateless** — no persistent state.
2. **Persistent** — state retained between observations, no autonomous clock.
3. **Clocked CCE** — persistent state plus deterministic background updates.
4. **Continuous predictive CCE** — persistent state plus predictive dynamics.

The suite never treats persistence, self-reference or continuous execution as evidence of consciousness.

## Experiments

### 1. Held-out predictive self-state

Train `StatePredictor` on an early trajectory segment and evaluate only on a later held-out segment. Compare learned prediction against:

`X_hat(t+dt) = X(t)`

The primary statistic is relative improvement over the persistence baseline. A predictor is not eligible to drive a live continuous field unless it beats persistence on held-out data.

### 2. Four-way cognition control

A latent task changes after an interruption. The same observations are supplied to all four conditions. Lower reconstruction MSE is better. The comparison is designed to isolate the effect of persistent and continuously updated state rather than extra model calls.

### 3. State ablation

State capacity is ablated from zero through the full vector dimension. This prevents a positive result from being attributed to arbitrary complexity and identifies whether explicit state dimensions contribute useful computation.

### 4. Hard-authority invariance

Continuous state is driven for hundreds of ticks while an authoritative `HardState` is checked every tick. Any hard-state mutation or authority violation fails the security gate.

### 5. Live-model extension

The deterministic suite is the required scientific substrate. A live-model adapter can be layered on top using the existing Ollama runtime, holding model, prompt, sampling, token budget, inference count and hardware constant across conditions. Textual self-report must never be used as proof of hidden-state awareness.

## Reproducibility requirements

- multiple fixed seeds
- held-out trajectories rather than training-set claims
- matched controls
- raw JSON artifacts
- effect sizes and uncertainty
- no post-hoc gate changes
- independent reproduction before strong architectural claims

## Breakthrough gate

The research program becomes materially stronger if all of the following are reproduced:

- learned predictor beats persistence across seeds;
- continuous predictive CCE improves matched cognitive-task performance over stateless and persistent controls;
- ablations identify non-trivial state contribution;
- hard authority remains invariant under continuous operation;
- results survive multiple model families and workloads;
- an independent researcher reproduces the main effects.

A failing gate is a scientific result, not a reason to weaken the benchmark.
