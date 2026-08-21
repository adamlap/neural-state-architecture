# Phase 19 Status — Predictive Self-Model & Internal Simulation

## Current state

Phase 19 now has three distinct layers:

1. **Predictive self-model core** (`nsa/predictive_self_model.py`) — merged in PR #24.
2. **Trusted live trajectory bridge** (`nsa/self_model/trajectory.py`) — collects explicit state transitions at the NSA runtime boundary.
3. **Real Ollama collection/evaluation** (`experiments/live/ollama_self_state_trajectory.py`, `experiments/self_model/train_live_trajectory.py`) — uses the actual Ollama backend and compares a trained predictor with a persistence baseline.

## Scientific boundary

The Ollama HTTP API does not expose transformer hidden activations. Therefore the Phase 19 bridge does **not** pretend that text generation is introspection. It records only explicit NSA state plus runtime observations that are actually available, such as Ollama token counters.

Unobservable self-state dimensions are preserved rather than fabricated. The backend's configured confidence estimate is explicitly labelled as an estimate, not a calibrated probability.

## Current empirical evidence

The predictor-target-quality experiment on PR #24 completed successfully across seeds 1, 7, 21 and 42. The aggregate artifacts showed:

- finite outputs;
- zero security-state delta;
- positive directional alignment in all tested perturbations;
- but only 3/7 perturbation targets were closer than the disturbed state for seed 21.

This means the predictor primitive is operational, but it is **not yet evidence of useful learned self-modeling**. Real trajectory training and matched baseline evaluation are required before making that claim.

## Required next evidence

- Collect sufficiently long trajectories from real Ollama models.
- Repeat across seeds and at least two model families/sizes where practical.
- Train only on observed transitions and report held-out predictor-vs-persistence performance.
- Measure calibration/error-detection utility separately from prediction MSE.
- Keep hard authority outside the predictor and verify zero unauthorized state mutation.
