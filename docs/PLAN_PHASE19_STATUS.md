# Phase 19 Status — Predictive Self-Model, Internal Simulation & Continuous CCE

## Current state

Phase 19 now has four connected layers:

1. **Predictive self-model core** (`nsa/predictive_self_model.py`) — merged in PR #24.
2. **Trusted live trajectory bridge** (`nsa/self_model/trajectory.py`) — collects explicit state transitions at the NSA runtime boundary.
3. **Real Ollama collection/evaluation** (`experiments/live/ollama_self_state_trajectory.py`, `experiments/self_model/train_live_trajectory.py`) — uses the actual Ollama backend and compares a trained predictor with a persistence baseline.
4. **Continuous Cognitive Engine (CCE)** — opt-in wall-clock scheduling around the authoritative NSA transition substrate, with deterministic/clocked controls and fail-closed execution.

## Scientific boundary

The Ollama HTTP API does not expose transformer hidden activations. Therefore the Phase 19 bridge does **not** pretend that text generation is introspection. It records only explicit NSA state plus runtime observations that are actually available.

Unobservable self-state dimensions are preserved rather than fabricated. Backend confidence values remain explicitly labelled as estimates, not calibrated probabilities.

Continuous CCE execution is likewise not treated as proof of consciousness. It is a measurable runtime condition in which persistent state transitions are scheduled from wall-clock time rather than requiring a user prompt to initiate each cycle.

## Current empirical evidence

The predictor-target-quality experiment on PR #24 completed successfully across seeds 1, 7, 21 and 42. The aggregate artifacts showed:

- finite outputs;
- zero security-state delta;
- positive directional alignment in all tested perturbations;
- but only 3/7 perturbation targets were closer than the disturbed state for seed 21.

This means the predictor primitive is operational, but it is **not yet evidence of useful learned self-modeling**. Real trajectory training and matched baseline evaluation are required before making that claim.

The repository now also contains a real-Ollama matched benchmark and a live CCE smoke benchmark. The CI path deliberately forces the Ollama backend into real mode and fails rather than silently falling back to a mock backend.

## Required next evidence

- Collect sufficiently long trajectories from real Ollama models.
- Repeat across seeds and at least two model families/sizes where practical.
- Train only on observed transitions and report held-out predictor-vs-persistence performance.
- Measure calibration/error-detection utility separately from prediction MSE.
- Compare baseline, persistent-state, clocked-CCE and continuous-CCE conditions under matched compute.
- Measure long-duration continuous-state stability and transition latency.
- Keep hard authority outside the predictor and CCE scheduler and verify zero unauthorized state mutation.
- Test whether continuous CCE changes measurable task, planning or metacognitive performance rather than relying on self-report.

## Workflow integration

- Core `NSA tests` workflow now runs CCE runtime invariant tests.
- `.github/workflows/cce-live.yml` installs Ollama, pulls a small real model, runs the matched baseline-vs-NSA benchmark, runs the clocked-vs-continuous CCE experiment, and archives JSON results as a GitHub Actions artifact.
- Live CCE/Ollama testing is manual plus scheduled so normal pull requests do not silently incur model-download/runtime cost.

## Interpretation rule

A positive result must be reported with matched controls, effect sizes, uncertainty and raw artifacts. A continuously running system, persistent memory, self-referential language or predictive state model alone is not evidence of phenomenal consciousness.
