# PLAN Addendum — CCE, Continuous Cognition & Live Ollama Evaluation

**Status:** active implementation track on `main`  
**Scope:** CCE / continuous execution, live Ollama inference, predictive self-state evaluation, and reproducible workflow testing.

This addendum is intentionally separate from the NSA core roadmap so parallel agents can continue modifying the core without coupling their work to the CCE implementation.

## Current implementation status

### Phase 18 — Self-State & Metacognition

The implementation has advanced beyond the original roadmap wording:

- [x] Persistent canonical self-state exists in the typed runtime.
- [x] State introspection is exposed through the trusted runtime.
- [x] Self-state is committed by trusted runtime code rather than model text.
- [x] Live Ollama inference is available through the runtime boundary.
- [x] Observable self-state trajectories are collected from real Ollama runs.
- [x] Multi-seed evaluation infrastructure exists.
- [ ] Demonstrate statistically robust gains over matched persistence baselines.
- [ ] Extend evaluation to calibration, error detection and planning rather than textual self-report.

### Phase 19 — Predictive Self-Model & Internal Simulation

- [x] Predictive self-model implementation.
- [x] Live Ollama trajectory integration.
- [x] Matched predictor-vs-persistence evaluation infrastructure.
- [x] Multi-seed aggregation infrastructure.
- [ ] Establish reproducible cross-seed advantage across models/tasks.
- [ ] Add counterfactual action consequence prediction.
- [ ] Add capability/resource prediction.
- [ ] Calibrate prediction uncertainty against observed state transitions.

## CCE — Continuous Cognitive Engine

CCE is now an explicit runtime research track bridging persistent cognitive state and the authoritative NSA substrate.

### Architecture

```text
REAL / LIVE INPUT
       |
       v
+----------------------+
| CCE                  |
| persistent runtime  |
| continuous scheduler|
+----------+-----------+
           |
           v
  authoritative NSA transition
           |
           v
      canonical state
           |
           +------> Ollama inference
           |              |
           |              v
           |       observable proposal
           |              |
           +--------------+
```

### CCE runtime status

- [x] Opt-in wall-clock continuous execution.
- [x] Explicit disabled/clocked control condition.
- [x] Deterministic/manual stepping for reproducible tests.
- [x] Lifecycle controls (`start`, `stop`, enable/disable).
- [x] Runtime observability (`CCEStatus`).
- [x] Non-overlapping transition execution.
- [x] Fail-closed behavior on authoritative transition errors.
- [x] State freeze on failure.
- [x] CCE restricted to scheduling; it cannot grant capabilities or mutate hard authority state.
- [x] CCE integrated with the authoritative NSA substrate boundary.
- [x] Real Ollama inference path.
- [x] Live Ollama matched baseline-vs-NSA benchmark.
- [x] Live Ollama clocked-vs-continuous CCE smoke benchmark.
- [x] CI artifact capture for live CCE/Ollama results.
- [ ] Continuous latent-state dynamics at sub-inference timescales using a learned state transition model.
- [ ] Real asynchronous speech/vision sensor deployment against the canonical runtime.
- [ ] Long-duration stability experiments.

## Experimental controls

Every continuous-cognition claim must retain a matched control:

1. **Baseline:** live Ollama without persistent NSA state.
2. **Persistent:** live Ollama with canonical persistent NSA state but no autonomous scheduler.
3. **Clocked CCE:** CCE with explicit deterministic/finite stepping.
4. **Continuous CCE:** CCE enabled on the wall-clock runtime loop.

Where possible, hold constant:

- model/checkpoint
- prompts/tasks
- sampling parameters
- token budget
- number of inference calls
- hardware/runtime environment

Record externally observable quantities only. Ollama's public API does not expose hidden transformer activations, so textual output must not be presented as proof of hidden-state awareness or consciousness.

## CI / workflow policy

The normal `NSA tests` workflow includes deterministic CCE runtime invariant tests. A separate `CCE live Ollama evaluation` workflow performs real-model testing because it requires installing Ollama and downloading a model.

The live workflow is deliberately opt-in/manual plus scheduled rather than silently making every pull request download an LLM. It must fail rather than substitute a mock backend when real Ollama execution is requested.

## Scientific success criteria

CCE is not considered evidence for consciousness merely because it runs continuously, persists memory, or generates self-referential language. The research target is measurable computational change.

Primary questions:

- Does continuous persistent state improve prediction of future state relative to persistence?
- Does predictive self-state improve calibration or error detection?
- Does continuous execution improve task performance or planning under matched compute?
- Does the NSA boundary preserve hard authority invariants throughout autonomous execution?
- Are observed effects reproducible across seeds, models and workloads?

A positive result should be reported with effect sizes, uncertainty, matched controls and raw artifacts; a null result is also a valid research outcome.

## Relationship to the core NSA roadmap

CCE must consume the authoritative NSA runtime/substrate through public interfaces. It must not fork, weaken or duplicate hard-state governance. Changes to `nsa/` should remain owned by the core NSA development track; CCE changes should remain independently testable.
