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
- [x] First learned one-step state predictor with explicit persistence baseline.
- [x] Deterministic predictive-dynamics CI experiment.
- [ ] Establish reproducible cross-seed advantage across models/tasks.
- [ ] Add held-out trajectory evaluation as a mandatory deployment gate.
- [ ] Add counterfactual action consequence prediction.
- [ ] Add capability/resource prediction.
- [ ] Calibrate prediction uncertainty against observed state transitions.

**First live-model evidence (2026-08-26, not yet sufficient to check the boxes
above):** `experiments/live/cce_live_capability_benchmark.py` runs a 4-way
matched (stateless / raw_context / persistent_cce / predictive_cce) live-Ollama
capability benchmark reusing the validated `_kalman.py` estimator. On
qwen2.5:0.5b, all 4 gates pass and replicate across 2 independent 5-seed sets.
On qwen2.5:1.5b, an initial run failed `predictive_beats_raw_context`; tracing
the model's raw outputs against the exact Kalman filter values it was shown
found it was double-applying drift on top of an already-current estimate. A
prompt fix (state explicitly that the given estimate needs no further
extrapolation) more than halved this model's `predictive_cce` error but did
not close the gap. A second bug was then found while investigating further: a
fixed implausibility cutoff missed a real hallucination (`149.17` when the
true range for that episode was ~[10, 30]); replacing it with a bound derived
from the environment's actual generative range (not a new arbitrary number)
narrowed the gap further without changing its direction. After both fixes,
20-seed runs on two independent seed sets settle it as a genuine negative
result -- `raw_context` reliably beats `predictive_cce` for this model on this
task (0.934 vs 0.919, and 0.919 vs 0.843). The follow-up hypothesis
("predictive state helps less as the model gets more capable") was tested
directly with a third model, qwen2.5:3b, and did **not** hold: 3b passed all 4
gates cleanly (0.937 vs 0.930), contradicting a simple capability-scaling
story, and confirmed again on a second, disjoint 20-seed set (0.939 vs 0.925).
The qwen2.5:1.5b result looks like a model-specific idiosyncrasy, not a point
on a clean trend. A fourth model from a different family, `llama3.2:1b`, was
then tried and confirmed on two independent 5-seed sets: `persistent_cce`
passed both times, but `predictive_cce` failed `predictive_beats_raw_context`
both times -- this time via a third, distinct failure mode (the model
sign-flips its numeric answer on many turns, plausibly mirroring the sign of
the drift figure shown in the prompt, rather than hallucinating an unrelated
plausible number the way qwen2.5:1.5b does). Two different model families now
both show `predictive_cce` failing on at least one size, each confirmed on
two independent seed sets, via two different failure modes, while
`persistent_cce` has passed on every model tried so far -- suggesting
`predictive_cce`'s prompt phrasing (not predictive state itself) is the
fragile part, and that `persistent_cce` is the more robust integration target
for now. This is one task, four models across two families, CPU-only, single
machine, each model checked on two independent seed sets -- a data point
toward the items above, not a satisfaction of them. See
[`LIVE_CAPABILITY_BENCHMARK.md`](LIVE_CAPABILITY_BENCHMARK.md) for full results
and the prioritized next steps (ablate both models' specific failure-mode
hypotheses, more task types, before revisiting production wiring).

## CCE — Continuous Cognitive Engine

CCE is now an explicit runtime research track bridging persistent cognitive state and the authoritative NSA substrate.

### Architecture

```text
REAL / LIVE INPUT
       |
       v
+---------------------------+
| CCE                       |
| persistent runtime       |
| continuous state field    |
| predictive dynamics       |
+------------+--------------+
             |
             v
   authoritative NSA boundary
             |
             v
        canonical state
             |
             +------> Ollama inference
             |              |
             |              v
             |       observable proposal
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
- [x] PR-triggered live Ollama CCE workflow (`opened`, `synchronize`, `reopened`).
- [x] Continuous soft-state dynamics independent of Ollama inference calls.
- [x] Asynchronous input injection and explicit input reduction policy.
- [x] Learned one-step predictive state dynamics adapter.
- [x] Persistence-baseline comparison for learned dynamics.
- [x] Predictive dynamics experiment included in the PR CCE gate.
- [ ] Connect a validated learned predictor to the live continuous field behind an explicit opt-in mode.
- [ ] Held-out predictive evaluation across seeds and model/task families.
- [ ] Continuous latent-state dynamics at sub-inference timescales using learned state transition models.
- [ ] Real asynchronous speech/vision sensor deployment against the canonical runtime.
- [ ] Long-duration stability experiments.
- [ ] Four-way matched cognition benchmark: stateless, persistent, clocked CCE, continuous predictive CCE.

## Experimental controls

Every continuous-cognition claim must retain a matched control:

1. **Baseline:** live Ollama without persistent NSA state.
2. **Persistent:** live Ollama with canonical persistent NSA state but no autonomous scheduler.
3. **Clocked CCE:** CCE with explicit deterministic/finite stepping.
4. **Continuous CCE:** CCE enabled on the wall-clock runtime loop.

For predictive dynamics, additionally compare against:

5. **Persistence predictor:** `X_hat(t+dt) = X(t)`.
6. **Learned predictor:** `X_hat(t+dt) = F_theta(X(t), I(t), G(t))`.

Where possible, hold constant:

- model/checkpoint
- prompts/tasks
- sampling parameters
- token budget
- number of inference calls
- hardware/runtime environment

Record externally observable quantities only. Ollama's public API does not expose hidden transformer activations, so textual output must not be presented as proof of hidden-state awareness or consciousness.

## CI / workflow policy

The normal `NSA tests` workflow includes deterministic CCE runtime invariant tests. The `CCE live Ollama evaluation` workflow now runs automatically for every opened, synchronized or reopened pull request, in addition to manual and scheduled execution.

The live workflow caches the Ollama model store and performs real-model testing. It must fail rather than substitute a mock backend when real Ollama execution is requested. Each PR run produces machine-readable matched, continuous and predictive-dynamics artifacts.

## Scientific success criteria

CCE is not considered evidence for consciousness merely because it runs continuously, persists memory, or generates self-referential language. The research target is measurable computational change.

Primary questions:

- Does continuous persistent state improve prediction of future state relative to persistence?
- Does predictive self-state improve calibration or error detection?
- Does continuous execution improve task performance or planning under matched compute?
- Does the NSA boundary preserve hard authority invariants throughout autonomous execution?
- Are observed effects reproducible across seeds, models and workloads?

A positive result should be reported with effect sizes, uncertainty, matched controls and raw artifacts; a null result is also a valid research outcome.

A predictor must beat the persistence baseline on held-out trajectories before it is permitted to drive the live continuous cognitive field.

## Relationship to the core NSA roadmap

CCE must consume the authoritative NSA runtime/substrate through public interfaces. It must not fork, weaken or duplicate hard-state governance. Changes to `nsa/` should remain owned by the core NSA development track; CCE changes should remain independently testable.
