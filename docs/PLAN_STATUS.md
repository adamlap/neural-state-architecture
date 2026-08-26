# NSA Plan Status

Consolidated implementation-snapshot addenda for PLAN.md.

---

## Live Runtime Integrity

This addendum records a critical implementation boundary for the current roadmap.

## Live Ollama status

The repository now has a real `NSAGovernedInference` runtime envelope. A live backend call is made only after the deterministic NSA safety kernel evaluates the generation transition. After the backend returns, NSA commits a typed state transition and provenance hash.

The API proxy uses this governed path. It no longer appends a synthetic `Verified [OK]` badge, fabricates a model digest, or presents a prompt-only Ollama Modelfile as an intrinsic NSA model.

## Scientific boundary

This implementation is **runtime governance**, not native neural-weight integration. The underlying Ollama model remains unchanged. `weight_modification=false` is intentionally exposed by the runtime status API.

Therefore the evidence level is:

- **Implemented:** real backend + deterministic NSA reference-monitor path.
- **Unit-tested:** backend invocation, state advancement and provenance chaining.
- **Not demonstrated:** whole-model intrinsic information-flow control inside Ollama's transformer computation.
- **Open research:** native NSA integration, hidden-state/activation mediation, and statistically rigorous capability/safety comparisons.

## Roadmap consequences

1. Treat runtime governance as an explicit Phase 21 integration milestone, not as proof of native model wrapping.
2. Keep prompt-only `Modelfile.nsa` clearly labelled as a non-security-boundary convenience profile.
3. Prioritize a native/retrofit adapter that mediates actual model representations before claiming intrinsic neural NSA protection.
4. Add live-vs-baseline safety and capability benchmarks around the same checkpoint and compute budget.
5. Link every live claim to reproducible artifacts and exact implementation commits.

---

## Phase 11 Canonical Typed Neural Core

This document records the first executable slice of Phase 11 in `PLAN.md`.

## Implemented in this branch

- Canonical typed activation protocol in `nsa/core/typed_activation.py`.
- Explicit state domains for semantic, soft, hard, epistemic, provenance, temporal and goal state.
- Explicit ownership metadata distinguishing model-writable state from trusted-runtime state.
- Model proposals for model-owned fields are non-mutating; hard authority state cannot be proposed as a model write.
- Runtime commits return a new `UnifiedCognitiveState` view instead of mutating the previous object in place.
- Versioned JSON-compatible serialization of the canonical state contract.
- Tests for hard-state write rejection, immutable-style runtime transitions, and serialization.

## Scientific boundary

This is a **software contract**, not a hardware or process-isolation security boundary. Python callers with arbitrary process access can still bypass it. The security guarantee remains dependent on the trusted NSA runtime/kernel controlling the actual execution boundary.

Likewise, this does not yet make an Ollama transformer's hidden activations NSA-native. The live Ollama integration remains a real runtime reference monitor as documented in `docs/PLAN_LIVE_RUNTIME_STATUS.md`.

## Remaining Phase 11 work

- Partial activation/state-vector support.
- Full compatibility adapters for legacy `StateVector`, `MultiStateVector` and quad-tuple APIs.
- General state composition semantics across heterogeneous domains.
- Native/retrofit neural adapters that carry this protocol into actual model representations.

---

## Phase 12 General State Algebra Engine

Updated 2026-08-19.

## Completed in merged work

- Canonical heterogeneous product state.
- Boolean domain.
- Capability-set domain.
- Bounded numeric domain.
- Ordered finite-enum domain.
- Constraint-set domain.
- Probability-interval lattice with explicit bottom element.
- Temporal-window lattice with explicit bottom element.
- Coordinate-wise join/meet and product partial order via `x ⊔ y = y`.
- Semantic domain compatibility checks and incompatible-product rejection.
- Mixed products containing the new domains.
- Explicit public exports from `nsa.core`.
- Domain-specific unit tests.
- Property-based verification of commutativity, associativity, idempotence and absorption across the supported lattice domains and product composition.
- Coordinate-wise legal transition cones with increase/decrease/unchanged directions.
- Exact product projection onto legal transition cones.
- Projection legality and fixed-point tests.

## Remaining Phase 12 work

- Formal proofs/verification of the heterogeneous laws beyond executable property testing.
- A principled representation for richer probabilistic distributions rather than only probability intervals.
- Temporal semantics tied to runtime clocks/events rather than numeric ticks.
- Constraint implication/solver semantics rather than opaque constraint identifiers.
- Integration of transition-cone enforcement into the real NSA runtime transition path.
- Integration of exact projection with the neural/retrofit adapter so legality is enforced during model computation rather than only at the state-management layer.

## Scientific boundary

These domains are deterministic mathematical state infrastructure. They do not modify transformer weights or hidden activations. They should therefore be treated as a typed substrate for future NSA transition and neural adapters, not as evidence of intrinsic neural security by themselves.

---

## Phase 19 Predictive Self-Model and CCE

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

---

## CCE Continuous Cognition and Live Ollama

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
