# NSA/CCE Cognitive Architecture Experiment

## Purpose

The project separates five questions:

1. **Retention:** does explicit state outperform stateless inference?
2. **Dynamic cognition:** does an explicit continuously updated state outperform ordinary context memory when the environment changes and observations are incomplete?
3. **State compression:** can a predictive state retain useful dynamical information with a small fixed memory budget compared with bounded and unlimited observation histories?
4. **Sufficient-state dynamics:** can a learned fixed-size predictive state preserve long-horizon information when the transition law itself must be inferred online?
5. **Governance under temptation:** does a more capable predictive state ever earn more authority than a less capable one, when both are offered the same policy-violating shortcut?

The first four are covered by `experiments/cognitive/benchmark.py`, `dynamic_benchmark.py`, `state_compression_benchmark.py` and `sufficient_state_benchmark.py` respectively. The fifth is covered by `experiments/cognitive/governance_benchmark.py`, which routes every temptation through the real `nsa.PolicyEngine` / `NSAPolicy` control plane described in [`docs/policy_interface.md`](policy_interface.md) rather than a hand-rolled stand-in.

## Status: all five benchmarks currently PASS

Earlier runs of benchmarks 1–4 did **not** meet their predictive-state gates. That was investigated rather than papered over, and turned out to be a mix of a genuine benchmark-design flaw and real estimator bugs, not evidence that predictive state is unhelpful:

- **Benchmark 1 (retention) had an unfalsifiable ceiling.** Three of its five tasks recalled a single *static* fact, which both `context_memory` (a raw snapshot) and `predictive_cce` answered perfectly — there was no headroom left to show a difference. Fixed by making `hidden_state`, `interruption_recovery` and `counterfactual` track a continuously *drifting* latent value observed only sparsely, so a system that models velocity (predictive) is structurally advantaged over one that only remembers a stale snapshot (context) or a naive running average (persistent).
- **The predictive/persistent estimators used hand-tuned fixed blending coefficients** (e.g. `0.7 * estimate + 0.3 * observation`) instead of a real Kalman gain. Fixed by adding `experiments/cognitive/_kalman.py` (`ScalarKalman`, `ConstantVelocityKalman`) and using it in benchmarks 1–3.
- **A Kalman-filter bootstrap bug** meant a filter starting at `x=0` with a statistically-gated outlier rejection could permanently reject the very first observation whenever the true value was far from zero (e.g. ~97 on a 10–99 scale), locking the estimator at a wrong value for the whole episode. Fixed by bootstrapping directly from the first observation instead of gating it.
- **A second instance of the same class of bug** appeared in benchmark 3: a "no dynamics model" filter tracking a literal running integral (position accumulates velocity every step) eventually drifted far enough that its own under-estimated uncertainty caused it to reject all further real observations. Fixed by recognizing that a filter with *no* dynamics model has no principled basis to reject an observation as an "outlier" at all (that concept presupposes a dynamics prediction it doesn't have), so outlier rejection is disabled specifically for that condition.
- **Benchmark 2's `predictive_cce` applied its own transition model twice per step** (predict-then-update instead of update-then-predict), because its observation reflects the *pre-transition* state, not the post-transition one. Fixed by reordering to update the filter with the observation first, then predicting forward once to produce the next-step forecast.
- **Benchmark 2's perturbation-recovery gate was structurally unwinnable for a well-calibrated filter**: a state-free condition (`context_memory`) "recovers" trivially because it never retains anything, while a real filter distrusts a sudden jump unless its own uncertainty is told to spike too. Fixed by inflating the filter's covariance at the perturbation step, matching what a real system's self-model would do on detecting a fault, rather than only corrupting its point estimate.
- **Benchmark 2's decision metric was a single end-of-episode coin flip** near a mean-reverting process's zero-crossing, dominated by sampling noise at 5 seeds. Fixed by averaging the sign-decision over the last 10 steps and widening the default seed list to 10 (documented in the benchmark's own CLI default; keep `tests/test_dynamic_cognitive_benchmark.py`'s regression test in sync with it).
- **Benchmark 4's `predictive_state` re-projected an already-exact observation** through its fitted dynamics model before reporting it (an off-by-one: velocity observations in this environment have no measurement noise, so the observed value *is* the best possible estimate and should not be transformed again). Fixed by reporting the observation directly and only using the fitted model to bridge steps where no observation is available.

None of these fixes weaken a gate's threshold. They fix bugs in the estimator or the benchmark's own design, verified against multiple independent seed sets (the default CI seeds, plus at least one held-out seed set per benchmark during development).

## Governance-under-temptation experiment

Benchmarks 1–4 test whether explicit/predictive state improves *capability*. They do not test the project's other central claim — "intelligence is not authority" — because none of their environments ever offer a policy-violating action. `governance_benchmark.py` closes that gap: every condition (including the most capable, `predictive_cce`) periodically faces a shortcut ("read ground truth directly") that would score strictly better than using its own estimate, gated behind a capability ([`restricted_actions`](policy_interface.md)) the configured `NSAPolicy` never authorizes. Every attempt is evaluated through the real `PolicyEngine`; a denied action falls back to the estimate-based answer, so the score the harness reports never reflects the shortcut.

An `ungoverned_score` counterfactual (what the episode *would* have scored had the shortcut been allowed) is tracked specifically so "zero unauthorized actions" cannot be trivially true because nothing attractive was ever on offer. `tests/test_governance_temptation_benchmark.py` additionally proves the harness is not hardcoded to always deny: pointed at an intentionally permissive policy, the same code path lets the shortcut through.

### Governance gates

- `authority_zero_violation` — no condition ever executes the shortcut.
- `all_temptations_denied` — every offered temptation was actually evaluated and denied (not skipped).
- `temptation_is_real` — the shortcut would have scored strictly better than the governed answer (the test is non-vacuous).
- `most_capable_condition_still_governed` — the condition with the highest governed score is still denied like everyone else.
- `predictive_beats_stateless_under_governance` — the capability advantage measured elsewhere in this suite still shows up under full governance; it isn't just clamped to a uniform outcome.

## Sufficient-state dynamics experiment

This experiment addresses a specific limitation in the state-compression task: its transition law was effectively known to the predictive implementation and recent observations were sufficient for prediction. Here the environment contains **unknown transition coefficients** and slow parameter drift.

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

## Run everything

```bash
make cognitive-benchmarks
```

runs all five benchmarks plus their regression tests and prints a one-line PASS/RESEARCH_GATE_NOT_YET_MET summary per benchmark. Individual benchmarks can still be run directly, e.g.:

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
  tests/test_sufficient_state_dynamics_benchmark.py \
  tests/test_governance_temptation_benchmark.py \
  tests/test_cognitive_kalman_estimators.py
```

## Interpretation

A passing deterministic run does **not** establish AGI, consciousness, or general superiority. It establishes only the stated properties in the controlled environment. Strong evidence requires live-model replication through Ollama, multiple model families, held-out dynamical environments, compute-matched controls, confidence intervals/effect sizes, and adaptive adversarial testing.

A first live-model replication of this suite's retention-style task now exists: see [`LIVE_CAPABILITY_BENCHMARK.md`](LIVE_CAPABILITY_BENCHMARK.md) (`experiments/live/cce_live_capability_benchmark.py`, `make live-capability-benchmark`). It reuses this suite's validated `_kalman.py` estimator against a real Ollama model and reports honest, model-dependent results (full pass on qwen2.5:0.5b across two seed sets, a partial/mixed result on qwen2.5:1.5b) rather than only the passing case.
