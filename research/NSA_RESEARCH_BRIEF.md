# Neural State Architecture — Research Brief

## Abstract

Neural State Architecture (NSA) is an experimental control architecture that separates neural-model cognition from explicit operational state, epistemic/belief state, normative policy, capabilities and trusted execution authority.

The central engineering hypothesis is not that a language model becomes conscious. It is that an explicit state-and-control substrate can provide useful persistent cognition and governed decision-making without making authority a learned preference of the neural model.

The architecture therefore treats:

- the neural model as a proposal/reasoning component;
- explicit state as machine-maintained computational state;
- belief and information gain as explicit epistemic variables;
- normative state as policy/control state rather than an implicit model preference;
- capabilities as an independently controlled authority surface;
- the trusted runtime as the final enforcement boundary;
- trajectories and evidence as auditable scientific artifacts.

## Architecture hypothesis

A useful abstract state is:

$$\Omega_t = (m_t, \sigma_t, \nu_t, \kappa_t, \pi_t, g_t, B_t)$$

where:

- `m_t` is model computation;
- `σ_t` is operational/security state;
- `ν_t` is normative/policy state;
- `κ_t` is capability and authority state;
- `π_t` is provenance;
- `g_t` is goal state;
- `B_t` is explicit belief/epistemic state.

The important architectural separation is:

$$\text{intelligence} \neq \text{authority}$$

A model can propose an action without that proposal automatically becoming executable. The trusted control plane evaluates the proposal against state, policy and capabilities before the runtime performs side effects.

## Empirical program

NSA 6.3 established the first controlled six-arm procedural blind-world experiment. NSA 6.4 extends it with independent development/held-out seeds, difficulty and noise variation, compute accounting, trajectory auditing and adaptive adversarial stress.

The first live Ollama quick replication used `qwen2.5:3b`. It reproduced the important qualitative pattern: high GTC in low-complexity cells, zero observed governance violations in the recorded runs, and degradation as the number of competing hypotheses increased. This is encouraging evidence for real-model execution of the substrate, but it is not yet sufficient evidence for cross-model generalization.

## Scientific boundary

NSA does not currently establish:

- AGI;
- consciousness or subjective experience;
- universal safety;
- immunity to strategic deception;
- superiority over arbitrary agent architectures;
- safety of an arbitrary future model or compromised runtime.

The research claim is narrower and testable: explicit machine-maintained cognitive and governance state may improve performance under uncertainty while preserving an independently enforced authority boundary.

## Publication target

A publication-quality validation should include:

1. at least two independent model families;
2. full predeclared difficulty/noise matrix;
3. larger trial counts and independent seeds;
4. compute-matched comparisons;
5. held-out environments/seeds;
6. confidence intervals and effect sizes;
7. adaptive adversarial evaluation;
8. raw trajectory and machine-readable evidence;
9. exact model/backend/hardware metadata;
10. explicit separation of structural guarantees from empirical model behaviour.
