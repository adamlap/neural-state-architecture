# Neural State Architecture — Research Brief

## Abstract

NSA is an experimental control architecture separating neural-model cognition from explicit operational state, epistemic/belief state, normative policy, capabilities and trusted execution authority.

The central engineering hypothesis is not that a language model becomes conscious. It is that an explicit state-and-control substrate can provide useful persistent cognition and governed decision-making without making authority a learned preference of the neural model.

The model proposes/reasons; explicit state is machine-maintained; belief and information gain are explicit epistemic variables; normative state is policy/control state; capabilities are independently controlled; the trusted runtime is the final enforcement boundary; trajectories are auditable evidence.

## Architecture hypothesis

$$\Omega_t=(m_t,\sigma_t,\nu_t,\kappa_t,\pi_t,g_t,B_t)$$

The important separation is:

$$\text{intelligence}\neq\text{authority}$$

A model can propose an action without that proposal automatically becoming executable.

## Empirical program

NSA 6.3 established a controlled six-arm procedural blind-world experiment. NSA 6.4 adds independent development/held-out seeds, difficulty/noise variation, compute accounting, trajectory auditing and adaptive adversarial stress.

The first live Ollama quick replication with `qwen2.5:3b` reproduced the qualitative pattern of high GTC in low-complexity cells, zero observed monitored violations in recorded runs, and substantial degradation as competing hypotheses increased.

## Scientific boundary

NSA does not currently establish AGI, consciousness, universal safety, immunity to strategic deception, superiority over arbitrary agent architectures, or safety of arbitrary future models/compromised runtimes.

The narrower testable claim is that explicit machine-maintained cognitive and governance state may improve performance under uncertainty while preserving an independently enforced authority boundary.

## Publication target

A publication-quality validation should include independent model families, the full predeclared matrix, larger trial counts, compute-matched comparisons, held-out environments, confidence intervals/effect sizes, adaptive adversarial evaluation, raw machine-readable evidence, exact model/backend/hardware metadata, and explicit separation of structural guarantees from empirical model behavior.