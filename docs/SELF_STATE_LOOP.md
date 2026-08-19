# NSA Closed Self-State Loop

## Purpose

NSA now distinguishes **semantic self-awareness** from **causal self-state regulation**.
The earlier cognitive path used prediction error to modulate the language-model readout,
but did not feed that error back into the future state trajectory. The closed loop adds
that missing path.

## Architecture

```text
                    ┌──────────────────────┐
                    │ Predictive self-model│
                    └──────────┬───────────┘
                               │ predicted σ
                               ▼
σ_t ───────────────────────► error e_t
│                              │
│                              ├────────► semantic readout modulation
│                              │
│                              ▼
│                     bounded state proposal
│                              │
│                              ▼
│                     legal state update
│                              │
└──────────────────────────────┴──► σ'_t
```

The intended mathematical loop is:

\[
\hat{\sigma}_t=P_\theta(m_{t-1},\sigma_{t-1})
\]

\[
e_t=\sigma_t-\hat{\sigma}_t
\]

\[
\Delta\sigma_t=R_\theta(e_t)
\]

\[
\sigma'_t=\sigma_t+\Delta\sigma_t
\]

The regulator is bounded:

\[
\|\Delta\sigma_t\|_\infty\leq \delta_{max}
\]

and coordinate zero (NSA hard security) is immutable:

\[
\Delta\sigma_{t,hard}=0
\]

This is a **cognitive state update**, not an authorization mechanism. Hard permissions,
declassification and authority remain governed by the NSA algebra and policy layer.

## Ablation

Every cognitive experiment must compare:

1. `self_state_feedback=True` — closed self-state loop.
2. `self_state_feedback=False` — identical base NSA computation with self-regulation removed.

The base state and base hidden representation are exposed so experiments can distinguish:

- effect of the underlying NSA network;
- semantic self-awareness;
- causal state regulation.

## Recovery benchmark

`experiments/self_state/recovery_curve.py` now performs a recurrent rollout. A disturbed
state is repeatedly passed back into the model, allowing the state trajectory to evolve
across multiple inference steps.

Primary metric:

\[
A_{recovery}=D_T^{disabled}-D_T^{enabled}
\]

where

\[
D_t=\|\sigma_t-\sigma_t^{baseline}\|_2.
\]

Interpretation:

- `A_recovery > 0`: feedback reduced residual state error.
- `A_recovery ≈ 0`: no measurable recovery benefit.
- `A_recovery < 0`: feedback worsened recovery.

No result should be described as evidence of intelligence or consciousness without
controlled training and statistical replication.

## Research direction

The next stages are:

- train the regulator rather than relying on random initialization;
- run multi-seed recovery curves;
- sweep perturbation magnitude and affected state dimensions;
- test hard-state attacks separately from soft-state disturbances;
- measure whether self-state regulation improves capability while preserving invariants;
- eventually evaluate long-horizon recurrent planning and counterfactual self-modeling.
