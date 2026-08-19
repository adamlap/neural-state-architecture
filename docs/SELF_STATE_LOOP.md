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
│                    corrective + learned residual
│                              │
│                              ▼
│                     bounded state update
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

The regulator now has an explicit corrective component:

\[
\Delta\sigma_t = \delta_{max}\tanh(-k\tanh(e_t)+r_\theta(e_t))
\]

where `k >= 0` is the correction gain and `r_theta` is a zero-initialized,
trainable residual. Thus the untrained architecture has a defined corrective
direction rather than an arbitrary random perturbation, while training can learn
additional state-dependent regulation.

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

`regulation_delta` is also exposed so the actual intervention can be measured directly.

## Recovery benchmark

`experiments/self_state/recovery_curve.py` performs a recurrent rollout. A disturbed
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

The perturbation sweep additionally reports normalized residual distance and
area-under-the-distance-curve (AUC). A positive recovery advantage does not
necessarily imply a positive AUC advantage: early correction and later drift
must be distinguished rather than collapsed into a single claim.

No result should be described as evidence of intelligence or consciousness without
controlled training and statistical replication.

## Current empirical status

The initial seed-42 sweep showed a positive final-recovery advantage for 6/7
perturbation magnitudes, but a negative mean AUC advantage. In particular, the
feedback trajectory often corrected strongly in the first recurrent step and then
drifted upward. This is evidence that the original random regulator was not yet a
reliable recovery controller; it is not evidence of a solved self-stabilization
problem.

The regulator has therefore been changed from a random proposal to an explicit
bounded contraction term plus a zero-initialized trainable residual. The next sweep
should test whether this architectural change improves both final recovery and
trajectory-level AUC across seeds.

## Research direction

1. Run the corrected regulator across multiple seeds and perturbation magnitudes.
2. Measure final recovery, AUC, settling time, peak residual, and regulation magnitude.
3. Sweep correction gain and maximum delta to identify stable regions.
4. Train the residual regulator and compare against the deterministic contraction baseline.
5. Sweep affected state dimensions rather than perturbing every soft coordinate equally.
6. Test hard-state attacks separately from soft-state disturbances.
7. Measure whether self-state regulation improves capability while preserving invariants.
8. Eventually evaluate long-horizon recurrent planning and counterfactual self-modeling.
