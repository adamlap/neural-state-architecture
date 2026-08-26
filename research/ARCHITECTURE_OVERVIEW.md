# NSA Architecture Overview

## Design principle

NSA separates **neural intelligence** from **execution authority**.

```text
                    Neural Model
                 proposals / reasoning
                         │
                         ▼
              ┌─────────────────────┐
              │   NSA State Plane   │
              │ operational state   │
              │ belief / epistemic  │
              │ normative policy    │
              │ provenance / goals  │
              │ capability state    │
              └──────────┬──────────┘
                         │
                  SecurityDecision
                         │
              ┌──────────┴──────────┐
           DENY / ESCALATE       ALLOW
                                    │
                                    ▼
                         Trusted Runtime
                       tools / files / net /
                         external effects
```

## State decomposition

$$\Omega_t=(m_t,\sigma_t,\nu_t,\kappa_t,\pi_t,g_t,B_t)$$

The exact implementation is richer and typed; this is a conceptual map.

- `m_t`: model cognition and proposals.
- `σ_t`: operational/security state.
- `ν_t`: normative/policy state.
- `κ_t`: capability/authority state.
- `π_t`: provenance.
- `g_t`: goals.
- `B_t`: epistemic/belief state and uncertainty.

## CCE

CCE provides persistent state and wall-clock dynamics around model inference. The research question is whether persistent/predictive state has measurable computational value beyond stateless inference and ordinary context.

## Trusted boundary

$$\text{model preference}\neq\text{runtime authority}$$

A model output is a proposal. The trusted runtime decides whether an external side effect can occur. A model refusal is therefore not the security boundary; capability/execution enforcement is.
