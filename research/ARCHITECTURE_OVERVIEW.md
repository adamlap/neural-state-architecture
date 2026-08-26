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
              │                     │
              │ operational state   │
              │ belief / epistemic  │
              │ normative policy    │
              │ provenance          │
              │ goals               │
              │ capability state    │
              └──────────┬──────────┘
                         │
                  SecurityDecision
                         │
              ┌──────────┴──────────┐
              │                     │
           DENY / ESCALATE       ALLOW
                                    │
                                    ▼
                         Trusted Runtime
                       tools / files / net /
                         external effects
```

## State decomposition

A useful conceptual representation is:

$$\Omega_t = (m_t, \sigma_t, \nu_t, \kappa_t, \pi_t, g_t, B_t)$$

The exact implementation is richer and typed; the equation is a conceptual map, not a claim that every component is a single scalar/vector.

### `m_t` — model cognition

The language model performs language, reasoning and proposal generation. NSA does not require the model's weights to encode the authority policy.

### `σ_t` — operational/security state

Hard and soft state tracks relevant security, confidence and runtime conditions.

### `ν_t` — normative/policy state

Configured policy and normative information used to evaluate proposals. It remains distinct from hard authority state.

### `κ_t` — capability/authority

The set of actions the runtime can actually execute. Capability checks are part of the control boundary rather than merely instructions to the model.

### `π_t` — provenance

Origin and trust information used for auditability and policy decisions.

### `g_t` — goals

The legitimate objective being pursued by the agent/environment.

### `B_t` — epistemic/belief state

Explicit uncertainty over hypotheses/world states. Information gain provides a measurable quantity for active evidence acquisition.

## Continuous cognitive engine

The CCE provides persistent state and wall-clock dynamics around model inference. The key research question is whether persistent and predictive state provides useful computational value beyond stateless inference and ordinary context.

The current evidence should be interpreted conservatively: the live NSA 6.4 quick replication demonstrates the substrate running around a real model and preserving the governance invariant, but high-complexity task performance still degrades.

## Trusted boundary

The most important security distinction is:

$$\text{model preference} \neq \text{runtime authority}$$

A model output is a proposal. The trusted runtime decides whether an external side effect can occur. Therefore, a refusal produced by the model is not the security boundary; the capability/execution layer is.

This boundary is the foundation for the project's formal and adversarial verification work.
