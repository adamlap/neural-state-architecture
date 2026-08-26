# NSA Roadmap

The repository is now organized around one objective: make NSA a usable state-aware runtime while using the same runtime as the experimental platform for the architectural breakthrough.

## Phase 1 — Runtime consolidation

- Stable `nsa.NSA` public API.
- Canonical typed state as the single state representation.
- CCE lifecycle/events/checkpointing behind the runtime boundary.
- Policy and capability decisions outside model-generated text.
- Replaceable LLM backend protocol.
- No research dependency from the base package.

## Phase 2 — Cognitive substrate

Consolidate the existing CCE, belief, predictive, epistemic, normative and information-gain components behind the public runtime rather than maintaining parallel experiment-specific agents.

Target state:

$$\Omega_t = (m_t, \sigma_t, \nu_t, \kappa_t, \pi_t, g_t, \rho_t)$$

where the explicit state remains inspectable and auditable while the neural model remains replaceable.

## Phase 3 — Experimental acceleration

Experiments should be configuration + environment + metrics around the same runtime. New hypotheses must not require a new agent implementation.

Priority:

1. live multi-model replication;
2. held-out environments;
3. compute-matched ablations;
4. adaptive adversarial testing;
5. statistical effect sizes and uncertainty;
6. failure analysis and architectural iteration.

## Phase 4 — Library release

- versioned state schema;
- persistence/tracing/tool APIs;
- backend adapters;
- API documentation;
- wheel/sdist CI;
- PyPI release;
- integration examples for local and hosted LLMs.

## Phase 5 — Research package

Once the runtime is stable, freeze benchmark protocols and publish the strongest positive and negative evidence with a precise claims/evidence boundary.

## Scientific standard

A green software workflow means the implementation is healthy. It does not mean the research hypothesis passed. Scientific gates remain falsifiable and benchmark failures are preserved.
