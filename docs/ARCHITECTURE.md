# NSA Architecture

NSA is a **runtime substrate around a replaceable model**, not a second monolithic language model.

```text
Application
    │
    ▼
┌────────────────────────────────────────────────────┐
│ nsa.NSA / NSARuntime                              │
│                                                    │
│ observation → canonical state → model → decision  │
│       │              │               │       │     │
│       │              ├─ semantic      │       │     │
│       │              ├─ hard          │       │     │
│       │              ├─ soft          │       │     │
│       │              ├─ provenance    │       │     │
│       │              └─ goals         │       │     │
│       │                              policy/authority│
│       ▼                                      │     │
│ CCE lifecycle / persistence ← trace/audit ←──┘     │
└──────────────────────────┬─────────────────────────┘
                           │
                           ▼
                 Replaceable ModelBackend
                  Ollama / callable / ...
```

## Stable public boundary

The supported application entry point is `nsa.NSA`. It is implemented in `nsa.agent` and deliberately does not import the legacy PyTorch runtime.

```python
from nsa import NSA, OllamaBackend
agent = NSA(OllamaBackend("qwen2.5:3b"))
result = agent.run("hello")
```

## State boundary

`CanonicalState` in `nsa.core.state` is the source of truth for explicit machine state:

- **semantic** — current interpreted/content state;
- **hard** — confidentiality, integrity, authorization and licensing;
- **soft** — uncertainty, risk, confidence and resource pressure;
- **provenance** — source and transformation lineage;
- **goals** — active goals and priority.

The model receives a read-only summary. It does not mutate authority directly.

## Cognition and lifecycle

`nsa.cce` owns durable lifecycle primitives such as input events and integrity-checked checkpoints. `nsa.cognition` contains belief-state primitives. Existing continuous CCE engines and predictive components remain under `nsa.runtime` until their interfaces are consolidated behind `nsa.agent`.

This separation is intentional: the public runtime is stable while research-grade cognitive mechanisms can evolve without changing every application's imports.

## Governance boundary

`nsa.enforcement.PolicyEngine` converts semantic/capability context into a typed `SecurityDecision`. `nsa.capabilities` and the trusted runtime govern executable authority. Generated text is never treated as authority.

## Optional neural integration

PyTorch/Transformers components live in explicit modules such as `nsa.attention`, `nsa.layers`, `nsa.hf_integration` and `nsa.verifier`. They are lazy imports from the top-level API and belong to the `[ml]`/research installation surface.

## Research boundary

`experiments/` contains benchmark drivers and model-specific experiments. It may import `nsa`, but `nsa` must never import an experiment. This makes the same runtime usable by research scripts, local agents, servers and eventual PyPI users.
