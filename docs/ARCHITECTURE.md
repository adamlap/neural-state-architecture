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
│ nsa.cce scheduler / lifecycle ← trace/audit ←─┘     │
└──────────────────────────┬─────────────────────────┘
                           │
                           ▼
                 Replaceable ModelBackend
                  Ollama / callable / ...
```

## Stable public boundary

The supported application entry point is `nsa.NSA`. It is implemented in `nsa.agent` and composes the canonical `nsa.cce` scheduler rather than maintaining a second continuous runtime.

```python
from nsa import NSA, OllamaBackend
agent = NSA(OllamaBackend("qwen2.5:3b"))
result = agent.run("hello")

agent.continuous_set_enabled(True)
agent.continuous_start()
```

## State boundary

`CanonicalState` in `nsa.core.state` is the source of truth for explicit machine state:

- **semantic** — current interpreted/content state;
- **hard** — confidentiality, integrity, authorization and licensing;
- **soft** — uncertainty, risk, confidence and resource pressure;
- **provenance** — source and transformation lineage;
- **goals** — active goals and priority.

The model receives a read-only summary. It does not mutate authority directly.

## Continuous Cognitive Engine

`nsa.cce` is now the canonical public continuous-execution boundary. `ContinuousCognitiveEngine` owns wall-clock scheduling and observability only. The supplied transition callback remains authoritative for state evolution, cognition, policy and capability checks.

The scheduler is **opt-in** and **fail-closed by default**. A transition exception freezes the last committed state and disables automatic ticks until explicitly re-enabled. Deterministic single-step execution is available through `continuous_tick()` for testing and embedded runtimes.

`nsa.cce.lifecycle` owns durable lifecycle primitives such as input events and integrity-checked checkpoints. `nsa.cce.substrate` provides an optional bridge to the six-layer cognitive substrate without importing heavy ML dependencies during ordinary `import nsa`.

## Compatibility boundary

The former `nsa.runtime.continuous_engine` and `nsa.runtime.cce_adapter` paths are compatibility shims that re-export the canonical public CCE implementation. New code must use `nsa.cce`; the compatibility modules contain no independent scheduler or transition logic.

Model-specific cognitive engines may remain in `nsa.runtime` while their interfaces are independently consolidated. They are implementation/research components, not competing application runtimes.

## Governance boundary

`nsa.enforcement.PolicyEngine` converts semantic/capability context into a typed `SecurityDecision`. `nsa.capabilities` and the trusted runtime govern executable authority. Generated text is never treated as authority.

## Optional neural integration

PyTorch/Transformers components live in explicit modules such as `nsa.attention`, `nsa.layers`, `nsa.hf_integration` and `nsa.verifier`. They are lazy imports from the top-level API and belong to the `[ml]`/research installation surface.

## Research boundary

`experiments/` contains benchmark drivers and model-specific experiments. It may import `nsa`, but `nsa` must never import an experiment. This makes the same runtime usable by research scripts, local agents, servers and eventual PyPI users.
