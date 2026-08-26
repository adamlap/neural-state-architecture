# Development Guide

## Runtime first

Reusable capabilities belong in `nsa/`. Experiments belong in `experiments/`. If an experiment needs a new runtime capability, implement it once in `nsa/` and make the experiment consume it.

### Public API smoke test

```bash
python -c 'from nsa import NSA, EchoBackend; print(NSA(EchoBackend()).run("hello").text[:20])'
```

### Test the public runtime

```bash
python -m pytest -q tests/test_agent.py
```

### Test the complete repository

```bash
make test
```

## Dependency policy

The base package has no mandatory PyTorch dependency. Heavy neural/model integrations are optional:

```bash
pip install neural-state-architecture
pip install "neural-state-architecture[ml]"
```

This keeps the state-aware runtime usable on small servers and makes the package suitable for local Ollama deployments.

## Module ownership

| Module | Owns |
|---|---|
| `nsa.agent` | Stable high-level agent API. |
| `nsa.core` | Typed state and hard-state transitions. |
| `nsa.cce` | Continuous lifecycle, events and checkpoint primitives. |
| `nsa.cognition` | Belief/predictive cognitive primitives. |
| `nsa.capabilities` | Capability identity and authority. |
| `nsa.policy` | Policy representation/compiler. |
| `nsa.enforcement` | Runtime policy decisions. |
| `nsa.runtime` | Existing trusted/continuous runtime implementations being consolidated. |
| `nsa.attention`, `nsa.layers`, `nsa.hf_integration` | Optional neural retrofits. |
| `experiments` | Non-library scientific code. |
| `research` | Curated evidence and publication material. |

## Refactor rule

Do not create another top-level agent/runtime abstraction for an experiment. Extend `nsa.agent` or the appropriate owned module and add a regression test.

## Research rule

Benchmark results are evidence, not implementation gates. A failing scientific hypothesis should remain visible as a failing experiment while the software gate remains green.
