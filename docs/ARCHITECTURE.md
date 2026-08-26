# NSA Architecture

NSA is split into three layers:

```text
LLM backend
    ↓
NSA runtime / CCE
    ↓
Typed state + governance + tools
```

The LLM remains replaceable. NSA owns the stateful control loop: observation, belief/state update, information seeking, prediction, policy/governance checks, action selection, audit and persistence.

## Package boundary

`nsa/` is the public Python library. Research code under `experiments/` must not become a runtime dependency.

## Design goal

A user should be able to install NSA from PyPI, connect an Ollama/OpenAI-compatible backend, run a state-aware agent, inspect its state, and swap policies without importing research modules.
