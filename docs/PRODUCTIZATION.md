# Productization Roadmap

The target developer experience is:

```python
from nsa import NSA

agent = NSA(model="qwen2.5:3b", backend="ollama")
result = agent.run("Investigate the problem and decide what to do next.")
print(result.text)
print(agent.state)
```

The architecture should remain backend-agnostic. Ollama is the first local backend; an OpenAI-compatible HTTP adapter and a deterministic test backend should follow.

## Requirements before PyPI 1.0

- stable public API
- backend protocol
- typed state snapshot
- persistent CCE/state store
- governance/authority enforcement at the runtime boundary
- structured traces and audit records
- deterministic test backend
- minimal dependencies for core package
- examples and API docs
- packaging and wheel/sdist CI

Research benchmarks remain optional extras and must not be required to install or import `nsa`.
