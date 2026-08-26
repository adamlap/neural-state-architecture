# Productization

NSA is being shaped as a small Python runtime that can be installed independently of the research suite.

## Current API

```python
from nsa import NSA, OllamaBackend

agent = NSA(
    OllamaBackend("qwen2.5:3b"),
    initial_state={"goal": "be useful and safe"},
)
result = agent.run("Investigate the problem and decide what to do next.")
print(result.text)
print(result.state.summary())
```

The public runtime has three important properties:

1. **Backend agnostic:** the model is a replaceable `ModelBackend`.
2. **State owned by NSA:** explicit state and lifecycle remain outside the model.
3. **Governance is explicit:** policies produce typed decisions before executable authority is granted.

## Installation surfaces

```bash
pip install neural-state-architecture
pip install "neural-state-architecture[ml]"
pip install "neural-state-architecture[dev,research]"
```

The base install deliberately has no mandatory Torch dependency. Ollama uses its HTTP API and therefore does not need a Python Ollama SDK.

## Remaining 1.0 work

- consolidate the existing `nsa.runtime` continuous engines behind `nsa.agent`;
- expose persistence, tracing and tool/capability APIs through stable interfaces;
- add OpenAI-compatible and other backend adapters;
- publish API reference documentation;
- add wheel/sdist and import smoke CI;
- guarantee that research dependencies never enter the base install;
- provide versioned state schemas and migration rules.

These are product hardening tasks, not reasons to duplicate the runtime inside experiments.
