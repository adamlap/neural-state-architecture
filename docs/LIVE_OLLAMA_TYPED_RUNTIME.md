# Live Ollama + NSA Typed Runtime

NSA now has a real runtime integration path around the Ollama HTTP API.

## What is actually connected

`OllamaInferenceBackend` performs the real model inference. `NSATypedRuntime`
provides the trusted NSA state/control plane around that inference:

```text
user prompt
    |
    v
canonical NSA state (read-only model context)
    |
    v
Ollama HTTP API -> real local language model
    |
    v
model output
    |
    v
trusted runtime observation
    |
    +--> semantic observation
    +--> operational self-state
    +--> provenance record
    +--> temporal transition
    |
    v
next CanonicalTypedActivation
```

The model cannot mutate `authority_state`; it can only produce model output.
The trusted runtime owns post-generation state commits.

## Important scientific boundary

Ollama's public API does not expose transformer hidden activations. Therefore
this integration **does not claim that the canonical state is an internal
neural hidden state**. The current implementation is a genuine runtime-level
NSA wrapper around a real model.

`semantic_state` after generation is explicitly an external semantic
observation derived deterministically from the response text. It is not a
hidden-state embedding.

This distinction is required for scientifically valid claims.

## Run a real model

```bash
PYTHONPATH=. python experiments/live/ollama_nsa_chat.py --model qwen2.5:3b
```

Or run one prompt non-interactively:

```bash
PYTHONPATH=. python experiments/live/ollama_nsa_chat.py \
  --model qwen2.5:3b \
  --prompt "Explain what NSA state means in this runtime."
```

The script uses `mode="ollama"`; it does not silently fall back to the mock
backend.

## What this enables next

This is the control-plane bridge needed for the Phase 18 matched experiment:

1. baseline Ollama model;
2. same Ollama model through `NSATypedRuntime`;
3. matched prompts, model, token budget and sampling;
4. measure calibration, error detection, state tracking and task performance;
5. run multiple seeds/prompts and publish the complete paired artifacts.

The next step is to add a real benchmark rather than treating successful chat
as evidence of improved intelligence.
