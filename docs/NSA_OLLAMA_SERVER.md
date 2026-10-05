# NSA as a local Ollama-compatible cognitive server

NSA can run as a local model server that Open WebUI can connect to directly.

## Default architecture

- System 1: Qwen2.5-1.5B-Instruct, frozen, generation-free and continuously ticking.
- System 2: Qwen2.5-3B-Instruct, used for normal natural-language generation.
- Canonical NSA state: maintained independently of token generation.
- CCE: continues wall-clock state dynamics between user turns.
- Selective memory: System 1 chooses discard, working, episodic or durable retention.
- Governance: the existing NSA reference monitor, provenance and CCE server remain in the request path.
- HTTP: native Ollama-shaped routes and OpenAI-compatible routes.

The term "conscious" here means persistent, continuously updated cognitive state and control. It is not a scientific claim about subjective consciousness.

## Install and run

Install the server and ML dependencies:

    pip install -e ".[server]"

Run the server:

    python -m nsa.server.proxy --backend transformers --model Qwen/Qwen2.5-3B-Instruct --port 11434

The first startup downloads the 1.5B System 1 checkpoint. The 3B checkpoint is
downloaded when the first System 2 response is required, or immediately when
the model is explicitly pulled.

Transformers reuses the normal Hugging Face cache, so restarting the server
does not download the weights again.

## Open WebUI

Open WebUI supports the Ollama protocol and expects the Ollama API on port
11434. Point its Ollama connection at:

    http://localhost:11434

When Open WebUI runs in Docker, use the service address from the compose file:

    http://nsa:11434

The server exposes the following model identities:

- nsa:latest
- nsa-system1:1.5b
- nsa-system2:3b
- nsa-Qwen/Qwen2.5-3B-Instruct

The recommended model is nsa:latest. It means "use the full NSA control
plane": System 1 remains active between turns and can route generation to the
small or large local model.

## Model acquisition

Ollama-shaped model management is provided:

- GET /api/tags
- GET /api/show
- POST /api/pull

For the local Transformers backend, pulling a model causes its Hugging Face
checkpoint to be downloaded into the configured HF cache. The aliases are
mapped to the repository's existing Qwen model registry.

## Continuous System 1

System 1 is not implemented as a stream of generated thoughts. It runs a
generation-free decision heartbeat.

At each heartbeat it can update:

- salience
- uncertainty
- risk
- escalation
- model routing
- memory retention

Its decisions are derived from the frozen local model's logits and pass
through the typed NSA decision boundary.

Inspect the state with:

    GET /health

The response includes the current System 1 decision set and canonical state.

The CCE-specific state remains available through:

    GET /api/cce/state

An external event can enter the same continuous substrate through:

    POST /api/cce/sensor

## Selective memory

Every chat turn is offered to System 1 for retention policy selection.

Discard means it is not stored.
Working, episodic and durable are represented as typed MemoryItem records.

The selected memories are inserted into the next generation context rather
than blindly replaying the entire conversation.

This makes the server useful as a first end-to-end test of the selective-memory
architecture: prompt the system with facts, wait across multiple turns, and
inspect the memory policy and retained item count in /health.

The current server memory store is process-local. The next persistence step can
replace it with the existing durable SNM store without changing the System 1
decision contract.

## Hardware guidance

The default 1.5B plus 3B arrangement is intentionally modest.

The Qwen2.5-1.5B-Instruct checkpoint is about 3.1 GB in its repository, while
the 3B family is around 3B parameters. Quantized variants can be substituted
later when the machine is RAM/VRAM constrained.

If the machine cannot keep both models resident, set:

    NSA_SYSTEM_ONE_DEVICE=cpu
    NSA_MODEL_DEVICE=cpu

or configure System 2 to a smaller local model. System 1 can remain the small
always-on model while System 2 is loaded less frequently.

## Docker

The repository includes docker-compose.nsa.yml. Start it with:

    docker compose -f docker-compose.nsa.yml up --build

It runs NSA on port 11434 and Open WebUI on port 3000. The Hugging Face cache is
stored in the nsa-model-cache volume.

Open WebUI can therefore be used as the normal browser interface while NSA owns
the continuous state and governance layer underneath it.
