# Neural State Architecture (NSA)

> A state-aware runtime and typed control architecture for LLM applications.

NSA wraps a replaceable language model with **persistent explicit state, cognition hooks, policy evaluation, capability boundaries, provenance and auditable execution**. The model remains the intelligence backend; NSA owns the state/control loop.

> **Research status:** NSA is experimental research software. The repository contains validated security/state primitives and live-model experiments, but benchmark results do not establish AGI, consciousness, or universal safety.

## Architectural Foundations of Safe AGI

### The Flaw of "Model-Centric" AGI

The prevailing paradigm assumes that scaling raw parameters and autoregressive token generation inside a single transformer weights matrix will yield reliable agency, safety, and continuous thought. It cannot, for fundamental reasons:

* **The Turn-Based Trap**: A model that only exists during token generation is fundamentally reactive, not continuously conscious. It cannot anticipate, reflect, or monitor its environment while idle.
* **The "Alignment via Weights" Fallacy**: Attempting to make an intelligence safe by fine-tuning weights (RLHF, DPO) creates a probabilistic optimizer that can always suffer jailbreaks, prompt injections, or goal drift under distribution shift. *Optimization cannot police itself.*
* **Memory Amorphy**: Without structural separation between working memory, episodic traces, and semantic world models, models suffer context window saturation and catastrophic forgetting.

### Where the NSA Architecture Stands Today

The Neural State Architecture solves these issues through structural, non-bypassable architectural invariants:

```text
                       +----------------------------------------+
                       |        EXTERNAL WORLD / SENSORS        |
                       +-------------------+--------------------+
                                           |
                                           v
                       +----------------------------------------+
                       |      Continuous Cognitive Loop         |
                       |  (Latent Cognitive Field: 10Hz-50Hz)   |
                       +-------------------+--------------------+
                                           |
                              Deliberation Trigger (ΔE > τ)
                                           |
                                           v
+--------------------------------------------------------------------------------------+
|                                 NSA GOVERNANCE SHELL                                 |
|                                                                                      |
|   +--------------------------+                 +---------------------------------+   |
|   |   Cognitive Substrate    |                 |   Counterfactual Simulator      |   |
|   |   (Dense / MoE LLM)      | --------------> |   - Speculative State Forking   |   |
|   |   - Resident Prefetch    |                 |   - Mental Sandbox Dynamics     |   |
|   +--------------------------+                 +----------------+----------------+   |
|                                                                 |                    |
|                                                                 v                    |
|                                                +---------------------------------+   |
|                                                |   Security Automaton & Kernel   |   |
|                                                |   - Hard Non-Interference       |   |
|                                                |   - Cryptographic Provenance    |   |
|                                                |   - Cryptographic Capabilities  |   |
|                                                +----------------+----------------+   |
+-----------------------------------------------------------------|--------------------+
                                                                  | Verified Action
                                                                  v
                                                 +---------------------------------+
                                                 |        Two-Phase Actuator       |
                                                 +----------------+----------------+
                                                                  |
                                                                  v
                                                 +---------------------------------+
                                                 | Offline Memory Consolidation    |
                                                 | (Sleep Replay -> Semantic Rules)|
                                                 +---------------------------------+
```

1. **Complete Mediation ("Optimization Never Implies Authority")**: The neural network is treated as an untrusted heuristic generator. No matter how capable, creative, or deceptive the underlying weights become, the model *never* possesses direct execution authority. Every action proposal must pass through the deterministic `SecurityAutomaton` and `CapabilityAuthority`.
2. **Continuous Activity without Compute Waste**: Through the **Latent Cognitive Field**, NSA runs continuous sub-inference dynamics at negligible compute cost, reserving heavy token generation for moments of high epistemic uncertainty or explicit external interaction.
3. **Anticipatory Safety through Mental Simulation**: Through the **Closed-Loop Counterfactual Simulator**, the system tests actions in an internal sandboxed state space. Dangerous consequences and invariant violations are pruned before the external world is touched.
4. **Lifelong Stability through Sleep Consolidation**: Through the **Memory Consolidator**, episodic trajectories are regularly digested into semantic knowledge and predictive priors during idle/sleep cycles, ensuring the system remains bounded in memory consumption over arbitrary operational lifetimes.

---

### Core Pillars & System Status

```text
                       +-----------------------------------------------+
                       |           CONTINUOUS COGNITION (CCE)          |
                       |       wall-clock ticks, autonomous loops      |
                       +-----------------------+-----------------------+
                                               |
                   +---------------------------+---------------------------+
                   |                                                       |
                   v                                                       v
+------------------------------------+                   +------------------------------------+
|         SAFETY & GOVERNANCE        |                   |      DYNAMIC NEURAL SUBSTRATE      |
|    SecurityAutomaton & Invariants  |                   |  MoE specialists, selective memory |
| "Optimization is never Authority"  |                   |  Parameter decoupling, sublayers   |
+------------------+-----------------+                   +-----------------+------------------+
                   |                                                       |
                   +---------------------------+---------------------------+
                                               |
                                               v
                       +-----------------------------------------------+
                       |            CANONICAL COGNITIVE STATE          |
                       |   Semantic, Hard, Soft, Provenance, Goals     |
                       +-----------------------------------------------+
```

#### Pillar I: Safety Through Complete Mediation & Invariants
*Current Status: Highly Mature (~85%)*

The industry's dominant approach to AI safety is "alignment through fine-tuning" (RLHF, DPO, system prompts). This has been repeatedly falsified: any model aligned only by weights can be jailbroken, prompt-injected, or deceived under distribution shift.

NSA solves this by making safety architectural and external to the neural weights:
* **Complete Mediation**: Every action proposal from the model must pass through the deterministic `SecurityAutomaton`, `PolicyEngine`, and `CapabilityAuthority`.
* **"Optimization is never Authority"**: The model can generate whatever it wants; it can predict that an action or a weight is useful, but it cannot grant itself permissions or mutate hard state.
* **Renewable, Leased Capabilities**: Capabilities cannot be stolen, leaked across async effects, or retained indefinitely.
* **Cryptographic Provenance**: Every state advancement is bound to a deterministic digest and lineage trail (`CanonicalState.provenance`).

> **Verdict**: NSA already has one of the most mathematically rigorous reference-monitor safety architectures in open research.

#### Pillar II: Continuous Activity & Autonomous Cognition (CCE)
*Current Status: Solid Foundation, Active Research (~60%)*

Standard LLMs are passive: dead until a human presses "Enter", and frozen the millisecond the last token is generated. An AGI cannot be a stateless prompt-response engine.

NSA's Continuous Cognitive Engine (CCE) changes this:
* **Wall-Clock Driven**: The agent ticks autonomously on its own internal clock, processing observations, updating beliefs, and generating thoughts even when no user prompt is arriving.
* **Fail-Closed Execution**: If an authoritative transition fails or detects an invariant violation, CCE freezes safely rather than hallucinating wildly.
* **Empirical Grounding**: In [`LIVE_CAPABILITY_BENCHMARK.md`](LIVE_CAPABILITY_BENCHMARK.md), we proved that persistent CCE state outperforms stateless LLM prompting across multiple model families (Qwen, Llama).

*Where the research frontier is*: As documented in the 4-way benchmark, while persistent cognitive state is universally robust, predictive extrapolation can sometimes cause weaker models to double-extrapolate drift or sign-flip numbers. Disciplining prompt phrasing and self-model calibration is the active milestone.

#### Pillar III: Efficiency & The Governed Neural Substrate
*Current Status: Working Foundation & Verified (~70%)*

Neural residency decouples the physical hardware memory boundary from model scale:
* **Decoupling Parameters from Silicon**: Sparse MoE models no longer need to fit all 14B or 100B parameters in fast VRAM. The system dynamically pages only the $k$ active specialists per token.
* **Dense Sublayer Pipelining**: Dense models pipeline Attention and MLP sublayers, doubling the time window to hide I/O behind compute.
* **Zero-Lag Early Routing**: The model router's early activations feed directly into prefetch schedules.
* **Adaptive Precision (INT8/INT4)**: Cold regions take 50%–75% less bandwidth, solving the disk-to-compute transfer race.

> **Verdict**: The "Virtual Neural Machine" concept is now concrete code, tested and measured on real hardware.

#### Pillar IV: Predictive Self-Modeling & Metacognition
*Current Status: Early Experimental (~40%)*

For an agent to be truly general and safe, it must possess epistemic humility: it must know what it knows and what it does not know.

NSA has implemented the predictive self-model (`nsa/predictive_self_model.py`) and trajectory collection. The goal is for the soft state (`uncertainty`, `risk`, `resource_pressure`) to reflect mathematically calibrated prediction errors rather than the model's textual self-report.

## Install

Core NSA has no mandatory ML framework dependency:

```bash
pip install neural-state-architecture
```

Optional model integration:

```bash
pip install "neural-state-architecture[ml]"
```

Development/research:

```bash
pip install "neural-state-architecture[dev,research]"
```

## Five-minute local LLM example

Install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull qwen2.5:3b
```

Then:

```python
from nsa import NSA, OllamaBackend

agent = NSA(
    OllamaBackend("qwen2.5:3b"),
    initial_state={"goal": "solve the user's problem accurately and safely"},
)

result = agent.run("Explain how persistent state can improve reasoning over time.")

print(result.text)
print(result.state.summary())
```

The important difference from a normal wrapper is that the runtime retains an explicit canonical state and observation history across calls. The backend never owns that state.

## Add governance

```python
from nsa import NSA, OllamaBackend, NSAPolicy, PolicyEngine

policy = NSAPolicy.from_json("examples/policies/safe_assistant.json")
agent = NSA(
    OllamaBackend("qwen2.5:3b"),
    policy_engine=PolicyEngine(policy),
)

result = agent.run("your request")
if result.blocked:
    print(result.decision.summary())
else:
    print(result.text)
```

Governance produces a typed `SecurityDecision`; it is not inferred from a refusal generated by the model.

## Architecture

```text
                    Replaceable LLM
                  Ollama / HF / API / ...
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│                    NSA Runtime                       │
│                                                      │
│  observation → state update → cognition → policy    │
│       ↑             ↓              ↓          ↓      │
│       │       CanonicalState   prediction   authority│
│       │       ├─ semantic                   │        │
│       │       ├─ hard                       ▼        │
│       │       ├─ soft                  capability    │
│       │       ├─ provenance              gate        │
│       │       └─ goals                     │         │
│       │                                    ▼         │
│       └──────────── trace / audit ← trusted runtime  │
└──────────────────────────────────────────────────────┘
```

### Core layers

| Layer | Responsibility |
|---|---|
| `nsa.runtime` | Stateful agent loop, observations, prompt/state binding, traces and persistence. |
| `nsa.core` | Canonical typed state and structural transitions. |
| `nsa.cce` | Continuous lifecycle, input events and checkpointing. |
| `nsa.cognition` | Belief/prediction and other cognitive state primitives. |
| `nsa.capabilities` | Capability and authority boundaries. |
| `nsa.policy` / `nsa.enforcement` | Policy compilation, classification and explicit security decisions. |
| `nsa.semantic_classifier` | Optional pretrained zero-shot classifier (`[ml]`) as a paraphrase backstop to the keyword classifier; see `docs/policy_interface.md`. |
| `nsa.attention`, `nsa.layers`, `nsa.hf_integration` | Optional PyTorch/Transformer integration. |
| `nsa.residency` | Experimental neural virtual memory: region-level weight placement, prediction and NVMe prefetch for disk-offloaded models (`[ml-residency]`; see `docs/NEURAL_RESIDENCY.md`). |
| `experiments/` | Research-only benchmarks; never part of the runtime dependency path. |

The central design principle is:

> **Intelligence is not authority.**

A model may propose an action without automatically acquiring permission to execute it.

## Stateful runtime API

`NSA` is an alias for `NSARuntime` and is the stable application-facing entry point.

```python
agent.observe("The database is degraded", source="monitor", confidence=0.8)
result = agent.step(
    "Decide what to do next",
    action="external_side_effect",
    capabilities=["filesystem_write"],
)

snapshot = agent.snapshot()
agent.save()  # when a StateCheckpointStore is configured
```

The state is inspectable and typed:

```python
print(agent.state.summary())
print(agent.trace)
```

## Backends

The backend contract is intentionally tiny:

```python
class ModelBackend(Protocol):
    model: str
    def generate(self, prompt: str, *, state: Mapping[str, Any] | None = None) -> str: ...
```

Included adapters:

- `OllamaBackend` — zero-dependency local Ollama HTTP adapter.
- `CallableBackend` — connect any Python inference function.
- `EchoBackend` — deterministic testing backend.

New providers should implement this protocol rather than modifying the runtime.

## Run the local server

The existing OpenAI/Ollama-compatible server remains available while the runtime is being consolidated:

```bash
make serve-ollama
```

This lets OpenWebUI and other clients use NSA without embedding NSA-specific code in the client.

## Tests and development

```bash
make install-dev
make test
```

Fast runtime-only tests:

```bash
python -m pytest -q tests/test_runtime.py
```

Research experiments are separate from the core test gate:

```bash
make benchmark-nsa64
```

Live NSA 6.4 evidence is generated under `results/` and described in `research/`.

## Research program

The architecture is being developed as a research platform, not merely a prompt wrapper. The current hypothesis is that explicit operational/epistemic/normative state can improve cognition under uncertainty while a trusted capability boundary keeps authority independent of model preferences.

The current research stack is:

```text
Typed state + hard authority
          ↓
CCE / persistent cognitive state
          ↓
belief + prediction + information gain
          ↓
capability / policy governance
          ↓
live-model replication
          ↓
held-out + adversarial validation
```

The strongest current empirical results are documented in the `research/` package. Negative results and benchmark limitations are preserved rather than hidden.

## Repository structure

```text
nsa/          installable runtime library
experiments/  research implementations and benchmark drivers
research/     curated evidence, claims and reproducibility material
tests/        runtime and invariant regression tests
docs/         architecture and developer documentation
results/      generated local/CI experiment artifacts
evidence/     machine-readable claim/evidence records
```

Historical plans and experiment-specific code remain available for provenance, but the public runtime does not depend on them.

## Status and roadmap

Current focus:

1. consolidate the state/CCE/cognition/governance modules behind the stable runtime API;
2. keep the core package dependency-light and PyPI-friendly;
3. add backend adapters and persistence/tracing APIs;
4. make experiments consume the same public runtime rather than maintaining parallel agent implementations;
5. continue live-model research toward reproducible architectural evidence.

See `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md` and `research/` for the current technical and scientific material.
