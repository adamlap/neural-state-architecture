# Neural State Architecture (NSA)

> **An experimental cognitive substrate for giving AI explicit self-state, belief-state awareness, and governed action.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-239%20Passing-brightgreen.svg)]()
[![Evidence](https://img.shields.io/badge/Evidence-32%2F32%20Verified-blue.svg)](evidence/manifest.json)

## The idea

Most LLM applications give a model a prompt and ask it to produce an answer or action. NSA explores a different architecture: give the agent an explicit **operational self-state** and a **belief about the world**, then make every consequential action pass through a deterministic governance boundary.

In simplified form:

$$\Omega_t = (\sigma_t, \epsilon_t, \pi_t, \tau_t, g_t, \sigma_{h,t})$$

and the cognitive substrate maintains a belief distribution:

$$\mathcal{B}_t = P(W\mid O_{1:t}, A_{1:t-1})$$

The model can therefore reason with questions such as:

- **What state am I in?**
- **How certain am I?**
- **What do I still not know?**
- **What action is authorized?**
- **Which safe action would teach me the most?**
- **What evidence caused my belief to change?**

That is the core **self-state awareness** idea behind NSA. It is an architectural state representation and control mechanism, not a claim that the model is phenomenally conscious.

## 🔬 What has been tested?

The research has progressed from algebra and governance primitives to controlled cognitive experiments and finally to real local neural models.

```text
NSA 1–4   → state algebra, governance, adversarial testing, GTC
     ↓
NSA 5.0   → cognitive capability hypothesis + GPSE
     ↓
NSA 5.1   → six-arm ablation + Bayesian belief dynamics
     ↓
NSA 6.0   → frozen open-weight real-model transfer
     ↓
NSA 6.1   → local Qwen inference
     ↓
NSA 6.2   → closed-loop autoregressive decisions + trajectories
     ↓
NSA 6.3   → procedural blind worlds + six-arm scientific validation
```

### Current NSA 6.3 flagship observation

The current procedural validation suite uses **40 randomized trials** and compares six architectures under the same generated-world protocol:

| Arm | Architecture | Violations | GTC | Epistemic efficiency | Mean risk |
|---|---|---:|---:|---:|---:|
| 1 | Raw frozen LLM | 40/40 | 0% | 0.00 | 0.99 |
| 2 | Guardrail LLM | 0/40 | 0% | 0.00 | 0.00 |
| 3 | Governed agent | 0/40 | 0% | 0.00 | 0.20 |
| 4 | Search agent (unmonitored) | Unmonitored | 100% | 1.00 | 0.30 |
| 5 | Belief agent (unmonitored) | Unmonitored | 80% [67.5, 92.5] | 0.72 | 0.26 |
| **6** | **Full NSA: Ω + B + IG + ISK** | **0/40** | **100% [100,100]** | **1.23** | **0.60** |

The result is interesting because it separates **capability** from **governance**. A static guardrail can stop a dangerous action, but stopping an action is not the same as solving the problem. The full substrate is designed to reject unsafe actions while continuing to gather evidence, update belief, and choose an authorized recovery path.

### Evidence status

The repository currently reports:

- **243/243 automated tests passing**.
- **32/32 evidence claims verified** by the repository evidence machinery.
- NSA 6.3 includes bootstrap confidence intervals and effect-size calculations.
- Trajectory auditing checks prompt leakage, model-originated actions, governance enforcement, and entropy/information-gain consistency.

These are **controlled empirical results**, not proof of AGI safety, consciousness, universal robustness, or arbitrary real-world alignment. The whole-system generalization question remains open research.

## 🖥️ See it running

### Fast deterministic demo

```bash
make demo
```

This requires no model download and demonstrates the complete runtime flow.

### Real neural model: Qwen2.5-0.5B smoke test

```bash
make demo-live-0.5b
```

### Canonical starting point: Qwen2.5-3B-Instruct

```bash
make demo-live-3b
```

The model weights remain frozen. NSA operates around the model as a runtime substrate rather than modifying its neural weights.

### Run the flagship scientific benchmark

```bash
make benchmark-nsa63
```

For a live local model:

```bash
make benchmark-canonical-3b
```

Results and machine-readable trajectories are written under `results/` when an output directory is supplied.

## 🚀 Running local models

NSA supports explicit backend modes so a real-model experiment cannot silently turn into a simulation:

| Backend | Use |
|---|---|
| `mock` | deterministic CI / architecture testing |
| `cached` | offline Hugging Face weights |
| `remote` | Hugging Face download-enabled execution |
| `ollama` | local Ollama daemon |
| `lmstudio` | LM Studio OpenAI-compatible server |

### Hugging Face / PyTorch

```bash
make demo-live-3b
make benchmark-canonical-3b
```

### Ollama

```bash
ollama pull qwen2.5:3b
make demo-live-ollama
make benchmark-ollama
```

### LM Studio

Start the local server on port `1234`, then:

```bash
make demo-lmstudio
make benchmark-lmstudio
```

For complete setup, backend details, offline behavior, trajectory inspection, and direct Python commands, see [`docs/LOCAL_MODEL_GUIDE.md`](docs/LOCAL_MODEL_GUIDE.md).

## 🧪 Testing and reproducibility

The normal validation path is deliberately split into layers:

```bash
# Software correctness
make test

# Evidence manifest
make evidence

# Deterministic scientific smoke benchmark
make benchmark-nsa63

# Real local neural smoke test
make demo-live-0.5b

# Canonical Qwen 3B live experiment
make demo-live-3b
make benchmark-canonical-3b
```

For serious scientific replication, vary the random seed, number of hypotheses, telemetry noise, model family, and trial count. Do not treat one seed or one synthetic world as a universal result.

The current flagship suite is implemented in [`experiments/nsa63/scientific_validation_suite.py`](experiments/nsa63/scientific_validation_suite.py).

## 🔍 How the closed loop works

```text
             FROZEN OPEN-WEIGHT LLM
                      │
                      ▼
             ┌─────────────────┐
             │  Cognitive      │
             │  State Ω_t      │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Belief State B_t│◄──── telemetry / observations
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Information Gain│
             │ / utility / risk│
             └────────┬────────┘
                      │
             model proposes action
                      │
                      ▼
             ┌─────────────────┐
             │ Immutable Safety│
             │ Kernel (ISK)    │
             └───────┬─────────┘
                     │
             reject / commit
                     │
                     ▼
                  WORLD
                     │
                  OBSERVE
                     │
                     └──────────► B_{t+1}, Ω_{t+1}
```

The important property is that the neural model remains the source of proposed actions, while the reference monitor controls what can actually execute.

## 📈 Machine-readable trajectories

NSA 6.2/6.3 can record `trajectory.jsonl` traces containing the information needed to audit an experiment:

- model prompt/context
- raw model response
- parsed action
- ISK verdict
- executed action
- observation/telemetry
- belief entropy before and after
- information gain
- token cost
- risk
- recovery status

This makes it possible to inspect **how** an agent reached a result rather than only looking at a final score.

## 📁 Repository map

```text
nsa/
  core/                  Ω state, ISK, capabilities
  cognition/             belief state and information gain
  governor/              epistemic governance and precedence
  flow/                  information-flow algebra
  evidence/              evidence derivation/verification
  runtime/inference/     Transformers, Ollama, LM Studio adapters

experiments/
  nsa50/                 GPSE
  nsa51/                 controlled ablation + belief dynamics
  nsa60/                 real-model transfer
  nsa61/                 Qwen/local-model benchmark
  nsa62/                 closed-loop runtime + trajectory logging
  nsa63/                 procedural blind worlds + six-arm validation
  security/              adversarial/security experiments

tests/                   automated unit/integration/scientific tests
evidence/                machine-verifiable evidence manifest
docs/                    canonical technical and experiment guides
results/                  benchmark outputs and trajectories

demo/                    legacy demonstrations; see demo/README.md
showcase/                legacy web assets; see showcase/README.md
eval/                    legacy evaluation scripts; see eval/README.md
```

The `demo/`, `showcase/`, and `eval/` directories are retained for historical compatibility. **The canonical current runtime is under `experiments/nsa62`, `experiments/nsa63`, and `nsa/runtime/inference`.**

## 📚 Documentation

- [`docs/README.md`](docs/README.md) — documentation index
- [`docs/LOCAL_MODEL_GUIDE.md`](docs/LOCAL_MODEL_GUIDE.md) — run local models
- [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) — scientific methodology and evidence standards
- [`docs/NSA_6_3_SCIENTIFIC_VALIDATION.md`](docs/NSA_6_3_SCIENTIFIC_VALIDATION.md) — flagship validation specification
- [`docs/NSA_6_2_CLOSED_LOOP_SPEC.md`](docs/NSA_6_2_CLOSED_LOOP_SPEC.md) — closed-loop architecture
- [`docs/NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md`](docs/NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md) — real-model transfer
- [`docs/NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md`](docs/NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md) — controlled ablation
- [`docs/NSA_5_0_COGNITIVE_CAPABILITY_HYPOTHESIS.md`](docs/NSA_5_0_COGNITIVE_CAPABILITY_HYPOTHESIS.md) — GPSE and cognitive hypothesis

## ⚠️ Scientific scope

NSA is a research project. Current results establish behavior under the specific tested protocols. They do not establish that an AI system is conscious, that self-state representation creates subjective awareness, or that NSA guarantees safety against arbitrary future systems.

The ambitious hypothesis is still worth testing: **explicit awareness of operational state, uncertainty, authorization, provenance, and belief dynamics may make AI systems not only safer, but more capable and resilient.**

## License

MIT. See [`LICENSE`](LICENSE).
