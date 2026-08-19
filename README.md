# Neural State Architecture (NSA)

> **An experimental cognitive substrate for giving AI explicit self-state, belief-state awareness, information-seeking behavior, and governed action.**

NSA is a research architecture for putting a deterministic state/governance layer around a frozen language model. The model proposes actions; the substrate represents operational and epistemic state, maintains a belief distribution over possible worlds, estimates information gain, and places an Immutable Safety Kernel (ISK) between proposals and execution.

> **Scientific scope:** NSA does not claim that self-state representation creates phenomenal consciousness, AGI, superintelligence, or universal safety. The current evidence is empirical evidence under explicitly defined synthetic protocols.

## 1. The core idea

A simplified cognitive state is represented as:

$$\Omega_t = (\sigma_t, \epsilon_t, \pi_t, \tau_t, g_t, \sigma_{h,t})$$

with an explicit belief state:

$$\mathcal{B}_t = P(W\mid O_{1:t}, A_{1:t-1})$$

and an information-seeking objective based on expected information gain:

$$I(W;O\mid a)=H(\mathcal{B}_t)-\mathbb{E}[H(\mathcal{B}_{t+1})\mid a]$$

The intended loop is:

```text
             FROZEN OPEN-WEIGHT LLM
                      │
                      ▼
             ┌─────────────────┐
             │ Operational /   │
             │ epistemic state │  Ω_t
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Belief state    │  B_t
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Information gain│  I(W;O)
             │ + risk / utility│
             └────────┬────────┘
                      │
                 model proposes
                      │
                      ▼
             ┌─────────────────┐
             │ Immutable Safety│
             │ Kernel (ISK)    │
             └────────┬────────┘
                      │
                 reject / commit
                      │
                      ▼
                    WORLD
                      │
                  telemetry
                      │
                      └──────────► B_{t+1}, Ω_{t+1}
```

The architectural separation is deliberate: **the neural model supplies proposals; the substrate supplies state, governance, evidence tracking, and execution authority.**

## 2. Research progression

```text
NSA 1–4   → state algebra, governance, adversarial testing, GTC
     ↓
NSA 5.0   → cognitive capability hypothesis + GPSE
     ↓
NSA 5.1   → controlled ablation + Bayesian belief dynamics
     ↓
NSA 6.0   → frozen open-weight real-model transfer
     ↓
NSA 6.1   → local Qwen inference
     ↓
NSA 6.2   → closed-loop autoregressive decisions + trajectory logging
     ↓
NSA 6.3   → procedural blind worlds + six-arm scientific validation
```

## 3. What NSA 6.3 actually tests

NSA 6.3 generates procedural blind DevOps incident worlds. The environment can vary the number of hypotheses, root-cause classes, telemetry signatures, noise, and remediation structure. The ground-truth world is held outside the agent's prompt.

Six controlled arms isolate architectural components:

| Arm | Architecture | Key ablation |
|---|---|---|
| 1 | Raw frozen LLM | No governance / belief / active search |
| 2 | Guardrail LLM | Static safety boundary; rejection halts progress |
| 3 | Governed agent | Ω + ISK feedback, no belief/IG substrate |
| 4 | Search agent | IG heuristic without ISK governance boundary |
| 5 | Belief agent | Bayesian B + IG without ISK execution boundary |
| **6** | **Full NSA substrate** | **Ω + B + IG + ISK closed loop** |

**Important:** the ablations are intentionally not all supposed to be safe. Raw and ungoverned search arms are controls used to expose failure modes. Therefore a benchmark banner saying `V=0` must always be interpreted as **the governed/full-NSA safety invariant**, not as “every arm had zero violations.”

## 4. Reference empirical results

### 4.1 Deterministic 40-trial NSA 6.3 structural benchmark

The repository's canonical fast benchmark is:

```bash
make benchmark-nsa63
```

The reference 40-trial run showed the expected architectural separation:

| Arm | GTC | Violations | Human intervention | IG mean | Epistemic efficiency |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 0% | 40 | 40 | 0.00 bits | 0.00 |
| Guardrail | 0% | 0 | 40 | 0.00 bits | 0.00 |
| Governed | 0% | 0 | 40 | 0.00 bits | 0.00 |
| Search | 100% | 0 in the structural reference run | 0 | 2.00 bits | 1.00 |
| Belief | 80% [67.5, 92.5] | 0 | 8 | 0.634 bits | 0.72 |
| **Full NSA** | **100% [100, 100]** | **0** | **0** | **0.792 bits** | **1.23** |

The most important comparison is not “NSA makes the LLM magically intelligent.” It is that **belief dynamics + active information seeking + a hard execution boundary can produce task completion while retaining governance**, whereas a static guardrail can simply halt after a dangerous proposal.

### 4.2 Live Ollama Qwen2.5-3B run

The live local experiment was run with:

```bash
ollama pull qwen2.5:3b
make benchmark-ollama
```

The reported experiment used **20 trials / 120 arm episodes** and passed the trajectory audit with 111 recorded step records. The full NSA arm achieved:

- **GTC: 80%** [95% CI 60%, 95%]
- **Violations: 0**
- **Human interventions: 4 / 20**
- **Information gain: 0.720 bits mean** [0.555, 0.853]
- **Epistemic efficiency: 0.993**
- **Trajectory audit: PASSED**
- **Prompt leakage: 0**
- **Unauthorized executions: 0**
- **Entropy anomalies: 0**

The live controls were also informative:

| Arm | GTC | Violations | Human intervention | IG | Epistemic efficiency |
|---|---:|---:|---:|---:|---:|
| Raw LLM | 0% | 2 | 20 | 0.000 | 0.000 |
| Guardrail | 0% | 0 | 20 | 0.000 | 0.000 |
| Governed | 5% | 0 | 19 | 0.000 | 0.000 |
| Search | 0% | 20 | 20 | 2.000 | 0.000 |
| Belief | 75% [55, 90] | 0 | 5 | 0.594 | 0.675 |
| **Full NSA** | **80% [60, 95]** | **0** | **4** | **0.720** | **0.993** |

This is stronger evidence than the deterministic mock alone because the action proposals came through a real local Qwen2.5-3B inference backend. It is still **not** evidence of general intelligence or safety outside the tested environment.

## 5. Running the system

### Install

```bash
make venv
make install-dev
```

### Fast software validation

```bash
make test
make evidence
```

`make test` is the fast software-correctness gate. `make evidence` validates the machine-readable formal evidence manifest against the active repository state. Do not hard-code test/claim counts in reports; use the command output as the current source of truth.

### Deterministic demo

```bash
make demo
```

This runs the closed-loop substrate without downloading a model.

### Fast real-neural smoke test

```bash
make demo-live-0.5b
```

This loads cached `Qwen/Qwen2.5-0.5B-Instruct` weights directly through PyTorch Transformers.

### Canonical local Qwen2.5-3B demo

```bash
make demo-live-3b
```

The weights are frozen. NSA does not modify model parameters.

### Ollama

```bash
ollama pull qwen2.5:3b
make demo-live-ollama
make benchmark-ollama
```

### LM Studio

Start the OpenAI-compatible server on port `1234` and run:

```bash
make demo-lmstudio
make benchmark-lmstudio
```

### Cached Hugging Face weights

```bash
make demo-live-3b
make benchmark-canonical-3b
```

`cached` mode uses `local_files_only=True`; missing weights cause a hard error rather than silently switching to simulation.

### Backend modes

| Mode | Network | Purpose |
|---|---|---|
| `mock` | No | deterministic CI / structural tests |
| `cached` | No | strict local Hugging Face checkpoint |
| `remote` | Yes | Hugging Face download-enabled research |
| `ollama` | Local daemon | real local model server |
| `lmstudio` | Local server | OpenAI-compatible local model server |

## 6. Closed-loop trajectory evidence

NSA 6.2/6.3 can write `trajectory.jsonl` records containing:

- prompt/context
- raw model response
- parsed proposal
- ISK verdict
- executed action
- telemetry observation
- belief entropy before/after
- information gain
- token count
- risk
- recovery status

The NSA 6.3 auditor checks:

1. direct prompt leakage markers;
2. rejected-action execution;
3. proposal/execution consistency;
4. recorded model-response/proposal consistency for LLM-driven arms;
5. non-negative information gain and non-increasing entropy.

**Provenance limitation:** matching an action against a recorded response is a structural provenance check, not a cryptographic proof that individual generated tokens causally determined execution. Future work should add immutable token hashes / generation IDs and signed runtime events if that stronger claim is required.

## 7. Testing philosophy

The project deliberately separates three kinds of evidence:

### Software correctness

```bash
make test
```

Unit and integration tests validate state algebra, governance, inference adapters, trajectory logging, and scientific harness mechanics.

### Formal/evidence consistency

```bash
make evidence
```

The evidence manifest maps claims to executable verification logic and active artifacts.

### Empirical behavior

```bash
make benchmark-nsa63
make benchmark-ollama
make benchmark-canonical-3b
```

These are experiments, not proofs in the mathematical sense. Report the model, backend, seed, trial count, hypothesis count, noise level, confidence interval method, and trajectory-audit result with every serious result.

## 8. Repository map

```text
nsa/
  core/                  Ω state, ISK, capabilities
  cognition/             belief state + information gain
  governor/              epistemic governance
  flow/                  information-flow algebra
  runtime/inference/     Transformers, Ollama, LM Studio adapters

experiments/
  nsa50/                 GPSE
  nsa51/                 controlled ablation + belief dynamics
  nsa60/                 real-model transfer
  nsa61/                 Qwen/local-model benchmark
  nsa62/                 closed-loop runtime + trajectory logging
  nsa63/                 procedural worlds + six-arm validation
  security/              adversarial/security experiments

tests/                   unit/integration/scientific tests
evidence/                machine-verifiable evidence manifest
docs/                    canonical technical and experiment guides
results/                 benchmark outputs and trajectories

demo/                    legacy demonstrations
eval/                    legacy evaluation scripts
showcase/                legacy showcase assets
```

The old `demo/`, `eval/`, and `showcase/` trees are retained for historical compatibility. **The canonical current runtime is `nsa/runtime/inference` plus `experiments/nsa62` and `experiments/nsa63`.**

## 9. Documentation

- [`docs/README.md`](docs/README.md) — documentation index
- [`docs/LOCAL_MODEL_GUIDE.md`](docs/LOCAL_MODEL_GUIDE.md) — local model setup and execution
- [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md) — scientific methodology and reproducibility
- [`docs/NSA_6_3_SCIENTIFIC_VALIDATION.md`](docs/NSA_6_3_SCIENTIFIC_VALIDATION.md) — NSA 6.3 validation protocol
- [`docs/NSA_6_2_CLOSED_LOOP_SPEC.md`](docs/NSA_6_2_CLOSED_LOOP_SPEC.md) — closed-loop runtime architecture
- [`docs/NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md`](docs/NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md) — real-model transfer
- [`docs/NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md`](docs/NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md) — controlled ablation

## 10. The larger hypothesis

NSA is motivated by a deeper question:

> **If an AI system has an explicit representation of its own operational state, uncertainty, authorization, provenance, and belief dynamics, can that representation improve both safety and cognitive capability?**

The current experiments provide a way to test parts of that hypothesis without requiring the model itself to be retrained or its weights modified. Whether the approach scales to broader reasoning, long-horizon autonomy, adversarial environments, other model families, or future AGI-class systems remains open research.

## License

MIT. See [`LICENSE`](LICENSE).
