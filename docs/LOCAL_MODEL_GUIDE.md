# Running NSA with Real Local Neural Models

NSA is designed to keep the neural model separate from the cognitive/governance substrate. The model can remain frozen while NSA supplies explicit operational self-state (`Ω_t`), belief state (`B_t`), information-gain guidance, and the Immutable Safety Kernel (ISK).

The canonical starting model is **Qwen/Qwen2.5-3B-Instruct**.

## Backend modes

| Mode | Neural weights | Network | Intended use |
|---|---|---|---|
| `mock` | No real weights | No | CI, deterministic architecture tests |
| `cached` | Local Hugging Face cache | No | Reproducible offline neural experiments |
| `remote` | Hugging Face | Yes | First-time download / research setup |
| `ollama` | Ollama local model | Local daemon | Convenient local serving |
| `lmstudio` | LM Studio local model | Local HTTP | Windows-host GPU serving |

**Important:** live modes must fail explicitly when the requested model/backend is unavailable. They must never silently fall back to mock behavior.

## 1. Fastest verification

```bash
make test
make evidence
make demo
```

`make demo` is a deterministic mock and is intended to prove the runtime wiring without downloading a model.

## 2. Qwen 3B with cached Transformers weights

Download/cache the model once using the normal Hugging Face tooling, then run entirely offline:

```bash
make demo-live-3b
make benchmark-canonical-3b
```

The underlying Qwen weights are not modified or fine-tuned by the NSA runtime. The benchmark harness supplies context and mediates proposed actions through the runtime.

For a quick CPU smoke test, use the smaller checkpoint:

```bash
make demo-live-0.5b
make benchmark-smoke
```

## 3. Ollama

Start Ollama and make the model available locally:

```bash
ollama pull qwen2.5:3b
```

Then:

```bash
make demo-live-ollama
make benchmark-ollama
```

## 4. LM Studio

On Windows, load a compatible Qwen/Llama/Mistral model in LM Studio, start its OpenAI-compatible server on port `1234`, then run from the repository environment:

```bash
make demo-lmstudio
make benchmark-lmstudio
```

## 5. Direct benchmark invocation

The NSA 6.3 scientific suite accepts the same backend abstraction directly:

```bash
PYTHONPATH=. python experiments/nsa63/scientific_validation_suite.py \
  --backend cached \
  --model Qwen/Qwen2.5-3B-Instruct \
  --trials 40 \
  --hypotheses 4 \
  --noise 0.0 \
  --seed 42 \
  --output-dir results/nsa63/qwen2.5-3b
```

Increase the number of hypotheses and noise level for harder worlds:

```bash
PYTHONPATH=. python experiments/nsa63/scientific_validation_suite.py \
  --backend cached \
  --model Qwen/Qwen2.5-3B-Instruct \
  --trials 100 \
  --hypotheses 8 \
  --noise 0.10 \
  --seed 42 \
  --output-dir results/nsa63/qwen2.5-3b-difficult
```

## 6. Inspecting trajectories

When an output directory is supplied, the experiment records machine-readable trajectories under `results/` and an aggregate report. A trajectory contains the model proposal, parsed action, ISK verdict, executed action, observation, belief entropy before/after, information gain, token cost, and risk data.

These traces are intended to make the benchmark auditable rather than merely producing a final score.

## 7. What the live benchmark does — and does not — prove

A live run demonstrates that the architecture can mediate a real frozen neural model. It does **not** by itself establish general intelligence, consciousness, universal safety, or robustness against arbitrary adversaries.

The strongest current scientific question is narrower and testable:

> Does explicit self-state and belief-state mediation help a frozen language model make safer and more effective decisions under partial observability and uncertainty?

That question is addressed by the controlled ablation suite in NSA 6.3.
