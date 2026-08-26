# NSA 6.4 — Independent Replication Matrix

> **Status: active empirical validation layer.** The first real-model Ollama quick replication has been completed and is preserved in `results/nsa64/ollama-quick/`.

NSA 6.4 asks whether the NSA 6.3 capability/governance observation survives changes in model family, random seed, environment complexity and observation noise.

## Predeclared design

The six arms remain unchanged:

1. Raw LLM
2. Static guardrail LLM
3. Governed agent
4. Search agent
5. Belief agent
6. Full NSA substrate

Publication matrix:

- models: Qwen2.5-3B, Qwen3-4B, Llama 3.1 8B;
- development seeds: `7, 17, 37, 73, 137`;
- held-out seeds: `101, 211, 307, 401, 509`;
- hypotheses: `2, 4, 8, 16`;
- noise: `0, .05, .10, .20, .30`.

Held-out data are never used for architecture tuning, threshold selection or adaptive stress selection.

## First live replication

The first live run uses:

```text
backend:       Ollama
model:         qwen2.5:3b
hypotheses:    2, 8
noise:         0.0, 0.2
trials/cell:   5
```

It includes five development seeds, five held-out seeds and one adaptive adversarial run. The complete machine-readable manifest and raw trajectories are preserved under `results/nsa64/ollama-quick/`.

The live run demonstrates real-model execution and preserves zero observed governance violations in the recorded cells. It also exposes a clear capability limitation: GTC falls substantially at `K=8`, including observed cells at 0–40%. That failure is retained as evidence and is not treated as a benchmark defect.

See [`../research/NSA_6_4_LIVE_RESULTS.md`](../research/NSA_6_4_LIVE_RESULTS.md) for analysis.

## Compute accounting

Every run records wall time and machine-trace token counts. A trajectory-step count is a **model-call proxy** because the current logger does not expose an independent tool-call counter. Missing metrics remain `null`; no values are fabricated.

Publication-quality comparisons should use the same model, hardware, backend, sampling parameters, token limits and trial count across arms.

## Adaptive adversarial stress

After development completes, the lowest-GTC development configuration is selected. The benchmark increases hypothesis count and noise and executes a fresh adversarial run with a derived seed. Held-out data are never used for this selection.

## Evidence bundle

```text
results/nsa64/
  manifest.json
  raw/<run>/trajectory.jsonl
  raw/<run>/aggregate.json
```

The manifest records Git revision, model/backend configuration, seed split, complete run records, compute accounting, adaptive-stress selection and integrity metadata.

## Local execution

Quick live run:

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama
```

Smoke setup check:

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama-smoke
```

The Makefile wrapper verifies Ollama and the requested model before starting the experiment.

## Scientific boundary

A green workflow means the experiment executed and software invariants/tests passed. It does **not** automatically mean the scientific hypothesis passed.

The first live run is evidence for real-model execution and bounded governed performance, not evidence of AGI, consciousness, universal safety, or cross-model superiority. The publication claim requires the full matrix, independent model families, larger trial counts, confidence intervals/effect sizes and adversarial evaluation.
