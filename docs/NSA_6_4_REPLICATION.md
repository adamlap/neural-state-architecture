# NSA 6.4 — Independent Replication Matrix

NSA 6.4 is the next empirical layer after the NSA 6.3 six-arm procedural blind-world validation. It is designed to answer whether the observed full-substrate capability/governance result survives changes in model family, random seed, environment complexity and observation noise.

## Predeclared design

The six arms remain unchanged:

1. Raw LLM
2. Static guardrail LLM
3. Governed agent
4. Search agent
5. Belief agent
6. Full NSA substrate

The development matrix varies:

- models: Qwen2.5-3B, Qwen3-4B, Llama 3.1 8B (backend permitting);
- seeds: `7, 17, 37, 73, 137`;
- hypotheses: `2, 4, 8, 16`;
- noise: `0, .05, .10, .20, .30`.

Held-out evaluation uses independent seeds `101, 211, 307, 401, 509` and the same predeclared grid. Held-out observations are never used to select thresholds or tune the architecture.

## Compute accounting

Every run records wall time and machine-trace token counts. A trajectory-step count is reported as a **model-call proxy** because the current NSA 6.3 logger records one model interaction per trajectory step but does not expose an independent tool-call counter. Missing metrics are explicitly null rather than fabricated.

For publication-quality comparisons, the benchmark should be run with the same model, hardware, backend, token limits and number of trials across arms.

## Adaptive adversarial stress

After the development matrix completes, the lowest-GTC development configuration is selected. The benchmark then increases both hypothesis count and noise and executes a fresh adversarial run with a derived seed. The held-out set is never used for this selection. This is a stress test, not a post-hoc pass/fail threshold.

## Evidence bundle

Each run preserves:

```text
results/nsa64/
  manifest.json
  raw/<run>/trajectory.jsonl
  raw/<run>/aggregate.json
```

`manifest.json` contains the git revision, model/backend configuration, seed split, complete run records, compute accounting, adaptive-stress selection and a SHA-256 integrity hash.

## Run locally

Smoke run:

```bash
PYTHONPATH=. python experiments/nsa64/replication_matrix.py \
  --backend mock \
  --models Qwen/Qwen2.5-3B-Instruct \
  --dev-seeds 7 17 \
  --heldout-seeds 101 211 \
  --hypotheses 2 4 \
  --noise 0.0 0.2 \
  --trials 2 \
  --out results/nsa64
```

Full predeclared matrix:

```bash
PYTHONPATH=. python experiments/nsa64/replication_matrix.py \
  --backend mock \
  --trials 20 \
  --out results/nsa64
```

For live-model replication, the same protocol can be run with `--backend ollama` after the selected models are available in Ollama.

## Scientific boundary

A green workflow means the experiment executed and its invariants/tests passed. It does **not** automatically mean the scientific hypothesis passed. NSA 6.4 is intended to produce evidence about replication, robustness and compute efficiency; it makes no claim of AGI, consciousness or general superiority by itself.

The subsequent research package should report both positive and negative cells, held-out results, confidence intervals/effect sizes, model metadata, raw trajectories and the exact git revision used to generate the evidence.
