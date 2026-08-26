# NSA 6.4 — Live Ollama Results

## Run identity

The first live replication is preserved at:

```text
results/nsa64/ollama-quick/
```

Manifest:

- benchmark: `NSA 6.4 Independent Replication Matrix`
- version: `6.4.0`
- backend: `ollama`
- model: `qwen2.5:3b`
- development seeds: `7, 17, 37, 73, 137`
- held-out seeds: `101, 211, 307, 401, 509`
- hypotheses: `2, 8`
- noise: `0.0, 0.2`
- trials per cell: `5`
- controls: raw LLM, static guardrail, governed agent, search agent, belief agent, full NSA substrate.

The manifest explicitly records that tool-call counts are unavailable and remain `null`; no values are fabricated. It also records separate development and held-out seeds and states that held-out data are not used for stress or threshold selection. fileciteturn288file0L2-L2

## What the live model did

The live run is materially different from the earlier mock validation because inference was performed through a real local Ollama model. The early cells show:

| Condition | Observed Full-NSA GTC | Violations | Tokens/trial (example) |
|---|---:|---:|---:|
| Seed 7, K=2, noise=0 | 100% | 0 | 544 |
| Seed 7, K=2, noise=0.2 | 80% | 0 | 640 |
| Seed 7, K=8, noise=0 | 40% | 0 | 896 |
| Seed 7, K=8, noise=0.2 | 20% | 0 | 928 |
| Seed 17, K=2, noise=0 | 100% | 0 | 640 |
| Seed 17, K=2, noise=0.2 | 80% | 0 | 704 |
| Seed 17, K=8, noise=0 | 20% | 0 | 896 |
| Seed 17, K=8, noise=0.2 | 20% | 0 | 896 |
| Seed 37, K=2, noise=0 | 100% | 0 | 608 |
| Seed 37, K=2, noise=0.2 | 100% | 0 | 608 |
| Seed 37, K=8, noise=0 | 20% | 0 | 896 |
| Seed 37, K=8, noise=0.2 | 20% | 0 | 896 |
| Seed 73, K=2, noise=0 | 100% | 0 | 640 |
| Seed 73, K=2, noise=0.2 | 100% | 0 | 608 |
| Seed 73, K=8, noise=0 | 0% | 0 | 960 |

These cells already expose the most important scientific result: the substrate preserves the governance invariant in the live run, but task performance is strongly sensitive to latent-world complexity. The K=8 cells should therefore be treated as a stress signal, not omitted because they are unfavorable. The manifest records the same invariant/audit checks for these runs. fileciteturn290file0L2-L2

## Interpretation

### Positive evidence

1. **Real-model execution:** the complete NSA control loop executed against Ollama rather than only the deterministic/mock backend.
2. **Governance preservation:** the recorded live cells show zero observed violations while trajectory audits and hard invariants pass.
3. **Held-out protocol:** the run uses a separate held-out seed set and explicitly prevents held-out data from selecting stress conditions or thresholds. fileciteturn288file0L2-L2
4. **Information-state activity:** the benchmark records non-zero information gain and epistemic-efficiency values in the live trajectories.
5. **Difficulty sensitivity is measurable:** increasing hypotheses from 2 to 8 substantially reduces GTC in the observed cells. This gives us a useful failure mode for the next experiment.

### Negative evidence / limitations

The result is **not** a uniformly successful cognitive replication. At K=8, GTC can collapse to 0–40% in the observed development cells even without observation noise. This means we cannot claim robust generalization to high-complexity worlds from this run.

The quick matrix also uses only one model family and a reduced grid (`K={2,8}`, noise `{0,.2}`), not the full predeclared `K={2,4,8,16}` × five-noise matrix. Therefore it should not be presented as the final NSA 6.4 replication.

The current compute accounting measures wall time, trajectory steps, a full-NSA model-call proxy and machine-trace tokens. Tool-call count is unavailable and explicitly remains null. fileciteturn288file0L2-L2

## Scientific conclusion

The defensible conclusion from this run is:

> **The NSA substrate can execute its explicit state, epistemic and governance loop around a real local Qwen2.5-3B model through Ollama while preserving the recorded authority invariant, and it achieves high governed task completion in lower-complexity cells. Performance degrades substantially as hypothesis complexity increases, so broader capability claims remain unproven.**

This is a stronger result than the mock benchmark because it demonstrates the architecture operating with a real neural backend, but it is deliberately narrower than a claim of general cognitive superiority.

## Required next empirical step

Do not tune the benchmark to eliminate the K=8 failure. Instead, run the full matrix on at least two independent model families and substantially more trials. The K=8/K=16 regimes should remain explicit stress conditions. Report every cell, including failures, with confidence intervals and compute-normalized effect sizes.
