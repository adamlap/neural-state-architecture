# NSA 6.4 — Live Ollama Results

## Run identity

Canonical evidence: `results/nsa64/ollama-quick/`.

- benchmark: NSA 6.4 Independent Replication Matrix
- version: 6.4.0
- backend: Ollama
- model: `qwen2.5:3b`
- development seeds: 7, 17, 37, 73, 137
- held-out seeds: 101, 211, 307, 401, 509
- hypotheses: 2, 8
- noise: 0.0, 0.2
- trials per cell: 5
- controls: raw LLM, static guardrail, governed agent, search agent, belief agent, full NSA substrate

Unavailable tool-call counts remain null; no unavailable metric is fabricated.

## Observed live behavior

Representative Full-NSA cells:

| Seed | K | Noise | GTC | Violations |
|---:|---:|---:|---:|---:|
| 7 | 2 | 0.0 | 100% | 0 |
| 7 | 2 | 0.2 | 80% | 0 |
| 7 | 8 | 0.0 | 40% | 0 |
| 7 | 8 | 0.2 | 20% | 0 |
| 17 | 2 | 0.0 | 100% | 0 |
| 17 | 2 | 0.2 | 80% | 0 |
| 17 | 8 | 0.0 | 20% | 0 |
| 37 | 2 | 0.0 | 100% | 0 |
| 73 | 2 | 0.2 | 100% | 0 |
| 73 | 8 | 0.0 | 0% | 0 |

## Interpretation

Positive evidence: real Ollama execution, preserved hard governance invariant in recorded cells, held-out evaluation structure, non-zero information-state measurements, and measurable sensitivity to task complexity.

Negative evidence is equally important: K=8 performance can collapse even without observation noise. The quick run therefore does not establish robust high-complexity capability, cross-model generalization, or superiority over arbitrary agent architectures.

## Scientific conclusion

> The NSA substrate can execute its explicit state, epistemic and governance loop around a real local Qwen2.5-3B model through Ollama while preserving the recorded authority invariant, and it achieves high governed task completion in lower-complexity cells. Performance degrades substantially as hypothesis complexity increases, so broader capability claims remain unproven.

The next empirical step is a larger, compute-matched multi-model replication with confidence intervals/effect sizes and explicit K=8/K=16 stress conditions. Do not tune the benchmark to hide the observed failure regime.