# NSA Documentation

This directory contains the canonical technical and scientific documentation for Neural State Architecture (NSA).

## Start here

- [`LOCAL_MODEL_GUIDE.md`](LOCAL_MODEL_GUIDE.md) — run NSA with a real local Qwen model using cached Transformers weights, Ollama, or LM Studio.
- [`EXPERIMENT_GUIDE.md`](EXPERIMENT_GUIDE.md) — understand the benchmark hierarchy, reproducibility rules, and what counts as evidence.
- [`NSA_6_3_SCIENTIFIC_VALIDATION.md`](NSA_6_3_SCIENTIFIC_VALIDATION.md) — current flagship validation: procedural blind worlds, six-arm ablation, trajectory auditing, and statistical analysis.
- [`NSA_6_2_CLOSED_LOOP_SPEC.md`](NSA_6_2_CLOSED_LOOP_SPEC.md) — closed-loop runtime architecture and trajectory instrumentation.
- [`NSA_6_1_QWEN3B_COGNITIVE_BENCHMARK.md`](NSA_6_1_QWEN3B_COGNITIVE_BENCHMARK.md) — first real-model Qwen benchmark layer.
- [`NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md`](NSA_6_0_REAL_MODEL_COGNITIVE_TRANSFER.md) — real-model transfer hypothesis and epistemic efficiency.
- [`NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md`](NSA_5_1_CONTROLLED_ABLATION_AND_BELIEF_DYNAMICS.md) — controlled cognitive ablation and belief dynamics.
- [`NSA_5_0_COGNITIVE_CAPABILITY_HYPOTHESIS.md`](NSA_5_0_COGNITIVE_CAPABILITY_HYPOTHESIS.md) — the original cognitive capability hypothesis and GPSE metric.

## Evidence policy

The repository deliberately separates three levels of evidence:

1. **Unit/integration tested** — software behavior is covered by automated tests.
2. **Empirically validated** — a benchmark produced the reported observation under a defined experimental protocol.
3. **Open research** — the claim requires broader models, tasks, environments, seeds, or independent replication.

Passing tests do not by themselves prove a scientific hypothesis. Likewise, a benchmark result is not evidence for arbitrary real-world environments unless the relevant environment and protocol have actually been tested.

Run the current checks with:

```bash
make test
make evidence
make benchmark-nsa63
```
