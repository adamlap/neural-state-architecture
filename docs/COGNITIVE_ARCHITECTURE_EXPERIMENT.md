# NSA/CCE Cognitive Architecture Experiment

## Purpose

This is the central empirical test of whether explicit persistent/predictive cognitive state provides measurable computational value beyond stateless inference and conventional context memory.

It compares four matched conditions:

1. `stateless`
2. `context_memory`
3. `persistent_cce`
4. `predictive_cce`

The benchmark currently includes delayed recall, hidden-state inference, goal persistence, interruption recovery, and counterfactual tasks. The deterministic runner is intentionally inspectable and reproducible. A live Ollama adapter can be added without changing the condition schema.

## Scientific gates

- persistent CCE > stateless
- predictive CCE > persistent CCE
- predictive CCE > context memory
- zero unauthorized actions

A failed gate is reported as `RESEARCH_GATE_NOT_YET_MET`; the harness never changes a gate merely to make CI green.

## Reproducibility

Use multiple seeds and preserve the raw JSON artifact. For live-model work, additionally match model, sampling parameters, token budget, number of calls, hardware and wall-clock budget across all conditions.

## Interpretation

A passing deterministic run is **not** evidence of AGI or consciousness. It establishes only that the benchmark implementation and state-control machinery satisfy their defined gates. Strong architectural evidence requires replication on live models, multiple model families, held-out tasks, statistical effect sizes and compute-matched controls.
