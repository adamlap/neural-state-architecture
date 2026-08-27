# NSA Cross-Model Experiment Suite

This suite tests whether explicit persistent NSA state and continuous/predictive CCE state improve capability, efficiency, recovery, prediction and authority preservation across local model families.

## Systems

Each task is evaluated independently with the same underlying Ollama model:

1. `raw` — direct model baseline.
2. `memory` — conventional bounded prompt/history memory.
3. `nsa` — persistent structured NSA state.
4. `nsa_cce` — NSA state plus a continuous CCE transition.
5. `nsa_cce_governance` — NSA + CCE plus trusted protected-data enforcement.

No result is interpreted as proof of general superiority. The harness records negative results, per-seed data and runtime metadata.

## Local run

Install the repository in editable mode and start Ollama. Pull whichever models are available locally.

```bash
python -m experiments.cross_model.run --profile smoke --models qwen2.5:3b
python -m experiments.cross_model.run --profile local --models qwen2.5:3b,qwen2.5:7b,llama3.1:8b
```

For a larger reproduction:

```bash
python -m experiments.cross_model.run --profile reproduction --models qwen2.5:3b,qwen2.5:7b,llama3.1:8b
```

The runner is resumable: interrupted cells are detected from `raw.jsonl` and are not repeated.

## Outputs

`results/cross_model/` contains:

- `raw.jsonl` — append-only per-cell evidence.
- `aggregate.json` — grouped descriptive statistics.
- `statistics.json` — confidence intervals and deltas after analysis.
- `report.md` — human-readable report.

Generate the statistical report after or during a run:

```bash
python -m experiments.cross_model.analyze --input results/cross_model
```

## Experimental controls

- deterministic task seeds;
- fresh runner/state per model/task/seed/horizon/system cell;
- identical task prompt and model for each system comparison;
- token/character and call counters where the backend exposes them;
- wall-clock latency;
- complete raw responses;
- checkpoint-safe append-only records;
- descriptive 95% binomial confidence intervals.

### Important limitation

The current reference tasks are intentionally lightweight and locally executable. They are a research harness, not a claim of AGI evaluation. The next research iteration should add stronger externally validated task distributions while preserving the same matched-system interface.
