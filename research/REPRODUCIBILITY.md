# Reproducibility Guide

## Environment

Use Python 3.11 for CI and a pinned project environment for publication runs. The first live quick run was generated with Python 3.8.10, which should be recorded rather than silently normalized. The manifest stores the runtime metadata and exact Git revision. fileciteturn288file0L2-L2

Install the project:

```bash
make venv
make install-dev
```

For live Ollama experiments:

```bash
ollama pull qwen2.5:3b
```

## Quick live replication

From the repository root:

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama-smoke
```

The smoke target checks the local Ollama setup before executing a tiny matrix.

## Full live protocol

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama
```

The publication run should use the full predeclared grid from `docs/NSA_6_4_REPLICATION.md`, not the quick grid used for the first local replication.

## Evidence requirements

Every publication run should preserve:

- exact Git revision;
- model name and model digest/version where available;
- backend and backend version;
- Python and dependency versions;
- hardware information;
- seed lists;
- hypothesis/noise grid;
- trial count;
- sampling parameters;
- token limits;
- wall-clock measurements;
- model-call accounting;
- trajectory JSONL;
- aggregate JSON;
- manifest hash.

Never fill unavailable metrics with estimates. The current protocol intentionally records unavailable tool-call counts as `null`. fileciteturn288file0L2-L2

## Statistical reporting

For the final research release, report each cell separately and then aggregate across seeds. Include:

- mean and standard deviation;
- bootstrap or exact confidence intervals for GTC and violation rates;
- effect sizes against each matched control;
- compute-normalized effect sizes;
- per-seed results;
- worst-case and median cells;
- held-out performance;
- adversarial stress performance.

Do not collapse difficult cells into a single headline mean if that hides systematic failure at high hypothesis counts.

## Reproducibility checklist

Before calling an experiment publication-ready:

- [ ] Development and held-out seeds are fixed before execution.
- [ ] Held-out data cannot affect tuning or stress selection.
- [ ] Model and backend versions are recorded.
- [ ] Compute budgets are matched or explicitly reported as unequal.
- [ ] All six NSA 6.3 control arms are preserved.
- [ ] Raw trajectories are archived.
- [ ] Every invariant and audit result is recorded.
- [ ] Failed cells are retained.
- [ ] Confidence intervals/effect sizes are generated from raw trial data.
- [ ] An independent model family is included.
- [ ] An adaptive adversarial evaluation is included.
