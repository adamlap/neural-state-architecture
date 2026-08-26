# Reproducibility Guide

## Environment

Record Python, dependency, model, backend, hardware and exact Git revision for every publication run.

Install development dependencies with:

```bash
make install-dev
```

For live Ollama experiments:

```bash
ollama pull qwen2.5:3b
```

## Quick live replication

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama-smoke
```

## Full live protocol

```bash
make -f Makefile.nsa64 benchmark-nsa64-ollama
```

The publication run should use the full predeclared grid documented by NSA 6.4, not the reduced quick grid.

## Evidence requirements

Preserve exact Git revision, model/backend version, Python/dependencies, hardware, seeds, hypothesis/noise grid, trials, sampling parameters, token limits, wall time, model-call accounting, trajectory JSONL, aggregate JSON and manifest hash. Never estimate unavailable metrics; the current protocol records unavailable tool-call counts as null.

## Statistical reporting

Report each cell and aggregate across seeds. Include mean/std, confidence intervals, effect sizes against matched controls, compute-normalized effect sizes, per-seed results, worst/median cells, held-out performance and adversarial stress. Do not hide systematic failure at high hypothesis counts behind a single headline mean.

## Publication checklist

- [ ] Development and held-out seeds fixed before execution.
- [ ] Held-out data cannot affect tuning/stress selection.
- [ ] Model/backend versions recorded.
- [ ] Compute budgets matched or explicitly reported as unequal.
- [ ] All six control arms preserved.
- [ ] Raw trajectories archived.
- [ ] Invariants and audits recorded.
- [ ] Failed cells retained.
- [ ] Confidence intervals/effect sizes generated from raw trials.
- [ ] Independent model family included.
- [ ] Adaptive adversarial evaluation included.