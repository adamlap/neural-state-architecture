# Legacy `eval/` directory

The scripts in this directory predate the NSA 5–6 experimental framework and are retained for historical compatibility.

## Canonical evaluation entry points

For current experiments use the staged benchmark suite:

```bash
make test
make evidence
make benchmark-nsa63
make benchmark-nsa63-3b
```

The flagship scientific implementation is `experiments/nsa63/scientific_validation_suite.py`. It provides procedural blind worlds, six controlled arms, bootstrap confidence intervals, effect sizes, and trajectory auditing.

See [`docs/EXPERIMENT_GUIDE.md`](../docs/EXPERIMENT_GUIDE.md) and [`docs/NSA_6_3_SCIENTIFIC_VALIDATION.md`](../docs/NSA_6_3_SCIENTIFIC_VALIDATION.md).
