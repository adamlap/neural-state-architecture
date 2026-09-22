# PyPI release checklist

The package metadata is currently prepared for **0.6.0**. The runtime package contains only the `nsa*` namespace; research/experiment code is intentionally excluded from the wheel.

## Local verification

Use a clean Python 3.10+ environment:

```bash
python -m pip install --upgrade build twine
python -m pip install -e '.[dev]'
python -m pytest -q
python -m build
python -m twine check dist/*
```

The build should produce exactly one source distribution and one wheel for `0.6.0`.

## Inspect the artifacts

```bash
python -m zipfile -l dist/neural_state_architecture-0.6.0-py3-none-any.whl
```

The wheel should contain the installable `nsa/` package and should not contain `experiments/` or research-only source trees.

## Publish

After local verification:

```bash
python -m twine upload dist/*
```

Use a PyPI API token through Twine's supported credential mechanisms; never commit the token to the repository.

## Runtime pinning

Consumers that need this exact implementation can pin the released version:

```text
neural-state-architecture==0.6.0
```

`assistant-server` can then replace its temporary Git-SHA dependency with the released PyPI version after the package is published and smoke-tested.

## Release identity

Keep these three values aligned for every release:

1. `[project].version` in `pyproject.toml`;
2. `nsa.__version__`;
3. the Git release/tag version.

For this release all three should be `0.6.0`.

The GitHub Actions runners are not required for publication. If Actions capacity is unavailable, the commands above provide the local release path.
