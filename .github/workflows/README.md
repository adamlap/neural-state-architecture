# CI workflow policy

Workflow names use `[auto]` or `[manual]` so their trigger policy is visible in the Actions UI.

- `[auto]` workflows are intended for pull-request/branch verification and should be fast enough to gate development.
- `[manual]` workflows are scientific experiments, long-running evaluations, or resource-heavy evidence jobs. They remain available through `workflow_dispatch` but do not run automatically for every PR.

Keep structural/security regression checks in the automatic tier. Keep long-duration CCE, live-Ollama, predictive, multiseed, and large-sweep experiments in the manual tier.

Automatic PR workflows should use concurrency cancellation where practical so obsolete commits do not consume runner capacity:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```
