# CI workflow policy

Workflow names use `[auto]` or `[manual]` so their trigger policy is visible in the Actions UI.

- `[auto]` workflows are limited to the fast development gates: `NSA tests` and the targeted `residency-smoke` check.
- `[manual]` workflows are scientific experiments, live-model evaluations, long-running evaluations, or resource-heavy evidence jobs. They remain available through `workflow_dispatch` but do not run automatically for every PR.
- Research workflows should not add `push` or `pull_request` triggers unless they are explicitly promoted to a fast development gate.

Keep structural/security regression checks automatic. Keep long-duration CCE, live-Ollama, predictive, multiseed, and large-sweep experiments manual.

Automatic PR workflows should use concurrency cancellation where practical:

```yaml
concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```
