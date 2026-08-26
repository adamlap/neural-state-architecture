# Experiments

Experiments are **thin research consumers of `nsa`**, not alternate runtimes.

## Rules

1. Import reusable architecture from `nsa`.
2. Keep benchmark environments, prompts, metrics and statistical analysis here.
3. Never duplicate the state/control loop inside an experiment.
4. Write raw evidence to `results/` and curated conclusions to `research/`.
5. Keep scientific gates independent from software CI gates.

## Current suites

- `nsa63/` — controlled cognitive/governance validation.
- `nsa64/` — independent replication, held-out and adversarial validation.
- `cognitive/` — earlier CCE hypothesis tests retained for provenance.
- `breakthrough/` — exploratory architectural experiments retained for provenance.
- `nsa50/`–`nsa62/` — historical validation stages.

The active development path is to make new experiments configure and exercise the same public `nsa.NSA` runtime used by applications.
