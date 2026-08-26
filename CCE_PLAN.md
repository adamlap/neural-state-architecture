# CCE status

CCE is now a first-class subsystem of NSA rather than a separate project plan.

The implementation is split between:

- `nsa.cce` — lifecycle, input events and checkpoint primitives;
- `nsa.cognition` — belief and cognitive-state primitives;
- `nsa.runtime` — existing continuous/predictive runtime implementations;
- `nsa.agent` — stable application-facing integration boundary.

The CCE research history remains in `experiments/` and `research/`. New CCE capabilities should be implemented behind the public runtime rather than through a new experiment-specific engine.

See [`PLAN.md`](PLAN.md) for the unified roadmap and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the current architecture.

CCE remains an experimental computational architecture. Persistent or continuous machine state is not, by itself, evidence of consciousness.
