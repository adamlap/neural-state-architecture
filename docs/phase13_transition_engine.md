# Phase 13 — Algebra-Preserving Transition Engine

This phase turns the Phase 12 heterogeneous algebra into an executable state-transition boundary.

## What is now implemented

`nsa.transitions.TransitionEngine.apply_heterogeneous()` accepts:

- a typed source state;
- a model/runtime candidate state;
- a `TransitionCone` describing legal per-coordinate motion.

The engine then either:

1. accepts the candidate unchanged when it is legal;
2. exactly projects it onto the legal cone; or
3. rejects it when projection is disabled.

No scalar safety score is introduced. Each coordinate continues to use its own
join/meet semantics.

## Security boundary

The transition engine is authoritative for the typed state it owns, but it does
not constrain transformer weights or hidden activations. The live Ollama wrapper
therefore remains correctly described as a **runtime NSA governance wrapper**.
Claims of intrinsic neural enforcement require a future native/retrofit adapter
that consumes these transition semantics inside the neural computation path.

## Why this matters

The architecture now has a clean separation:

```text
model/runtime proposal
        |
        v
heterogeneous typed state
        |
        v
TransitionCone + TransitionEngine
        |
   +----+----+
   |         |
 legal    illegal
   |         |
   v         v
commit    projection/reject
```

This makes state invariants executable rather than merely documented. The next
research step is to connect the transition engine to the native TNC and retrofit
paths and measure the capability/quality cost of enforcement against matched
unconstrained baselines.
