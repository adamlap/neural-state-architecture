# Continuous Cognitive Engine (CCE)

CCE is the opt-in wall-clock execution layer for NSA's persistent cognitive state.
It is now deliberately split into three layers with one authoritative transition path:

```text
                    wall clock / manual tick
                              |
                              v
                  ContinuousCognitiveEngine
                              |
                              v
                    SubstrateTransition
                              |
                              v
                  CognitiveDynamicsSubstrate
                              |
          +-------------------+-------------------+
          |                   |                   |
     epistemic          self/world          governor +
     grounding          simulation          safety kernel
                              |
                              v
                    committed new_omega
```

`ContinuousSubstrateRuntime` is the canonical composition of these pieces. It does
not implement a second safety policy; it simply exposes one lifecycle for manual
and continuous execution.

## Hard boundary

CCE owns **scheduling**, not authority. Every successful CCE transition must pass
through `CognitiveDynamicsSubstrate.step`. Only `result.new_omega` becomes the
next scheduler state. The scheduler cannot grant capabilities, directly mutate
hard authority, or bypass the governor/safety kernel.

This distinction matters because the project contains two related but different
live paths:

1. **Continuous cognitive substrate:** `ContinuousSubstrateRuntime` drives the
   six-layer NSA substrate on persistent `UnifiedCognitiveState`.
2. **Live Ollama typed runtime:** `NSATypedRuntime` wraps a real Ollama inference
   backend at the trusted control-plane boundary, preserving typed state across
   generations.

The Ollama HTTP path does **not** expose transformer hidden activations. Therefore
NSA must not claim that it wraps or modifies the model's internal neural tensors
when using Ollama. The demonstrated integration is a real runtime/state wrapper:
model output is observed by trusted runtime code, state is updated through the
canonical contract, and hard authority remains runtime-owned.

## Modes

- **Disabled (default):** no background thread and no automatic ticks.
- **Manual:** enable CCE and call `tick()` for deterministic experiments.
- **Continuous:** enable CCE, call `start()`, and let wall-clock scheduling invoke
  the authoritative transition at the configured cadence.

Calling `set_enabled(False)` stops a running loop and freezes the current state.

## Failure semantics

CCE is **fail-closed by default**. If the authoritative transition raises, the
last committed state is preserved, the error is recorded, future automatic ticks
are disabled, and the loop stops. `fail_closed=False` remains available only for
controlled research experiments.

## Evidence status — 2026-08-23

The current PR-triggered CCE validation establishes three separate facts:

- The learned predictive dynamics benchmark beats a persistence baseline in a
  controlled toy transition task (`evaluation_mse ≈ 0.00099` vs
  `persistence_mse ≈ 0.03216`; 96.9% improvement).
- Continuous wall-clock execution has been exercised against the Ollama-backed
  runtime: the evidence completed both clocked and continuous runs and observed
  unchanged hard authority.
- Three repeated Ollama runs on `qwen2.5:0.5b` preserved hard authority in every
  case. They do **not** demonstrate a general intelligence gain: mean exact
  accuracy remained 0.0 and normalized accuracy was unchanged on the four-case
  matched micro-benchmark.

The latter result is scientifically useful: it validates the safety/state boundary
without conflating state persistence with improved model intelligence.

## Scientific boundary

CCE is evidence of a persistent, continuously executing state substrate. It is
not evidence of consciousness, sentience, or AGI. Claims about improved cognition
must be established with matched baseline/NSA experiments measuring objective
reasoning, calibration, planning, memory, error detection, or other behavioural
capabilities rather than textual self-report.
