# CCE integration status

## Implemented

- Isolated `cce/` package under the parallel research prototype.
- Persistent CCE state with autonomous multi-timescale dynamics.
- Runtime-configurable dynamics coefficients.
- Asynchronous input queue and continuous tick loop.
- Live Ollama reasoning adapter using the local Ollama HTTP API.
- Live Ollama structured action-proposal adapter.
- Opaque, deployment-defined capability model; no privileged action names.
- NSA `ProductLattice` / `ProductStateVector` governance boundary.
- Independent deployment policy checks for capability, confidence, risk,
  reversibility, and human approval.
- Real actuator interface; absent actuator means HOLD, never simulated execution.
- Tests covering policy changes, autonomous state evolution, and real actuator
  dispatch.

## Deliberately not changed

- No files under `nsa/` were modified by CCE.
- No NSA internal implementation was copied into CCE.
- No model weights are modified.
- No synthetic model output is used by the live Ollama adapters.

## Next research layers

1. Replace the small explicit dynamical state with a learned latent state while
   retaining the same external contract.
2. Add real STT and sensor adapters as asynchronous event producers.
3. Add persistent episodic/semantic memory with NSA metadata ingress.
4. Route all consequential transitions through the mature NSA transition
   operator/automaton once that API is stable.
5. Build controlled experiments comparing stateless, persistent, continuous,
   and continuous+NSA systems.
