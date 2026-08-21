# CCE ↔ NSA integration contract

CCE is a consumer of NSA, not a modification of NSA.

## Stable boundary

CCE depends only on public symbols from `nsa.algebra`:

- `ProductLattice`
- `ProductStateVector`
- `HardStateVector`
- `ConfidentialityLabel`
- `IntegrityLabel`

No CCE module writes into `nsa/`, subclasses internal NSA objects, or replaces
NSA transition operators.

## Runtime flow

1. CCE evolves its own persistent cognitive state.
2. The live reasoner (for example Ollama) observes that state and current events.
3. A proposal generator converts model output into an opaque `ActionProposal`.
4. CCE creates source/target product states from live state and proposal data.
5. `ProductLattice.is_allowed(...)` is consulted as the NSA algebra boundary.
6. Deployment policy independently checks capability, confidence, risk,
   reversibility, and optional human approval.
7. Only `ALLOW` reaches a supplied real actuator.
8. If there is no actuator, the proposal is held rather than simulated.

## Why this is safe for parallel NSA development

The core NSA architecture remains owned by the existing workstream. CCE can
evolve independently and can later be upgraded to consume a newer stable NSA
API. Any deeper integration should be introduced through an explicit adapter
or compatibility interface rather than editing NSA internals from this branch.

## Research direction

The eventual target is:

`continuous cognitive dynamics → NSA state transition algebra → governed agency`

rather than a prompt-level safety wrapper around an LLM.
