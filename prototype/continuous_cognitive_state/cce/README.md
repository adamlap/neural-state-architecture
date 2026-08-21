# CCE — Continuous Cognitive Engine

CCE is a parallel runtime research module for NSA. It is deliberately isolated
from `nsa/`: it imports NSA's public algebra and policy primitives but does not
modify, monkey-patch, replace, or depend on private implementation details of
the core NSA architecture.

## Architecture

```text
                    live sensory/input events
                              |
                              v
                    +---------------------+
                    | CCE asynchronous    |
                    | event queue          |
                    +----------+----------+
                               |
                               v
             +--------------------------------------+
             | persistent CCE state                 |
             | memory + self state + dynamic state  |
             +------------------+-------------------+
                                |
                       autonomous tick
                                |
                                v
                    +---------------------+
                    | continuous dynamics |
                    | fast/medium/slow    |
                    +----------+----------+
                               |
                               v
                         live Ollama
                         inference
                               |
                    reasoning + proposal
                               |
                               v
                    +---------------------+
                    | NSA ProductLattice   |
                    | + deployment policy  |
                    +----------+----------+
                               |
                    ALLOW / HOLD / DENY
                               |
                               v
                    deployment actuator
```

## Dynamic, not simulated

CCE does not contain a fake cognition loop or a hard-coded action catalogue.
When configured with `OllamaReasoner` and `OllamaProposalGenerator`, actual
local Ollama inference determines reasoning and action proposals at runtime.
The CCE clock continues to evolve the internal dynamical state when no input
is present. External inputs perturb that state asynchronously.

Actuators are also real integration points: CCE will call only an actuator
object supplied by the deployment after governance returns `ALLOW`. With no
actuator attached, an allowed proposal becomes `HOLD` rather than being
silently simulated.

## NSA boundary

The governor imports `ProductLattice`, `ProductStateVector`, and the existing
NSA state algebra. Capability names and policy thresholds are deployment data.
CCE cannot grant itself a capability; the model can only request an opaque
capability string.

The core `nsa/` tree is intentionally untouched by CCE development.

## Consciousness research boundary

CCE is an experimental substrate for persistent, self-referential,
continuously evolving computation. It does **not** claim that continuous
execution proves consciousness. Experiments should compare stateless LLM,
persistent-memory LLM, continuous-state LLM, and continuous-state + NSA
conditions using measurable behavioral and dynamical metrics.

## Live example wiring

```python
from prototype.continuous_cognitive_state.cce import (
    CCEConfig, CCEGovernor, CCEPolicy, CCEState,
    ContinuousCognitiveEngine, OllamaProposalGenerator, OllamaReasoner,
)

state = CCEState()
model = "llama3.2:3b"  # deployment choice; not required by CCE
reasoner = OllamaReasoner(model)
proposer = OllamaProposalGenerator(model)
policy = CCEPolicy(
    capabilities=frozenset(),
    minimum_confidence=0.8,
    maximum_risk=0.2,
)
engine = ContinuousCognitiveEngine(
    state=state,
    reasoner=reasoner,
    governor=CCEGovernor(policy),
    proposal_generator=proposer,
    config=CCEConfig(tick_hz=2.0),
)

# await engine.ingest("live sensory text")
# await engine.run()
```

For production use, supply a real actuator implementation and a deployment
policy appropriate to that actuator. Never put actuator credentials into the
LLM prompt or allow the model to alter CCE/NSA policy.
