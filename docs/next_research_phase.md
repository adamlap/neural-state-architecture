# Next Research Phase: Semantic, Normative, and Continuous Integration

This phase turns the typed `ν` primitive into a testable control architecture without conflating normative assessment with authority or consciousness.

## Information flow

```text
input / model output
        │
        ▼
 semantic assessment
        │
        ├── provenance
        ├── confidence
        └── normative state ν
                 │
                 ▼
          normative policy
                 │
                 ▼
          security decision
                 │
                 ▼
        trusted capability gate
```

The hard security state remains authoritative. No semantic or normative component can directly grant a capability or weaken σ_h.

## Continuous coupling

CCE should maintain a persistent state stream independently of request frequency. Normative state may be updated by observations, model heartbeats, and relevant events, but each update must be explicit and auditable.

A future continuous transition can be represented as:

`(σ_t, ν_t, m_t, e_t) -> (σ_{t+1}, ν_{t+1}, m_{t+1})`

where `m` is persistent memory and `e` is an observed event. This is a research model, not a claim about biological consciousness.

## Experiments

Compare episodic and persistent configurations using measurable metrics: state continuity, temporal prediction error, recovery after perturbation, memory integration, normative-state stability, uncertainty calibration, adversarial robustness, and compute/energy cost.

## Safety requirements

Every experiment must preserve hard-state invariants, fail-closed behavior where policy requires it, provenance of normative assessments, capability isolation, and deterministic replay where practical.
