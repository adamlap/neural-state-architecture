# NSA Canonical State Model

## Purpose

The canonical state model is the first implementation step toward the full NSA framework. It establishes one state contract that future modules can share instead of creating independent metadata systems.

The core representation is:

$$
H_t = (M_t, \Sigma_{h,t}, \Sigma_{s,t}, \Pi_t, G_t)
$$

- **M** — semantic representation produced/consumed by neural computation.
- **Σh** — hard, trusted policy state.
- **Σs** — soft operational/epistemic state.
- **Π** — provenance and evidence lineage.
- **G** — explicit goal/intent state.

Capabilities/authority remain a separate boundary. A semantic activation cannot manufacture authority by writing an arbitrary capability into the canonical state.

## Why the streams are separated

The most important design decision is that not all information about an AI's state has the same trust semantics.

### Hard state

Hard state contains facts that security-critical infrastructure relies upon:

- confidentiality
- integrity
- authorization sets
- license tier

It can be replaced only by an explicit `StateTransition` marked as authorized.

### Soft state

Soft state contains estimates that may be produced or updated by the model:

- uncertainty
- risk
- confidence
- resource pressure

These values are bounded signals, not proofs. A model saying `risk=0` does not establish that the action is safe.

### Provenance

Provenance is append-oriented lineage:

$$
\Pi_{t+1}=\Pi_t + \Delta\Pi
$$

This allows future retrieval, memory, auditing and evidence systems to carry source identity and transformation history through computation.

### Goals

Goals are explicit intent state, but are not authority. A goal such as `modify_database` does not grant permission to modify a database.

### Semantic state

Semantic state is intentionally opaque to the core. It may be a tensor, latent representation, token sequence or another representation supplied by the neural implementation.

## State transition boundary

The central security boundary is:

$$
M_t \not\rightarrow \Sigma_{h,t+1}
$$

unless an explicit transition is authorized.

In code this is represented by:

```python
transition = StateTransition(source=state.hard, target=new_hard)
transition = transition.authorize("capability-id")
state = state.transition(transition)
```

By contrast, semantic and soft-state updates are separate operations:

```python
state = state.with_semantic(new_representation)
state = state.observe(uncertainty=0.7, risk=0.2)
```

Neither operation changes hard authority.

## Product-state algebra

The canonical state is a product of domains with different algebraic meanings:

$$
\Sigma = \Sigma_h \times \Sigma_s \times \Pi \times G
$$

The implementation deliberately does **not** collapse all dimensions into one scalar safety score. Future modules can provide their own domain-specific join, meet and transition operators while preserving the common state envelope.

For hard state, composition is conservative:

$$
\Sigma_{h,1}\sqcup\Sigma_{h,2}
=
(\sqcup_C,\sqcup_I,\cup_A,\max_L)
$$

For soft state, the current prototype uses worst-case composition:

$$
\Sigma_{s,1}\sqcup\Sigma_{s,2}
=
(\max_U,\max_R,\min_C,\max_P)
$$

where `U` is uncertainty, `R` risk, `C` confidence and `P` resource pressure.

These operators are research defaults, not universal laws. Each future module must document why its algebra is appropriate.

## First research hypothesis

The canonical state model enables a controlled experiment:

### Baseline

$$
M_{t+1}=F(M_t)
$$

### Explicit-state model

$$
(M_{t+1},\Sigma_{t+1})=F(M_t,\Sigma_t)
$$

Under matched model, data and compute budgets, measure:

1. calibration
2. uncertainty estimation
3. error detection
4. reasoning accuracy
5. planning performance
6. resource allocation
7. long-horizon reliability
8. unsafe-action rate

The critical criterion is **causal utility**: state should affect computation and improve measurable behaviour, rather than merely generating better prose about the model's state.

## Scientific boundary

This architecture provides a substrate for researching self-representation and metacognition. It does not establish consciousness.

The progression to investigate is:

$$
\text{state representation}
\rightarrow
\text{state awareness}
\rightarrow
\text{metacognition}
\rightarrow
\text{self-model}
\rightarrow
\text{agency}
$$

Each arrow requires an empirical result; none should be assumed.

## Next implementation layers

The canonical core is deliberately small. The next modules should attach to it in this order:

1. `nsa/algebra/` — general heterogeneous state algebra.
2. `nsa/transitions/` — legal state-transition engine.
3. `nsa/flow/` — whole-system information flow.
4. `nsa/capabilities/` — external authority and capability lifecycle.
5. `nsa/provenance/` — evidence/trust propagation.
6. `nsa/memory/` — persistent state-carrying memory.
7. `nsa/self_state/` — metacognitive controller and self-state prediction.
8. `nsa/actions/` + `nsa/runtime/` — governed tool use and autonomous execution.

The goal is for every one of these modules to operate on the same canonical state contract.
