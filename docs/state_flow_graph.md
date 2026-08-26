# NSA State Flow Graph

## Purpose

The original NSA security model primarily reasons about state propagation inside neural computation. The next step is to make state flow explicit across the entire AI system: model, memory, tools, agents, runtimes and external sinks.

We represent the system as:

$$G=(V,E,\Sigma)$$

where $V$ are computational/trust-boundary nodes, $E$ are explicitly permitted flows, and $\Sigma$ is the set of typed state dimensions.

An edge is not simply a connection. It is a contract:

$$e=(u,v,\Sigma_e,A_e,T_e)$$

where $\Sigma_e$ is the set of state dimensions allowed to cross, $A_e$ is the required authority, and $T_e$ is an optional transformation/declassification operation.

## Why this matters

A safe model can still be unsafe if state semantics disappear at a boundary such as:

```text
model -> RAG database
model -> tool gateway
model -> filesystem
model -> another agent
model -> external API
```

NSA therefore aims to preserve the same state semantics across the complete causal path.

## Core invariant

For a flow carrying state dimensions $D$ from $u$ to $v$:

$$
D \subseteq \Sigma_{(u,v)}
$$

and any privileged destination additionally requires:

$$
A_{(u,v)} \subseteq A_{trusted}
$$

A model-generated semantic representation cannot satisfy the second condition by itself.

## Initial implementation

`nsa/flow/graph.py` provides:

- `FlowNode`
- `FlowEdge`
- `FlowGraph`
- `FlowViolation`

This is intentionally declarative. It is the policy graph, not yet a complete tensor-level runtime.

## Next research steps

1. Attach concrete tensor/activation propagation to edges.
2. Model residual and FFN pathways as graph edges.
3. Add provenance transformations and explicit declassification.
4. Add sink/source policies.
5. Generate counterexamples for illegal paths.
6. Connect the graph to the transition engine.
7. Verify that memory and tool adapters preserve state semantics.
8. Define whole-system non-interference properties.

## Security boundary

The graph must never be treated as a substitute for the trusted runtime. It specifies allowed flow; the runtime must enforce the policy at the actual execution boundary.

---

## State Propagation Engine

The state-flow graph defines what may cross a system boundary. The propagation engine is the first runtime-facing layer that enforces that graph before state is transferred.

## Model

Given canonical state $S_t$, source $u$, destination $v$, and requested dimensions $D$:

$$
P(S_t,u,v,D,G) \rightarrow S_t'
$$

The first implementation is intentionally conservative:

$$
S_t' = S_t
$$

when the flow is permitted. No implicit transformation or privilege escalation occurs.

A denied flow produces the original state plus a structured violation reason.

## Why this is the right first step

It separates three concepts that are often conflated:

1. **State representation** — what the system knows about its current condition.
2. **Flow policy** — where each state dimension is permitted to travel.
3. **State transformation** — how a state dimension may legally change at a boundary.

Future versions will add explicit transformation/declassification objects. They should never be hidden inside adapters.

## Cognitive loop target

The propagation layer enables the eventual NSA loop:

$$
X_t
\rightarrow
S_t
\rightarrow
M_t
\rightarrow
A_t
\rightarrow
X_{t+1}
\rightarrow
\hat S_{t+1}
\rightarrow
S_{t+1}
$$

where the difference between predicted and observed self-state becomes a measurable metacognitive signal.

## Next extensions

- tensor/activation-level propagation
- provenance transformations
- explicit declassification
- residual/FFN flow edges
- memory adapters
- tool adapters
- runtime enforcement
- state prediction and observation hooks
