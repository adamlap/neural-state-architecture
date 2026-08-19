# NSA State Propagation Engine

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
