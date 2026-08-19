# NSA Transformer Integration Plan

The next experimental stage connects explicit self-state to the existing NSA neural architecture rather than maintaining a separate toy recurrent model.

## Existing architectural primitive

NSA represents an edge as `(w, V)`: `w` controls semantic magnitude and `V` is a state transition operator. The existing `StateTransitionOperator` projects `V` into the legal transition cone before application.

## Proposed transformer block

For hidden activation `m_t` and state `sigma_t`:

$$q_t,k_t,v_t = W_qm_t,W_km_t,W_vm_t$$

$$a_t=Attention(q_t,k_t,v_t; sigma_t)$$

$$hat(m_t) = m_t + a_t$$

Then update explicit state:

$$sigma_(t+1)=P_(T_Sigma)(V_theta sigma_t + f_theta(hat(m_t),x_t))$$

and couple state back into cognition:

$$m_(t+1)=Block(hat(m_t))+G_theta(sigma_(t+1))$$

The state path must be measurable and ablatable.

## Three required conditions

1. **Baseline** — conventional transformer with matched parameter and compute budgets.
2. **NSA-native** — transformer uses explicit state and the existing NSA transition operator.
3. **NSA state-ablation** — the same trained NSA model is evaluated with cognitive state feedback disabled while hard security projections remain active.

## Metrics

- accuracy
- calibration / ECE
- Brier score
- state prediction error
- uncertainty response to distribution shift
- recovery after state perturbation
- capability-boundary recognition
- compute and memory overhead
- illegal-transition mass before projection
- post-projection invariant violations

## Scientific requirement

No claim of increased intelligence should be made from a single benchmark. Results should use multiple seeds, matched compute, confidence intervals and explicit ablations.

## Long-term experiment

Compare a baseline `M_0(X)` against `M_NSA(X,S_t)` on tasks requiring uncertainty management, planning, recovery and self-limitation.
