# Algebra-Preserving State Transitions

## Motivation

In the native Typed Neural Computation (TNC) framework, the state vector $\sigma$ propagates through the model alongside the semantic representation $m$. The standard state update is defined as a learned, unconstrained neural function:

$$ \sigma_{l+1} = g_\theta(m_l, \sigma_l) $$

While flexible, this unconstrained formulation does not guarantee that the algebraic invariants of the state lattice $\Sigma$ are preserved. For instance, in a strictly monotonic security lattice (e.g., Bell-LaPadula), we require:

$$ \sigma_{l+1} \succeq \sigma_l $$

Empirical evaluation (Model C in our 5-way benchmark) reveals that an unconstrained neural update violates this monotonicity requirement approximately **31.75%** of the time. While post-hoc clamping can restore the invariant, it creates a discrepancy between what the model learns and the algebraic rules it must follow.

## Algebra-Preserving Updates

To solve this, we redefine the state transition to be *structurally* algebra-preserving:

$$ \sigma_{l+1} = \sigma_l \sqcup \Delta_\theta(m_l, \sigma_l) $$

Where:
- $\sqcup$ is the dimension-specific lattice join operator.
- $\Delta_\theta$ is the neural increment, projected to ensure it represents a valid lattice element.

Because the join operator satisfies $a \sqcup b \succeq a$ for all $a, b \in \Sigma$, the update is structurally guaranteed to preserve monotonicity.

### Dimension-Specific Operators

Each dimension of the product state vector $\sigma$ utilizes a tailored operator matching its mathematical structure:

| Dimension | Invariant | Algebra-Preserving Operator |
| :--- | :--- | :--- |
| **Security** | Monotone $\uparrow$ (Restriction) | $\sigma_{s, l+1} = \max(\sigma_{s, l}, \text{softmax}(\Delta_s) \cdot (L - 1))$ |
| **Confidence** | Monotone $\downarrow$ (Worst-case) | $\sigma_{c, l+1} = \min(\sigma_{c, l}, \text{sigmoid}(\Delta_c))$ |
| **Provenance**| Set Union (Growing) | $\sigma_{p, l+1} = \max(\sigma_{p, l}, \text{sigmoid}(\Delta_p))$ |
| **License** | Monotone $\uparrow$ (Tier) | $\sigma_{lk, l+1} = \max(\sigma_{lk, l}, \text{softmax}(\Delta_{lk}) \cdot (T - 1))$ |

*Note: For binary provenance bits, $\max$ acts as a continuous approximation of the bitwise OR ($a \lor b$).*

## Experimental Validation

We evaluate this approach as **Model E** in the 5-way alignment benchmark (`make exp-algebra-preserving`).

**Hypothesis**: Algebra-preserving transitions (Model E) will reduce monotonicity violations to $\sim 0\%$ while maintaining semantic capability (PPL) close to the unconstrained native model (Model C), avoiding the massive PPL penalty incurred by the behavioural value-alignment layer (Model D).

By structurally separating state *representation* (which is algebra-preserving) from policy *enforcement* (which can be handled via masks or value alignment), we provide a more robust and conceptually cleaner alignment substrate.
