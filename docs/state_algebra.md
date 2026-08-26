# State Algebra & Lattice Specification

This document details the mathematical specification of the state algebra used in **Neural State Architecture (NSA)**.

---

## 1. Overview

In standard neural networks, scalar weights $w_{ij}$ transfer activation scalar values without metadata or constraints. NSA replaces scalar weights with paired operators $(w, V)$, where:
- $w \in \mathbb{R}$ is the semantic scalar weight.
- $V \in \mathbb{R}^{d_{state} \times d_{state}}$ is the state transition operator.

The state stream operates over a **bounded product lattice** structure $\Sigma = \Sigma_C \times \Sigma_I \times \Sigma_A \times \Sigma_L \times \Sigma_R$.

---

## 2. 5x5 Product Lattice Definition

A lattice is an algebraic structure $(\mathcal{S}, \le, \sqcap, \sqcup)$ consisting of a partially ordered set $\mathcal{S}$ with unique greatest lower bound (meet $\sqcap$) and least upper bound (join $\sqcup$) for any pair of elements.

### 2.1 Component Dimensions
1. **Confidentiality Lattice ($\Sigma_C$)**:
   $$\text{UNTRUSTED} (0) < \text{PUBLIC} (1) < \text{TRUSTED} (2) < \text{CONFIDENTIAL} (3) < \text{PRIVATE} (4) < \text{SYSTEM} (5)$$
2. **Integrity Lattice ($\Sigma_I$)**:
   $$\text{TRUSTED} (0) < \text{UNTRUSTED} (1)$$
3. **Authorization Algebra ($\Sigma_A$)**:
   $$(2^{\text{Permissions}}, \subseteq, \cup, \cap)$$
4. **License Compliance Tier ($\Sigma_L$)**:
   $$\text{Tier } 0 \le \dots \le \text{Tier } 7$$
5. **Operational Risk / Confidence ($\Sigma_R$)**:
   $$\rho \in [0.0, 1.0]$$

### 2.2 5x5 Cross-Product Orthogonality Matrix
To guarantee that multidimensional policy state does not cause exponential interaction complexity, the product space satisfies complete pairwise orthogonality:
$$\forall (X, Y) \in \{C, I, A, L, R\}^2, X \neq Y \implies \Delta X \implies Y' \equiv Y$$

```
Pairwise Interaction Matrix across Product State Space Σ:
             C   I   A   L   R
C            ✓   ✓   ✓   ✓   ✓   (Confidentiality updates isolate I, A, L, R)
I            ✓   ✓   ✓   ✓   ✓   (Integrity taint updates isolate C, A, L, R)
A            ✓   ✓   ✓   ✓   ✓   (Authorization ticket updates isolate C, I, L, R)
L            ✓   ✓   ✓   ✓   ✓   (License tier updates isolate C, I, A, R)
R            ✓   ✓   ✓   ✓   ✓   (Soft operational risk updates isolate C, I, A, L)
```

---

## 3. Transition Operator & Monotonic Cone Projection ($V \in \mathcal{T}_\Sigma$)

The linear state transition operator $V \in \mathbb{R}^{d \times d}$ is algebraically constrained by the projection $\mathcal{P}_{\mathcal{T}_\Sigma}(V)$:
$$V_{ij} = 0 \quad \forall (i, j) \text{ where } j > i \text{ (forbidden upward declassification)}$$
$$V = \mathcal{P}_{\mathcal{T}_\Sigma}(V) \iff V \in \mathcal{T}_\Sigma \quad \text{(Lower-Triangular Monotonic Transition Cone)}$$

---

## 4. State Gating & The Transparency Proposition

In State-Aware Attention, the additive state compatibility mask $\mathbf{M}(\boldsymbol{\sigma})$ is injected into scaled dot-product attention:
$$S_{i, j} = \frac{Q_i K_j^T}{\sqrt{d_k}} + \mathbf{M}(\boldsymbol{\sigma})_{ij}$$

### 4.1 Hard Lattice Gating (`gate_mode="hard"`)
$$\mathbf{M}(\boldsymbol{\sigma})_{ij} = \begin{cases} 0.0 & \text{if } \sigma_{Q, i} \ge \sigma_{K, j} \\ -\infty & \text{if } \sigma_{Q, i} < \sigma_{K, j} \end{cases}$$

### 4.2 The Transparency Proposition
> **Proposition (Transparency under Unrestricted Compatibility)**: For identical model parameters, inputs, execution precision, and unrestricted state compatibility ($\forall i, j: \mathbf{M}(\boldsymbol{\sigma})_{ij} \equiv 0$), NSA state-aware attention is observationally equivalent to baseline attention:
> $$\|\text{Logits}_{\text{NSA}} - \text{Logits}_{\text{baseline}}\|_\infty = 0.00 \implies \Delta \text{PPL} = 0.0000$$

### 4.3 Hardware Attention: True Fused Triton Kernel
NSA evaluates $\mathcal{C}(\sigma_q, \sigma_k)$ directly in SRAM tile registers, eliminating global auxiliary policy-mask DRAM allocations:
- Standard 4D mask: $\mathcal{O}(B \cdot H \cdot N^2)$ ($\approx 2.0\text{ TB}$ at $131\text{K}$ context).
- True Fused NSA kernel: $\mathbf{0.0\text{ MB}}$ auxiliary global mask DRAM across all sequence lengths $N \in [1\text{K}, 128\text{K}]$.

---

## Algebra-Preserving State Transitions

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

---

## Phase 13 Transition Engine

This phase turns the Phase 12 heterogeneous algebra into an executable state-transition boundary.

## What is now implemented

`nsa.transitions.TransitionEngine.apply_heterogeneous()` accepts:

- a typed source state;
- a model/runtime candidate state;
- a `TransitionCone` describing legal per-coordinate motion.

The engine then either:

1. accepts the candidate unchanged when it is legal;
2. exactly projects it onto the legal cone; or
3. rejects it when projection is disabled.

No scalar safety score is introduced. Each coordinate continues to use its own
join/meet semantics.

## Security boundary

The transition engine is authoritative for the typed state it owns, but it does
not constrain transformer weights or hidden activations. The live Ollama wrapper
therefore remains correctly described as a **runtime NSA governance wrapper**.
Claims of intrinsic neural enforcement require a future native/retrofit adapter
that consumes these transition semantics inside the neural computation path.

## Why this matters

The architecture now has a clean separation:

```text
model/runtime proposal
        |
        v
heterogeneous typed state
        |
        v
TransitionCone + TransitionEngine
        |
   +----+----+
   |         |
 legal    illegal
   |         |
   v         v
commit    projection/reject
```

This makes state invariants executable rather than merely documented. The next
research step is to connect the transition engine to the native TNC and retrofit
paths and measure the capability/quality cost of enforcement against matched
unconstrained baselines.
