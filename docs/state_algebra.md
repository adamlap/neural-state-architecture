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
