# State Algebra Specification

This document details the mathematical specification of the state algebra used in **Neural State Architecture (NSA)**.

---

## 1. Overview

In standard neural networks, scalar weights $w_{ij}$ transfer activation scalar values without metadata or constraints. NSA replaces scalar weights with paired operators $(w, V)$, where:
- $w \in \mathbb{R}$ is the semantic scalar weight.
- $V \in \mathbb{R}^{d_{state} \times d_{state}}$ is the state transition operator.

The state stream operates over a **bounded lattice** structure.

---

## 2. Lattice Definition

A lattice is a algebraic structure $(\mathcal{S}, \le, \sqcap, \sqcup)$ consisting of a partially ordered set $\mathcal{S}$ with unique greatest lower bound (meet $\sqcap$) and least upper bound (join $\sqcup$) for any pair of elements.

### Default Security Lattice Hierarchy

```
       [SYSTEM]          (Level 5 - Highest Restriction)
          │
      [PRIVATE]         (Level 4)
          │
    [CONFIDENTIAL]      (Level 3)
          │
      [TRUSTED]         (Level 2)
          │
       [PUBLIC]         (Level 1)
          │
      [UNTRUSTED]       (Level 0 - Lowest Restriction)
```

### Transition Rule Formula

Information flow from state $s_{src}$ to $s_{dst}$ is **valid** if and only if:

$$\text{is\_allowed}(s_{src}, s_{dst}) \iff s_{src} \le s_{dst}$$

| From / To | UNTRUSTED | PUBLIC | TRUSTED | CONFIDENTIAL | PRIVATE | SYSTEM |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **UNTRUSTED** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **PUBLIC** | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **TRUSTED** | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| **CONFIDENTIAL** | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |
| **PRIVATE** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **SYSTEM** | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |

*Note: $\text{PRIVATE} \to \text{PUBLIC}$ is explicitly forbidden ($\mathbf{\text{✗}}$).*

---

## 3. Continuous Mapping & Differentiability

To make lattice constraints differentiable for PyTorch models:

1. **Soft State Vector Representation**:
   States $\sigma \in \mathbb{R}^{d_{state}}$ are continuous vectors.
   For discrete lattice mapping, $\sigma$ is converted to a probability vector $p \in \Delta^{|\mathcal{S}|}$ via $\text{softmax}(\sigma)$.

2. **Expected Level Projection**:
   $$\text{Level}(\sigma) = \sum_{k=0}^{|\mathcal{S}|-1} k \cdot p_k$$

3. **Loss Constraint**:
   $$\mathcal{L}_{state} = \text{ReLU}\left(\text{Level}(\sigma_{in}) - \text{Level}(\sigma_{out}) - \gamma\right)$$
   This penalizes any step where the output state restriction level falls below the input state restriction level.

---

## 4. State Gating Parameters & Mask Tensor ($M_{\text{state}}$)

In State-Aware Attention, the additive state compatibility mask $M_{\text{state}}$ is injected into scaled dot-product attention:

$$S_{i, j} = \frac{Q_i K_j^T}{\sqrt{d_k}} + M_{\text{state}}(\sigma_{Q, i}, \sigma_{K, j})$$

### Gating Formulations

1. **Hard Lattice Gating (`gate_mode="hard"`)**:
   $$M_{\text{state}}(\sigma_Q, \sigma_K) = \begin{cases} 0.0 & \text{if } \sigma_Q \ge \sigma_K \\ -10000.0 & \text{if } \sigma_Q < \sigma_K \end{cases}$$

2. **Differentiable Soft Log-Sigmoid Gating (`gate_mode="soft"`)**:
   $$M_{\text{state}}(\sigma_Q, \sigma_K) = \alpha \cdot \log\left( \text{sigmoid}\left( \frac{\sigma_Q - \sigma_K}{T} \right) \right)$$
   - $\alpha > 0$: Policy weight scale (default $\alpha = 10.0$).
   - $T > 0$: Lattice temperature parameter (default $T = 1.0$).

### Memory Efficiency: Bitpacked State Vectors (`BitpackedStateVector`)
For high-concurrency production inference (e.g. vLLM integration via `nsa.vllm_plugin`), state vectors are compressed from `float32` into `uint8` bitpacked byte tensors:

```python
from nsa.algebra import bitpack_states, unpack_states

# Compress float32 state tensor to uint8 bitpacked representation
packed_states = bitpack_states(state_levels)  # 75% memory bandwidth reduction
unpacked_states = unpack_states(packed_states)
```
