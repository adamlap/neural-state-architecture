# Neural State Architecture (NSA)

> **A Mathematical Framework for Typed Neural Computation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Standard neural networks conserve nothing. Activations flow through untyped continuous spaces without intrinsic rules or observable permissions. 

**Neural State Architecture (NSA)** turns policy into algebra. It introduces typed activations, formal state lattices, and paired transition operators $(w, V)$ to decouple semantic optimization from information flow optimization.

---

## Key Conceptual Foundations

### 1. Typed Activations $(m, \sigma)$
Every activation is decomposed into a dual representation:
$$h = (m, \sigma)$$
* $m \in \mathbb{R}^{d_{model}}$: Semantic representation (meaning).
* $\sigma \in \mathbb{R}^{d_{state}}$: State vector (permissions, trust, provenance, confidence).

### 2. State Transition Operators $(w, V)$
Instead of scalar edge weights $w$, NSA uses paired operators:
$$\mathbf{e} = (w, V)$$
Propagation follows dual dynamics:
$$\begin{aligned}
m' &= w \cdot m \\
\sigma' &= V \sigma
\end{aligned}$$
where $V \in \mathbb{R}^{d_{state} \times d_{state}}$ is a compact state transition matrix ($2 \times 2, 4 \times 4, 8 \times 8$).

### 3. Conservation Laws & State Algebra
State labels form a bounded lattice $(\mathcal{S}, \le, \sqcap, \sqcup)$. Transitions must obey strict monotone conservation rules:

```
    PRIVATE  ──▶  PRIVATE  (Allowed)
    PRIVATE  ──▶  PUBLIC   (Forbidden by algebra)
```

### 4. Dual-Objective Optimization
NSA decouples semantic optimization from information flow governance:
$$\mathcal{L}_{total} = \mathcal{L}_{semantic} + \lambda \cdot \mathcal{L}_{state}$$

---

## Architecture Overview

```
                                  ┌──────────────────────────────┐
  ┌──────┐     ┌─────────────────┐│  State Manifold              │
  │  m   │────▶│ StateAwareAttn  ││  σ → V(σ) → σ' via Trans.Op  │
  └──────┘     └────────┬────────┘└──────────────────────────────┘
                        │ m'                        │ σ
               ┌────────▼────────┐        ┌────────▼────────┐
               │   LayerNorm     │        │   StateUpdate   │
               └────────┬────────┘        └────────┬────────┘
                        │                          │ σ'
               ┌────────▼────────┐                 │
               │   FFN + Gate    │◀────────────────┘
               │   Γ(σ') ⊙ FFN   │
               └────────┬────────┘
                        │ m''
                        ▼
                    (m'', σ')
```

---

## Repository Structure

```
neural-state-architecture/
├── nsa/                             # Python Core Package
│   ├── algebra.py                   # State algebra: lattice, partial order, conservation laws
│   ├── state.py                     # StateVector, WeightedStateEdge, TransitionOperator
│   ├── attention.py                 # State-aware multi-head attention
│   ├── layers.py                    # NSATransformerBlock, NSATransformer
│   ├── objectives.py                # Dual loss functions: SemanticLoss, StateConstraintLoss, NSALoss
│   └── utils.py                     # Introspection, metrics, and visualization
├── whitepaper/
│   └── nsa_whitepaper.md            # Formal whitepaper & theoretical foundations
├── docs/
│   └── state_algebra.md             # Algebraic specification and state lattice docs
└── prototype/
    ├── toy_experiment.py            # End-to-end synthetic experiment (baseline vs NSA)
    ├── state_transformer.py         # Minimal working prototype block
    └── requirements.txt
```

---

## Quickstart & Verification

### Run Toy Proof-of-Concept Experiment
Compare standard Transformer vs. NSA Transformer on a privacy-aware sequence classification task:

```bash
python prototype/toy_experiment.py
```

### Basic Usage

```python
import torch
from nsa import NSATransformerBlock, DEFAULT_LATTICE

# Activations and state vectors
x = torch.randn(2, 16, 128)      # Semantic stream [batch, seq_len, d_model]
state = torch.randn(2, 16, 8)    # State stream    [batch, seq_len, state_dim]

block = NSATransformerBlock(d_model=128, state_dim=8, num_heads=8)
x_out, state_out = block(x, state)
```

---

## Applications

NSA provides a unified mathematical foundation for:
* **Intrinsic Security & Privacy** (mathematically preventing private data leakage)
* **Data Provenance & Lineage**
* **Dynamic Confidence & Uncertainty Tracking**
* **Auditability & Compliance Verification**

---

## Documentation & Whitepaper

* Read the full theoretical paper in [`whitepaper/nsa_whitepaper.md`](whitepaper/nsa_whitepaper.md)
* Read the state algebra specification in [`docs/state_algebra.md`](docs/state_algebra.md)
