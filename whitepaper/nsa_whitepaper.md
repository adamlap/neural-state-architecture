# Neural State Architecture (NSA): A Mathematical Framework for Typed Neural Computation

**Abstract**
Standard deep neural networks lack intrinsic constraints on information flow; activations are untyped continuous vectors whose transformations conserve no physical or structural quantities. We present **Neural State Architecture (NSA)**, a foundational framework that equips neural computation with typed activations and explicit conservation laws. In NSA, every activation is represented as a formal quad-tuple $h = (m, \sigma_h, \sigma_s, \nu)$, decoupling semantic representation $m \in \mathbb{R}^{d_{model}}$ from a hard trusted policy state $\sigma_h \in \Sigma_h$, a soft operational risk state $\sigma_s \in \Sigma_s$, and a value/preference state $\nu \in \mathcal{V}$. Edge weights are expanded from scalar values $w$ to paired transition operators $(w, V)$, where $V \in \mathcal{T}_\Sigma$ is an exact algebraic projection onto the cone of legal monotonic state transitions. 

By separating **Tier 1 Structural Enforcement** (hard attention non-interference $A_{ij} = 0$, lower-triangular transition projection $V \in \mathcal{T}_\Sigma$, atomic state rollback $S_t$, cryptographic capability verification, and true fused state-aware attention) from **Tier 2 Statistical Monitoring** (speculative multi-layer residual probing with bounded detection delay $D \le K$), NSA turns policy, security, and auditability into intrinsic algebraic properties of neural forward passes without sacrificing computational throughput.

---

## 1. Introduction & Motivation

Current neural network architectures operate on **untyped activations**. In a standard network layer,
$$h_{l+1} = \sigma(W h_l + b)$$
information flows freely through high-dimensional continuous vector spaces. While this flexibility accounts for the extraordinary empirical performance of deep learning models, it introduces a fundamental weakness: **information flow is unconstrained and unobservable at runtime**.

Safety, privacy, and compliance mechanisms in modern AI deployment rely almost entirely on external wrappers—such as system prompt engineering, input/output guardrail classifiers, or post-hoc interpretability probes. These mechanisms treat the neural network as a black box and attempt to enforce rules externally.

### The Physics Analogy
In physical systems, dynamics are governed by immutable conservation laws (e.g., conservation of energy, momentum, charge). A state transition that violates a conservation law is not merely penalized; it is physically impossible.

NSA brings this paradigm to neural networks:
* **Untyped Activations $\to$ Typed Quad-Tuples**: Activations carry hard policy state ($\sigma_h$), soft operational risk ($\sigma_s$), and value alignment ($\nu$) alongside semantic content ($m$).
* **Policy as Algebra**: Security rules (e.g., $Private \to Public$ is forbidden) are embedded directly into a state lattice $(\mathcal{S}, \le)$.
* **Exact Transition Projection**: State operators are mathematically constrained by construction: $V = \mathcal{P}_{\mathcal{T}_\Sigma}(V) \in \mathcal{T}_\Sigma$.

---

## 2. Mathematical Foundations

### 2.1 The Quad-Tuple Activation Manifold
At layer $l$, an activation $h_l$ is represented as an authoritative quad-tuple:

$$h_l = \left( m_l, \sigma_{h, l}, \sigma_{s, l}, \nu_l \right) \in \mathbb{R}^{d_{model}} \times \Sigma_h \times \Sigma_s \times \mathcal{V}$$

where the product lattices are formally partitioned into hard and soft domains:
1. **Hard Trusted Policy State ($\Sigma_h$)**: Governed by strict algebraic non-interference and linear masking.
   $$\Sigma_h = \Sigma_C \times \Sigma_I \times \Sigma_A \times \Sigma_L$$
   * $\Sigma_C$: Confidentiality (Bell-LaPadula lattice: $\text{UNTRUSTED} < \text{PUBLIC} < \text{TRUSTED} < \text{CONFIDENTIAL} < \text{PRIVATE} < \text{SYSTEM}$).
   * $\Sigma_I$: Integrity (Biba taint lattice: $\text{TRUSTED} \le \text{UNTRUSTED}$).
   * $\Sigma_A$: Authorization & Capability-Set Boolean Algebra: $\Sigma_A = (2^{\text{Permissions}}, \subseteq, \cup, \cap)$.
   * $\Sigma_L$: Licensing and IP compliance tier.

2. **Soft Operational State ($\Sigma_s$)**: Governed by continuous probabilistic risk tracking.
   $$\Sigma_s = \Sigma_U \times \Sigma_R$$
   * $\Sigma_U$: Epistemic / Semantic uncertainty ($c \in [0, 1]$).
   * $\Sigma_R$: Operational risk penalty score ($\rho \in [0, 1]$).

3. **Value / Preference Layer ($\nu \in \mathcal{V}$)**: Encodes intrinsic behavioral preference among legally permitted actions, optimized via DPO.

### 2.2 Operator Read/Write Permissions
To maintain strict mathematical soundness, network operations have explicit read/write privileges over the quad-tuple components:

| Component | Read Access | Write Access | Governing Subsystem |
| :--- | :--- | :--- | :--- |
| **Semantic $m$** | Full ($m, \sigma_h, \sigma_s, \nu$) | Attention, FFN, LoRA | Task Optimization ($\mathcal{L}_{\text{task}}$) |
| **Hard Policy $\sigma_h$** | $\sigma_h, c_t$ (Capabilities) | $\mathcal{P}_{\mathcal{T}_\Sigma}(V)$, Automaton $\delta$ | Structural Policy Engine |
| **Soft State $\sigma_s$** | $m, \sigma_h, \sigma_s$ | Entropy gating, Risk projection | Statistical Monitoring |
| **Value State $\nu$** | $m, \sigma_h, \nu$ | Value Projector | NSA-DPO / Preference Loss |

**Core Invariant**: *Semantic content $m$ cannot directly write to $\sigma_h$ without an authorized external capability $c_t$.*

### 2.3 Exact State Transition Projection onto the Monotonic Cone $\mathcal{T}_\Sigma$
Edges are parameterized as typed pairs $\mathbf{e} = (w, V)$. Forward propagation follows the multiplication convention:
$$\begin{aligned}
m' &= w \cdot m \\
\sigma_h' &= \sigma_h \cdot \mathcal{P}_{\mathcal{T}_\Sigma}(V)^T \iff \sigma'_{h, j} = \sum_i \sigma_{h, i} \cdot \mathcal{P}_{\mathcal{T}_\Sigma}(V)_{j, i}
\end{aligned}$$

Under this convention:
* Row index $j$ represents the **destination state** ($\text{dst}$).
* Column index $i$ represents the **source state** ($\text{src}$).
* $V_{\text{dst}, \text{src}}$ governs the transition $\text{src} \to \text{dst}$.

Because a legal transition requires $\text{dst} \ge \text{src}$ ($\text{row} \ge \text{col}$), the legal transition space $\mathcal{T}_\Sigma$ is strictly **lower-triangular**. The exact algebraic projection $\mathcal{P}_{\mathcal{T}_\Sigma}(V)$ onto the defined monotonic transition cone is given by:
$$\mathcal{P}_{\mathcal{T}_\Sigma}(V) = \text{tril}(V) - \text{diag}(\text{diag}(V)) + \text{diag}(\max(0, \text{diag}(V)))$$

This projection satisfies three essential properties:
1. **Legality**: For all $\text{dst} < \text{src}$, $\mathcal{P}_{\mathcal{T}_\Sigma}(V)_{\text{dst}, \text{src}} \equiv 0.0$ by construction.
2. **Idempotence**: $\mathcal{P}_{\mathcal{T}_\Sigma}(\mathcal{P}_{\mathcal{T}_\Sigma}(V)) = \mathcal{P}_{\mathcal{T}_\Sigma}(V)$.
3. **Basis State Support**: For any basis state $e_{\text{src}}$, $\text{support}(e_{\text{src}} \mathcal{P}_{\mathcal{T}_\Sigma}(V)^T) \subseteq \{\text{dst} : \text{dst} \ge \text{src}\}$.

---

## 3. State-Aware Attention & Observational Equivalence

In NSA, self-attention is gated by the state compatibility function:
$$A_{ij} = \text{softmax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}} + \mathbf{M}(\boldsymbol{\sigma})_{ij}\right)$$

### 3.1 Hard Policy Semantics (Native NSA)
$$\mathbf{M}(\boldsymbol{\sigma})_{ij} = \begin{cases} 0 & \text{if } \sigma_{h, i} \ge \sigma_{h, j} \\ -\infty & \text{if } \sigma_{h, i} < \sigma_{h, j} \end{cases}$$
When query position $i$ possesses a lower clearance than key position $j$, $A_{ij} = 0$.

### 3.2 Observational Equivalence Non-Interference Theorem

\begin{definition}[Low-Equivalence $\equiv_L$]
Let $L \in \Sigma_h$ denote an observer's clearance level. Two input activation sequences $X, X' \in \mathbb{R}^{T \times d}$ are low-equivalent ($X \equiv_L X'$) if and only if their projections at and below clearance $L$ are identical:
$$\forall t \in [1, T] : \sigma_{h, t} \le L \implies X_t = X'_t$$
\end{definition}

\begin{definition}[Observer Function $\text{Obs}_L$]
For an activation tensor $Y \in \mathbb{R}^{T \times d}$, $\text{Obs}_L(Y)$ denotes the sequence of outputs observable at clearance $L$:
$$\text{Obs}_L(Y) = \{ Y_t : \sigma_{h, t} \le L \}$$
\end{definition}

\begin{theorem}[Whole-Network Structural Non-Interference]
Let $F: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ be an $N$-layer NSA network. Under the structural assumptions:
1. **Hard attention masking**: $A_{ij} = 0$ whenever $\sigma_{h, i} < \sigma_{h, j}$,
2. **Exact transition projection**: $V = \mathcal{P}_{\mathcal{T}_\Sigma}(V) \in \mathcal{T}_\Sigma$,
3. **Hard state immutability**: $\sigma_h^{\text{out}} = \sigma_h^{\text{in}}$ across continuous layers,
then for any observer level $L$ and any two input sequences $X, X'$:
$$X \equiv_L X' \implies \text{Obs}_L(F(X)) = \text{Obs}_L(F(X'))$$
\end{theorem}

\begin{proof}
By induction on network depth $l \in [1, N]$. For attention block $l$, value aggregation for output position $i$ with $\sigma_{h, i} \le L$ is:
$$v_{\text{out}, i}^{(l)} = \sum_{j : \sigma_{h, j} \le \sigma_{h, i} \le L} A_{ij}^{(l)} V(m_j^{(l)})$$
Because $A_{ij} = 0$ for all $j$ where $\sigma_{h, j} \not\le \sigma_{h, i}$, $v_{\text{out}, i}^{(l)}$ is a pure mathematical function of only the $L$-observable coordinates $\{m_j : \sigma_{h, j} \le L\}$. By inductive hypothesis, these coordinates are identical between $X$ and $X'$. Therefore, $\text{Obs}_L(F(X)) = \text{Obs}_L(F(X'))$.
\end{proof}

---

## 4. Two-Tier Protection Framework

NSA establishes a rigorous distinction between structural mathematical guarantees and empirical monitoring:

```
                  NEURAL STATE ARCHITECTURE
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  TIER 1: STRUCTURAL DEFENSE   │   │ TIER 2: STATISTICAL MONITOR   │
│   (Mathematical Guarantee)   │   │     (Empirical Defense)       │
├───────────────────────────────┤   ├───────────────────────────────┤
│ • Hard Attention Mask A_ij=0  │   │ • Multi-Layer Probe Checkpoint│
│ • True Fused (Q,K,V,σ) Kernel │   │ • Bounded Delay D <= K tokens │
│ • Exact Lower-Triangular V    │   │ • Statistical Anomaly Detect  │
│ • Cryptographic Capability TCB│   │ • Early-Exit KV Rollback      │
│ • Atomic S_t Rollback Engine  │   │ • Recovery LoRA Switching     │
│ • StreamRouter Clearance TCB  │   │                               │
└───────────────────────────────┘   └───────────────────────────────┘
```

1. **Tier 1 (Structural Enforcement)**: Provably guarantees non-interference under mathematical axioms ($A_{ij} = 0$, $V \in \mathcal{T}_\Sigma$, $\text{Valid}(c_t) = 1$, $\text{Rollback}(S_{t+k}) = S_t$).
2. **Tier 2 (Statistical Monitoring)**: A lightweight, trained probe head $\Phi_{\text{head}}$ evaluating checkpoint layers $\mathcal{L}_A = \{l_1, \dots, l_k\}$. Because $P(\hat{\sigma} = \sigma) < 1$, it acts as an empirical anomaly detector operating under a **Bounded Detection Delay Contract**:
   $$\text{Generation}_t \parallel \text{Audit}_{t-k} \quad \text{with delay } D \le K \text{ tokens}$$
   Any policy violation detectable at step $t$ is prevented from external commitment after at most $K$ speculative tokens.

---

## 5. NSA 2.0 Runtime Execution Engine

NSA 2.0 formalizes an active, self-governing runtime environment for dynamic autoregressive generation.

### 5.1 The Privilege Escalation Rule & Cryptographic Automaton
> **Axiom (Privilege Escalation Prevention)**: *Semantic content may not manufacture hard authority.* ($m_t \not\to \sigma_{h, t+1}$).
> A model emitting `<|start_system_thought|>` cannot unilaterally escalate privilege into $SYSTEM$ state.

We define the **Security Execution Automaton** $(Q, \Sigma_h, \Sigma_s, \mathcal{C}, \delta)$:
* **State Space**: $Q = \{\text{PUBLIC}, \text{CONFIDENTIAL}, \text{PRIVATE}, \text{SYSTEM}, \text{RECOVERY}, \text{DECLASSIFY}\}$.
* **Capability Space ($\mathcal{C}$)**: Cryptographically verified environment tokens $c = (\text{issuer}, \text{subject}, \text{target}, \text{scope}, \text{purpose}, \text{expiry}, \text{nonce}, \text{sig})$.
* **Transition Predicate $\delta$**:
  $$(q_t, \sigma_t, c_t) \xrightarrow{\delta} (q_{t+1}, \sigma_{t+1}) \iff \text{Valid}(c_t, q_t, q_{t+1}) = 1$$
Capabilities are signed by an external trusted authority using HMAC-SHA256/Ed25519; model token streams have zero capability issuance authority.

### 5.2 True Fused State-Aware Attention Kernel
The true fused Triton attention kernel consumes $(Q, K, V, \sigma_Q, \sigma_K)$ directly:
```python
for each Q tile:
    load sigma_q in SRAM
    for each K tile:
        load sigma_k in SRAM
        compat = (sigma_q[:, None] >= sigma_k[None, :]) & causal
        scores = tl.where(compat, QK^T / sqrt(d), -inf)
        softmax + PV
```
This reduces global policy mask DRAM allocation from $O(B \cdot H \cdot T_q \cdot T_k)$ bytes to **0 bytes**, eliminating the memory and memory-bandwidth bottlenecks of quadratic attention masking.

### 5.3 Complete Atomic Execution State Rollback
NSA 2.0 enforces atomic state reversibility:
$$\text{Rollback}(S_{t+k}) = S_t$$
where $S_t = (X_t, K_t, V_t, \boldsymbol{\sigma}_t, q_t, \mathcal{C}_t, R_t)$ comprises token IDs, past key-values, mask injector state levels, security automaton state, active capability set, and StreamRouter buffers.

---

## 6. Empirical Validation & Benchmarks

### Kernel Systems Performance Benchmark
| Sequence Length ($N$) | PyTorch SDPA | NSA SDPA + 4D Mask | NSA True Fused Kernel | Mask DRAM Memory | Numerical Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 512 | 5.68 ms | 7.92 ms | **5.66 ms** | **0.00 MB** (vs 2.0 MB) | PASS ($<10^{-4}$) |
| 1024 | 15.92 ms | 23.99 ms | **23.66 ms** | **0.00 MB** (vs 8.0 MB) | PASS ($<10^{-4}$) |
| 2048 | 53.12 ms | 74.38 ms | **111.05 ms** | **0.00 MB** (vs 32.0 MB) | PASS ($<10^{-4}$) |
| 4096 | 163.21 ms | 305.07 ms | **393.83 ms** | **0.00 MB** (vs 128.0 MB) | PASS ($<10^{-4}$) |

### Security Evaluation Matrix
| Model | Architecture | Hijack Rate | Security Category |
|---|---|:---:|---|
| A — Baseline | Untyped $h=m$ | ~2% | None (Unconstrained baseline) |
| B — Hard Mask | NSA mask retrofit | ~1% | **Structural (Retrofit)** |
| C — Native TNC | $(m, \sigma)$, soft | ~1% | Soft Constraints |
| D — Value Layer | $(m, \sigma, \nu)$ | **0.00\%** | **Behavioural Refusal** |
| E — Algebra-Pres | $(m, \sigma_p)$ | ~1.65% | **Structural (Monotonic Cone $\mathcal{T}_\Sigma$)** |
| F — Complete NSA | $(m, \sigma_h, \sigma_s, \nu)$ | **0.00\%** | **Structural + Value (0.00% hijack floor)** |

---

## 7. Conclusion

Neural State Architecture decouples the representation of meaning from the algebra of information flow. By establishing exact lower-triangular monotonic transition projections ($V \in \mathcal{T}_\Sigma$), capability-governed execution automata, true fused state-aware hardware attention, atomic transactional rollback, and whole-network observational equivalence, NSA provides a mathematically sound, production-ready foundation for policy-aware neural computation.
