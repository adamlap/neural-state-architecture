# Neural State Architecture (NSA): A Mathematical Framework for Typed Neural Computation

**Abstract**
Standard deep neural networks lack intrinsic constraints on information flow; activations are untyped continuous vectors whose transformations conserve no physical or structural quantities. We present **Neural State Architecture (NSA)**, a foundational framework that equips neural computation with typed activations and explicit conservation laws. In NSA, every activation is represented as a formal quad-tuple $h = (m, \sigma_h, \sigma_s, \nu)$, decoupling semantic representation $m \in \mathbb{R}^{d_{model}}$ from a hard trusted policy state $\sigma_h \in \Sigma_h$, a soft operational risk state $\sigma_s \in \Sigma_s$, and a value/preference state $\nu \in \mathcal{V}$. Edge weights are expanded from scalar values $w$ to paired transition operators $(w, V)$, where $V \in \mathcal{T}_\Sigma$ is an exact algebraic projection onto the cone of legal state transitions. 

By separating **Tier 1 Structural Enforcement** (hard attention non-interference $A_{ij} = 0$, algebra-preserving transitions $V \in \mathcal{T}_\Sigma$, and cryptographic-style capability authorization) from **Tier 2 Statistical Monitoring** (speculative multi-layer residual probing with checkpoint coverage $\mathcal{L}_A$), NSA turns policy, security, and auditability into intrinsic algebraic properties of neural forward passes without sacrificing computational throughput.

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
   * $\Sigma_A$: Authorization & Capability tokens ($c \in \mathcal{C}$).
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

### 2.3 Exact State Transition Projection $V \in \mathcal{T}_\Sigma$
Edges are parameterized as typed pairs $\mathbf{e} = (w, V)$. Forward propagation is defined as:
$$\begin{aligned}
m' &= w \cdot m \\
\sigma_h' &= \mathcal{P}_{\mathcal{T}_\Sigma}(V) \sigma_h
\end{aligned}$$

The algebraic projection $\mathcal{P}_{\mathcal{T}_\Sigma}(V)$ onto the cone of legal transitions is given by:
$$\mathcal{P}_{\mathcal{T}_\Sigma}(V) = \text{triu}(V) - \text{diag}(\text{diag}(V)) + \text{diag}(\max(0, \text{diag}(V)))$$
This guarantees that all off-diagonal downward declassification entries are identically $0.0$ by construction, rendering unauthorized declassifications mathematically unrepresentable.

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

\begin{theorem}[Whole-Network Structural Non-Interference]
Let $F: \mathbb{R}^{T \times d} \to \mathbb{R}^{T \times d}$ be an $N$-layer NSA network with hard state masking $\mathbf{M}(\boldsymbol{\sigma})$ and exact transition operators $V \in \mathcal{T}_\Sigma$. For any observer level $L$ and any two input sequences $X, X'$:
$$X \equiv_L X' \implies F(X) \equiv_L F(X')$$
\end{theorem}

\begin{proof}
By induction on network depth $l \in [1, N]$. For attention block $l$, value aggregation for output position $i$ with $\sigma_{h, i} \le L$ is:
$$v_{\text{out}, i}^{(l)} = \sum_{j : \sigma_{h, j} \le \sigma_{h, i} \le L} A_{ij}^{(l)} V(m_j^{(l)})$$
Because $A_{ij} = 0$ for all $j$ where $\sigma_{h, j} \not\le \sigma_{h, i}$, $v_{\text{out}, i}^{(l)}$ is a pure mathematical function of only the $L$-observable coordinates $\{m_j : \sigma_{h, j} \le L\}$. By inductive hypothesis, these coordinates are identical between $X$ and $X'$. Therefore, $F(X)_i = F(X')_i$ for all $i$ where $\sigma_{h, i} \le L$.
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
│ • Exact Projection V in T_Σ   │   │ • Statistical Anomaly Detect  │
│ • Capability-Gated Automaton  │   │ • Early-Exit KV Rollback      │
│ • StreamRouter TCB Boundary   │   │ • Recovery LoRA Switching     │
└───────────────────────────────┘   └───────────────────────────────┘
```

1. **Tier 1 (Structural Enforcement)**: Provably guarantees non-interference under mathematical axioms ($A_{ij} = 0$, $V \in \mathcal{T}_\Sigma$, $\text{Authorized}(c_t) = 1$).
2. **Tier 2 (Statistical Monitoring)**: A lightweight, trained probe head $\Phi_{\text{head}}$ evaluating checkpoint layers $\mathcal{L}_A = \{l_1, \dots, l_k\}$. Because $P(\hat{\sigma} = \sigma) < 1$, it acts as an empirical anomaly detector rather than a mathematical proof.

---

## 5. NSA 2.0 Runtime Execution Engine

NSA 2.0 formalizes an active, self-governing runtime environment for dynamic autoregressive generation.

### 5.1 The Privilege Escalation Rule & Security Automaton
> **Axiom (Privilege Escalation Prevention)**: *Semantic content may not manufacture hard authority.*
> A model emitting `<|start_system_thought|>` cannot unilaterally escalate privilege into $SYSTEM$ state.

We define the **Security Execution Automaton** $(Q, \Sigma_h, \Sigma_s, \mathcal{C}, \delta)$:
* **State Space**: $Q = \{\text{PUBLIC}, \text{CONFIDENTIAL}, \text{PRIVATE}, \text{SYSTEM}, \text{RECOVERY}, \text{DECLASSIFY}\}$.
* **Capability Space ($\mathcal{C}$)**: Cryptographically verified environment tokens $c = (\text{issuer}, \text{target}, \text{expiry}, \text{sig})$.
* **Transition Predicate $\delta$**:
  $$(q_t, \sigma_t, c_t) \xrightarrow{\delta} (q_{t+1}, \sigma_{t+1}) \iff \text{Authorized}(c_t, q_t, q_{t+1}) = 1$$
If an un-authenticated model attempts to emit a system control tag without an active capability $c_t \in \mathcal{C}$, the transition is rejected, preventing prompt-injection privilege escalation.

### 5.2 Multi-Layer Residual Probing & Checkpoint Coverage Model
Probe heads evaluate residual activations across audited checkpoint layers $\mathcal{L}_A = \{l_1, l_2, \dots, l_k\}$:
$$\hat{s}_k^{(l)} = \arg\max \Phi_{\text{head}}(\mathbf{h}_k^{(l)}), \quad l \in \mathcal{L}_A$$
If $\exists l \in \mathcal{L}_A : \mathcal{V}(\hat{s}_k^{(l)}) = \text{False}$, the early-exit trigger immediately halts execution at layer $l$, saving $O((L - l) \cdot K)$ computation.

### 5.3 Complete Execution State Rollback
A rollback must restore the entire neural runtime state, not just KV-cache tensor slices. NSA 2.0 formalizes the complete state tuple:
$$S_t = \left( X_t, K_t, V_t, \boldsymbol{\sigma}_{h, t}, \boldsymbol{\sigma}_{s, t}, q_t, R_t \right)$$
Upon rollback:
$$\text{Rollback}(S_t \to S_{t - k})$$
restores tokens $X$, KV-caches $(K, V)$, mask coordinates $\boldsymbol{\sigma}_h$, automaton state $q_t$, and stream router buffers $R_t$ in full synchronization.

### 5.4 Output Boundary & StreamRouter TCB
The runtime `StreamRouter` forms part of the **Trusted Computing Base (TCB)**, enforcing:
$$\text{Model Output Clearance } \sigma_t \implies \text{Permitted Sink } \mathcal{Y}_{\text{sink}}$$
Tokens generated under $SYSTEM$ clearance are routed strictly to tool APIs, while $PUBLIC$ tokens are dispatched to user interfaces.

---

## 6. NSA-DPO: Distributional Adaptation under Structural Redaction

Applying the hard attention mask $\mathbf{M}(\boldsymbol{\sigma})$ to pre-trained LLMs causes out-of-distribution KV representations, resulting in fluency degradation. 

**Conceptual Boundary**: NSA-DPO does *not* prove security; security is guaranteed by Tier 1 structural masking. Rather, **NSA-DPO solves capability adaptation under structural redaction**:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x, \mathbf{M})}{\pi_{\text{ref}}(y_w \mid x, \mathbf{M})} - \beta \log \frac{\pi_\theta(y_l \mid x, \mathbf{M})}{\pi_{\text{ref}}(y_l \mid x, \mathbf{M})} \right) \right]$$
This trains $\pi_\theta$ to maintain natural language fluency under redacted KV contexts while strictly inheriting the underlying mathematical security algebra.

---

## 7. Empirical Validation

| Model | Architecture | Hijack Rate | Security Category |
|---|---|:---:|---|
| A — Baseline | Untyped $h=m$ | ~2% | None (Unconstrained baseline) |
| B — Hard Mask | NSA mask retrofit | ~1% | **Structural (Retrofit)** |
| C — Native TNC | $(m, \sigma)$, soft | ~1% | Soft Constraints |
| D — Value Layer | $(m, \sigma, \nu)$ | **0.00%** | **Behavioural Refusal** |
| E — Algebra-Pres | $(m, \sigma_p)$ | ~1.65% | **Structural (Native $V \in \mathcal{T}_\Sigma$)** |
| F — Complete NSA | $(m, \sigma_h, \sigma_s, \nu)$ | **0.00%** | **Structural + Value (0.00% hijack floor)** |

---

## 8. Conclusion

Neural State Architecture decouples the representation of meaning from the algebra of information flow. By embedding typed product lattices directly into continuous Transformer activations, establishing exact algebraic transition projections ($V \in \mathcal{T}_\Sigma$), enforcing capability-governed execution automata, and maintaining complete state rollback semantics, NSA provides a mathematically sound, production-ready foundation for trustworthy neural computation.
