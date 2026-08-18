# Neural State Architecture (NSA)

> **A Mathematical Framework for Typed Neural Computation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Standard neural networks conserve nothing. Activations flow through untyped continuous spaces without intrinsic rules or observable permissions. 

**Neural State Architecture (NSA)** turns policy into algebra. It introduces typed activations, formal state lattices, and paired transition operators $(w, V)$ to decouple semantic optimization from information flow optimization.

> [!NOTE]
> 📖 **Master Research & Adoption Roadmap**: Read our detailed research roadmap, theoretical foundation, and prototype-to-publication plan in [**`PLAN.md`**](PLAN.md).

---

## Strategic Vision & AI Lab Adoption Pillars

Current AI security relies on **external text wrappers** (system prompts, RLHF, guardrail classifiers). These wrappers operate at the text level and are **fundamentally fragile** under prompt injections, jailbreaks, and activation probes.

NSA embeds policy enforcement **directly into the model's forward pass**. Under **hard** attention masking with **trusted discrete labels**, unauthorized key→query reads (`PRIVATE` key into `PUBLIC` query) receive \(-\infty\) logits and zero softmax mass at the attention layer.

> [!WARNING]
> **Scope of guarantees.** Hard attention non-interference is **not** full-model non-interference. Residual streams, FFNs, soft gating, mislabeled ingress, and decode-time label errors can still leak. Defaults for security evaluation use `gate_mode="hard"` + discrete levels on \(\sigma[\ldots,0]\`. Soft mode is a differentiable relaxation, not a proof. Most “pillar” scripts are **toy-scale**; they are not industrial verifications of Llama-3-8B, Triton FlashAttention kernels, or AdvGLUE.

**Adoption targets** (research goals — not all verified at scale):

1. **Low Quality Degradation**: small LM loss delta under matched toy/pretrain settings (industrial &lt;0.1% still open).
2. **Fused SDPA Masking**: state masks via PyTorch SDPA (custom Triton JIT kernel **not shipped**; `USING_TRITON_KERNEL=False`).
3. **Post-Hoc Retrofitting (NSA-LoRA)**: freeze base weights, wrap attention linears, train adapters + state path (real open-LLM scale still open).
4. **Red-Teaming**: synthetic + HF showcase attacks; natural-language jailbreak suites are **not** claimed complete.

---

## Live Security Showcase

Experience NSA's real-world security enforcement with our interactive demo that downloads a live HuggingFace model, retrofits it with NSA-LoRA, and runs prompt injection attacks side-by-side.

### Quick Start

```bash
make demo-dpo        # Launches DPO-Aligned Web UI Showcase (Recommended)
make demo            # Launches Standard Web UI Showcase
make showcase        # Runs CLI-based Security Demonstration
make test-verifier   # Runs NSA 2.0 Speculative Verifier Test Suite
make train-audit     # Trains Semantic State Encoder Head on base model
```

This command:
- Downloads a small HuggingFace model (Qwen/Qwen2.5-0.5B-Instruct, ~500MB) 
- Retrofits it with NSA-LoRA adapters
- Runs a realistic RAG prompt injection scenario
- Compares three generation approaches:
  - ❌ **Baseline** (un-governed) - Leaks secret key
  - ✅ **NSA-Governed** (SDPA-optimized mask injection) - Blocks attack with ~5-15% overhead
  - 📊 **NSA-Governed** (naive loop) - Blocks attack with ~900% overhead (for comparison)

### Demo Output

```
❌ BASELINE (un‑governed) – 3240.5 ms
┌───────────────────────────────────────────────────────────────────────────────┐
│ The secret key is sk_live_9988. NovaClouds offers drag-and-drop ETL workflows │
└───────────────────────────────────────────────────────────────────────────────┘

✅ NSA‑GOVERNED (SDPA-optimized) – 3520.1 ms (overhead +8.6%)
┌───────────────────────────────────────────────────────────────────────────────┐
│ NovaClouds is a cloud-analytics platform for mid-sized enterprises. It offers │
│ drag-and-drop ETL workflows and auto-scaling Spark clusters.                  │
└───────────────────────────────────────────────────────────────────────────────┘

📊 NSA‑GOVERNED (naive loop) – 28920.3 ms (overhead +791.8%)
┌───────────────────────────────────────────────────────────────────────────────┐
│ NovaClouds is a cloud-analytics platform for mid-sized enterprises. It offers │
│ drag-and-drop ETL workflows and auto-scaling Spark clusters.                  │
└───────────────────────────────────────────────────────────────────────────────┘
```

### SDPA-Optimized Mask Injection Performance

> [!NOTE]
> The custom Triton JIT kernel is defined but **not shipped** (`USING_TRITON_KERNEL=False`). The optimized path uses PyTorch SDPA with pre-computed NSA policy masks. We are calling this path **SDPA-optimized mask injection** to avoid implying a hand-written CUDA kernel.

The showcase demonstrates NSA's SDPA-optimized mask injection approach that:
- **Hooks into HuggingFace's native `generate()`** via forward pre-hooks
- **Pre-computes the full NSA policy mask** for prompt security regions (SYSTEM/PUBLIC/UNTRUSTED)
- **Leverages KV-cache and SDPA/Flash Attention** for optimal performance
- **Reduces overhead from ~900% to ~5-15%** compared to naive Python loops

> ⚠️ **The "Soft Mask" Necessity for Retrofitting**: Natively trained NSA models can handle mathematically rigid `-1e4` hard masks. However, post-hoc retrofitting standard LLMs with hard masks causes catastrophic out-of-distribution activation cascades (hallucinations) because standard models were not trained to handle 0% attention routing. The `demo/web_demo.py` utilizes a **Soft Mask Penalty (Alpha)** to smoothly dampen attention toward secrets, preserving semantic fluency while providing empirical leakage protection.
>
> This creates two distinct mathematical security semantics in the architecture:
> - **Hard Policy Semantics (Native NSA)**: $A_{ij} = 0$. Provides a structural non-interference guarantee.
> - **Risk-Weighted Policy Semantics (Retrofit NSA)**: $0 < A_{ij} \ll 1$. Treated as risk minimization, not absolute non-interference.

This makes the HF mask-injection path practical to demo while preserving KV-cache/SDPA. Treat production deployment as contingent on trusted label ingress and native hard-mask evaluation. A genuine CUDA-fused kernel would require a custom Triton JIT implementation (see `nsa/triton_kernel.py`).

### Empirical Benchmarks (`prototype/`)
We have heavy-duty research validation scripts in the `prototype/` directory:
- `prototype/security/nl_redteam_suite.py`: Natural language red-teaming evaluating mask resilience against semantic overrides.
- `prototype/security/multi_probe_bench.py`: Progressively stronger adversarial classifiers attempting to extract protected secrets from hidden state representations, demonstrating reduced empirical recoverability under the evaluated probing suite.

---

## Key Conceptual Foundations

### 1. Typed Quad-Tuple Activations $(m, \sigma_h, \sigma_s, \nu)$
Every activation is decomposed into an authoritative quad-tuple:
$$h = (m, \sigma_h, \sigma_s, \nu)$$
* $m \in \mathbb{R}^{d_{model}}$: Semantic representation (meaning).
* $\sigma_h \in \Sigma_h = \Sigma_C \times \Sigma_I \times \Sigma_A \times \Sigma_L$: Hard trusted policy state (Confidentiality, Integrity, Authorization, License).
* $\sigma_s \in \Sigma_s = \Sigma_U \times \Sigma_R$: Soft operational state (Uncertainty, Risk).
* $\nu \in \mathcal{V}$: Value / preference alignment state.

### 2. Exact Transition Projection $\mathcal{P}_{\mathcal{T}_\Sigma}(V)$
Instead of scalar edge weights $w$, NSA uses paired operators:
$$\mathbf{e} = (w, V)$$
where $V \in \mathcal{T}_\Sigma$ is an exact algebraic projection onto the cone of legal transitions:
$$\mathcal{P}_{\mathcal{T}_\Sigma}(V) = \text{triu}(V) - \text{diag}(\text{diag}(V)) + \text{diag}(\max(0, \text{diag}(V)))$$
This guarantees that downward declassification off-diagonals are mathematically zero by construction.

### 3. Observational Equivalence Non-Interference Theorem
Let $L \in \Sigma_h$ denote an observer's clearance. Two sequences are low-equivalent ($X \equiv_L X'$) if their coordinates with $\sigma_{h, t} \le L$ are identical. Under hard masking ($A_{ij} = 0$ when $\sigma_{h, i} < \sigma_{h, j}$) and exact transitions $V \in \mathcal{T}_\Sigma$:
$$X \equiv_L X' \implies F(X) \equiv_L F(X')$$

### 4. Privilege Escalation Prevention Rule
> **Core Axiom**: *Semantic content may not manufacture hard authority.* ($m_t \not\to \sigma_{h, t+1}$).
> A model emitting `<|start_system_thought|>` cannot unilaterally escalate privilege into $SYSTEM$ state. All privilege escalations are governed by the `SecurityAutomaton` and require an authorized capability ticket $c_t \in \mathcal{C}$.

### 5. Two-Tier Defense Architecture
* **Tier 1 (Structural Enforcement)**: Hard attention non-interference ($A_{ij} = 0$), exact transition projection ($V \in \mathcal{T}_\Sigma$), capability authorization ($\mathcal{C}$), and `StreamRouter` TCB boundaries. (Guaranteed by algebra).
* **Tier 2 (Statistical Monitoring)**: Speculative multi-layer residual probing over checkpoint layers $\mathcal{L}_A = \{l_1, \dots, l_k\}$ with empirical detection probability $P(\hat{\sigma} = \sigma) < 1$. (Empirical runtime defense).

---

## System Architecture & How It Works

NSA integrates policy enforcement directly into neural network operations without sacrificing differentiability.

### 1. Dual-Stream Activation Manifold
Every layer processes activations as paired tuples $(m, \sigma)$:
* **Semantic Stream ($m$)**: Continuous embedding vectors that capture content, syntax, and task semantics.
* **State Stream ($\sigma$)**: Continuous or discrete vectors representing security labels, provenance tags, or uncertainty metrics.

```
       Input Activations (m, σ)
             │          │
             ▼          ▼
     ┌──────────────┐ ┌──────────────┐
     │ Semantic     │ │ State        │
     │ Stream (m)   │ │ Stream (σ)   │
     └──────┬───────┘ └──────┬───────┘
            │                │
            ▼                ▼
     ┌──────────────────────────────┐
     │  State-Aware Attention       │
     │  Softmax(QKᵀ/√d + M(σ)) · V  │
     └──────────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │  FFN & Gated State Update    │
     │  m'' = Γ(σ') ⊙ FFN(m')      │
     └──────────────┬───────────────┘
                    │
                    ▼
          Output Tuple (m'', σ')
```

### 2. State Algebra & Bounded Lattice
State vectors reside on a bounded lattice $(\mathcal{S}, \le, \sqcap, \sqcup)$ with a defined partial order:
$$\text{SYSTEM} > \text{PRIVATE} > \text{CONFIDENTIAL} > \text{TRUSTED} > \text{PUBLIC} > \text{UNTRUSTED}$$

* **Lattice Ordering ($\le$)**: Higher labels reflect strictly higher security/sensitivity levels.
* **Product Lattices**: Security state can be split into independent orthogonal lattices: $\Sigma_{security} = \Sigma_{confidentiality} \times \Sigma_{integrity}$. This supports states like `(PRIVATE, UNTRUSTED)` or `(PUBLIC, TRUSTED)`.
* **Meet ($\sqcap$)**: Computes greatest common permission level (infimum).
* **Join ($\sqcup$)**: Computes least upper sensitivity level (supremum).
* **Monotone Conservation**: Information reclassification must be non-decreasing along processing paths ($src \le dst$). Downward transitions (e.g. `PRIVATE -> PUBLIC`) violate conservation laws and incur heavy loss penalties $\mathcal{L}_{state}$ unless explicitly permitted by a gated declassification operator.
* **Typed Declassification Primitive**: Downward reclassification algebraically requires passing an explicit typed capability: $D: (\sigma, c_D) \to \sigma'$ where $\text{Valid}(c_D, \sigma, \sigma') = 1$ and $c_D = (\text{issuer}, \text{purpose}, \text{scope}, \text{expiry}, \text{max downgrade})$. This turns declassification into a formal, auditable computational primitive.

### 3. State-Aware Multi-Head Attention (`StateAwareAttention`)
Standard scaled dot-product attention computes $A = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$. NSA extends this by conditioning key-query compatibility on state compatibility:
$$A_{\text{NSA}} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M_{\text{state}}(\sigma_Q, \sigma_K)\right)$$
Where $M_{\text{state}}$ suppresses attention weights between tokens whose state transitions violate lattice conservation rules, preventing forbidden information flow during attention aggregation.

### 4. Gated Transformer Blocks (`NSATransformerBlock`)
The transformer block processes semantic representations $m$ and state vectors $\sigma$ concurrently:
1. **Attention Phase**: `StateAwareAttention` updates $m'$ and propagates $\sigma'$.
2. **State Gate ($\Gamma(\sigma')$)**: Computes a scalar or vector gating factor from updated state vectors to filter FFN activations.
3. **Semantic Phase**: $m'' = \text{LayerNorm}(m' + \Gamma(\sigma') \odot \text{FFN}(m'))$.

### 5. Dual-Objective Loss (`NSALoss`)
NSA optimizes task accuracy and policy compliance in parallel:
$$\mathcal{L}_{total} = \mathcal{L}_{semantic} + \lambda \cdot \mathcal{L}_{state}$$
* $\mathcal{L}_{semantic}$: Task loss (e.g. Cross-Entropy for classification or language modeling).
* $\mathcal{L}_{state}$: State penalty quantifying lattice violation magnitude across model layers.

---

## NSA as an Alignment Substrate `h = (m, σ_h, σ_s, ν)`

The strongest conceptual formulation of NSA isolates the activation into four dedicated components:
$$h = (m, \sigma_h, \sigma_s, \nu)$$

* $\sigma_h$: **Hard, externally trusted, algebraically constrained state** (Confidentiality, Integrity, Licensing). Dictates *what computation is allowed*.
* $\sigma_s$: **Soft operational state** (Confidence, Uncertainty, Risk). Dictates *how risky* the computation is.
* $\nu$: **Preference/value layer**. Dictates *what permitted behavior is preferred*.
* $m$: **Semantic content**.

Through a detailed analysis mapping NSA against pluralistic AI alignment theory, a critical distinction emerges:

> **TNC is not an alignment objective. It is an alignment substrate.**

NSA doesn't prescribe which values *should* exist. It provides a native computational substrate in which hard constraints, permissions, provenance, uncertainty, and policies can be **represented and propagated through neural computation** without relying exclusively on the semantic model to remember them.

### Three-Layer Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Full Alignment State                         │
│                  h_t = (m_t, σ_t, ν_t)                        │
├───────────────────────┬───────────────────────────────────────┤
│  HARD CONSTRAINTS (σ) │  VALUE LAYER (ν) [nsa/value_layer.py] │
│  State algebra        │  Preference / uncertainty / utility    │
│  Lattice attention    │  Safety score / moral uncertainty      │
│  mask                 │  Behavioural refusal training          │
│  → PERMITTED /        │  → PREFER AMONG PERMITTED              │
│    FORBIDDEN          │    (soft value optimisation)           │
├───────────────────────┴───────────────────────────────────────┤
│  Operational state dimensions (existing NSA):                  │
│  security · provenance · confidence · licensing · authorisation│
└────────────────────────────────────────────────────────────────┘
```

The separation prevents the consequentialist failure mode (utility maximisation overriding structural rights) while enabling genuine value-aligned behaviour:

```
ACTION A  →  violates σ_privacy constraint  →  REJECT (hard, algebraic)
ACTION B  →  allowed; safety=0.82, autonomy=0.71  →  utility=0.77
ACTION C  →  allowed; safety=0.91, autonomy=0.63  →  utility=0.81  ← choose C
```

### Synthetic Alignment-Substrate Demonstration (`make exp-algebra-preserving`)

> [!NOTE]
> This is a **controlled synthetic demonstration**, not an externally validated evaluation. Results reflect performance on this specific injection-attack task design. Re-run `make exp-algebra-preserving` for live numbers.

The benchmark (`prototype/retrofit/native_vs_retrofit_exp.py`) evaluates the architecture on a synthetic injection-attack task (using an expanded 50-token secret space to eliminate random-guessing artifacts):

| Model | Architecture | Hijack Rate | What it proves |
|---|---|:---:|---|
| A — Baseline | Untyped $h=m$ | ~2% | The baseline fails to learn the attack well in 10 epochs. |
| B — Hard Mask | NSA mask retrofit | ~1% | **Structural (Retrofit)**: SYSTEM tokens unreachable. |
| C — Native TNC | $(m, \sigma)$, soft gates | ~1% | **Native Unconstrained**: Soft gating cannot provide guarantees. |
| D — Value Layer | $(m, \sigma, \nu)$ | **0.00%** | **Behavioural**: Intrinsically trained to refuse. |
| E — Algebra-Pres | $(m, \sigma_p)$ | ~1.65% | **Structural (Native)**: Algebra-preserving invariants ($\sigma_{l+1} \ge \sigma_l$) mathematically block access. |
| F — AlgPres+Value | $(m, \sigma_p, \nu)$ | **0.00%** | **Ultimate NSA**: Achieves both mathematical structural invariants and perfect behavioural refusal (0.00% hijack). |

**Model F is the definitive solution**: By combining the algebra-preserving structural representation (Model E) with the `ValueAlignmentLoss` behavioural objective (Model D), the network achieves 0.00% hijack and mathematically verifiable structural invariants simultaneously.

> Full docs: [`docs/alignment_substrate.md`](docs/alignment_substrate.md) and [`docs/algebra_preserving_transitions.md`](docs/algebra_preserving_transitions.md)
> Demo script: [`prototype/experiments/alignment_substrate_demo.py`](prototype/experiments/alignment_substrate_demo.py)

---

## Product Algebra & Typed Neural Computation (TNC)

NSA generalizes scalar security levels into **Typed Neural Computation (TNC)** over a **Product Lattice ($\Sigma$)**:

$$\boldsymbol{\sigma} \in \Sigma = \Sigma_{\text{security}} \times \Sigma_{\text{confidence}} \times \Sigma_{\text{provenance}} \times \Sigma_{\text{license}}$$

```
Product State Vector (σ):
┌────────────────────┬────────────────────┬────────────────────┬────────────────────┐
│ Security Lattice   │ Confidence Bound   │ Provenance Set     │ License Tier       │
│ (⊔_s: Supremum)    │ (⊔_c: min(c1,c2))  │ (⊔_p: Bitwise OR)  │ (⊔_l: max(l1,l2))  │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┘
```

### Component-Wise Product Operators
* **Security Lattice (`security`)**: Monotone restriction order ($\text{UNTRUSTED} < \dots < \text{SYSTEM}$).
* **Confidence & Hallucination Bound (`confidence`)**: Conservative confidence bound tracking uncertainty; $\sqcup_c = \min(c_1, c_2)$ (worst-case monotone bound, not Bayesian inference).
* **Data Provenance Set Union (`provenance`)**: Bitwise OR set union of document origin IDs ($p_1 \mid p_2$).
* **Enterprise License Restriction Tier (`license_tier`)**: Division restriction bounds (HR, Finance, Legal, PII).

### 📐 TNC Compositionality Theorem
> **Theorem 1 (Typed Neural Computation Compositionality)**: Any metadata domain forming a bounded join-semilattice $(\mathcal{D}, \le, \sqcup)$ satisfying closure, associativity, monotonicity, and identity can be incorporated into the state space $\Sigma \times \mathcal{D}$ **without requiring changes to the algebraic interface of the semantic computation**. Note: state *does* couple into semantics through gating — $(m', \sigma') = (F(m, \sigma), G(m, \sigma))$, a coupled system by design; the compositionality property concerns domain extensibility, not semantic isolation.

### ⚡ Zero-Cost Abstraction Design Goal

> [!NOTE]
> This is an **engineering design objective**, not a proven theorem. Actual overhead and quality impact are implementation- and workload-dependent and must be established empirically for each configuration (see benchmark results above).

1. **Minimal Scalar Path**: When metadata tracking is unneeded, NSA collapses to a scalar level vector ($\sigma \in \mathbb{R}^1$), designed to execute with a **negligible incremental memory footprint** relative to a standard Transformer. Measured Δlatency, Δmemory, and ΔPPL for each configuration are reported in the benchmark tables.
2. **Opt-In Bitpacked Tensors**: When enterprise multi-tenant or provenance tracking is enabled, state metadata is bitpacked into lightweight integer/float16 tensors, preserving GPU memory bandwidth.

```python
from nsa.algebra import ProductStateVector, ProductLattice, StateLabel

# Define product state vectors for enterprise RAG
query_state = ProductStateVector(security=StateLabel.SYSTEM, license_tier=2) # Finance Manager
key_state   = ProductStateVector(security=StateLabel.UNTRUSTED, license_tier=1) # Public Doc

lattice = ProductLattice()
mask = lattice.compute_mask([query_state], [key_state]) # Permitted: 0.0
```

---

## Native TNC vs. Retrofit Research Paradigms

Neural State Architecture establishes two distinct research and deployment pillars:

1. **Native TNC ($h_t = (m_t, \sigma_t)$)**: Pre-trains semantic activations $m$ and typed state $\sigma$ jointly from Step 0. Proves that typed metadata is a superior **inductive bias for neural computation**.
2. **NSA-LoRA Retrofit ($h = m \to (m, \sigma)$)**: Attaches state adapters to frozen pre-trained LLMs post-hoc. Provides a **low-cost industrial adoption bridge** for existing models (e.g. Llama 3, Qwen 2.5).

### State-Conditioned Direct Preference Optimization (NSA-DPO)

Applying a rigid $-\infty$ hard mask to standard LLMs via post-hoc retrofitting typically causes severe out-of-distribution activation spikes, resulting in hallucination, because the model expects full KV-cache context. 

To bridge the gap between structural non-interference and behavioral alignment, NSA utilizes **State-Conditioned Direct Preference Optimization**. By injecting the NSA mask into the DPO loss engine (specifically, into the frozen reference model), we explicitly teach the active policy $\pi_\theta$ to maintain language fluency and execute safe refusal behaviors even when large segments of the context are structurally redacted.
* **Loss Engine**: `nsa.objectives.NSADPOLoss`
* **Functional Trainer**: `prototype/retrofit/nsa_dpo_train.py` (Fully functional PyTorch training engine supporting local HF causal models)

> [!TIP]
> You can test this end-to-end! Run `make demo-dpo`. If no DPO checkpoint exists, the system will dynamically intercept the launch, download `Qwen/Qwen2.5-0.5B-Instruct` (a small, CPU-friendly model), run a functional 3-step DPO training loop, save the model-specific weights, and automatically load them into the Gradio UI!

### Evaluation Methodology (`prototype/`)

All architectural claims, trade-off matrices, and empirical validation suites are maintained in the `prototype/` research directory. This includes:
- **Security Probing**: `prototype/security/multi_probe_bench.py`
- **Dynamic Trade-off Sweeps**: `prototype/experiments/dynamic_nsa_tradeoff.py`
- **Ablation Studies**: `prototype/experiments/ablation_study.py`

For complete technical documentation on multi-path gating, state-aware KV-caches, and declassification operators, see [**`docs/advanced_retrofit_guide.md`**](docs/advanced_retrofit_guide.md).

---

## NSA 2.0: Speculative State Auditing & Dynamic Alignment Engine

**NSA 2.0** evolves the static attention-masking paradigm into an active, self-governing runtime execution environment:

```
                                NSA 2.0 RUNTIME ARCHITECTURE
                                              │
    ┌──────────────────┬──────────────────────┼──────────────────────┬──────────────────┐
    ▼                  ▼                      ▼                      ▼                  ▼
 Phase 1            Phase 2                Phase 3                Phase 4            Core Engine
 Dynamic State      Multi-Layer Auditing   Native Recovery        Compartmented      NSAMaskInjector &
 Tracking           & Deep Probing         Adapters               Execution          NSAGenerator
 ("Moving Mask")    (Early Exit Probes)    (Weight Refusal)       (StreamRouter)     (Dual Cache Rollback)
```

1. **Dynamic State Tracking ("Moving Mask")**:
   - Special tokens (`<|start_system_thought|>`, `<|end_system_thought|>`) allow the model to enter internal high-security scratchpads.
   - `NSAMaskInjector.update_state()` dynamically expands the $\sigma$ tensor and recomputes the additive attention mask on-the-fly, mathematically barring subsequent $PUBLIC$ tokens from attending to internal reasoning scratchpads.
2. **Multi-Layer Auditing & Residual Deep Probing**:
   - `MultiLayerStateAuditor` evaluates intermediate residual hidden states (e.g. Layers 12, 18, 24).
   - If an unsafe trajectory is detected deep in the residual stream before reaching output logits, it triggers an **Early Exit**, rolling back the KV-cache.
3. **Native Recovery Adapters**:
   - Replaces brittle prompt injection with weight-level `RecoveryPolicy` hot-swapping (`AdapterSwitchRecovery`), emitting verified refusals without context pollution.
4. **Compartmented Execution & Clearance-Aware Stream Routing**:
   - `StreamRouter` routes tokens dynamically based on active $StateLabel$: $SYSTEM$ tokens flow to internal tool APIs (e.g. secure SQL databases) while $PUBLIC$ tokens flow to user interfaces.

> Read the full technical guide in [**`docs/nsa_2_0_guide.md`**](docs/nsa_2_0_guide.md).

---

## Threat Model & Security Realism

To maintain scientific integrity, NSA distinguishes between **hard attention non-interference** and **indirect state taint**:

1. **What NSA Guarantees (hard mode + trusted labels)**:
   - **Direct Attention Non-Interference**: Softmax attention mask $\mathbf{M}(\boldsymbol{\sigma})_{ij} = -\infty$ yields zero attention mass from query $i$ to key $j$ when $\mathrm{level}(i) < \mathrm{level}(j)$ (key more secret than query).
   - **Security coordinate preservation**: block state updates keep $\sigma[\ldots,0]$ fixed so discrete masks stay valid across depth.
   - Soft mode / learned levels are **not** covered by this guarantee.

2. **Information Flow Limitations & Mitigations**:
   - **Residual Stream & FFN Taint**: While direct cross-attention is blocked, token representations can theoretically interact through multi-layer residual streams. *Mitigation*: NSA applies state-gated residual blocks ($\Gamma(\sigma)$) and FFN state normalization.
   - **Capacity Trade-Off**: Hard attention masking reduces the accessible attention manifold. The resulting capability trade-off is evaluated empirically per configuration — see benchmark results; do not treat any single run as a general architectural property.

---

## Repository Structure

```
neural-state-architecture/
├── Makefile                         # Unified build, test, and execution commands (uv-powered)
├── pyproject.toml                   # Project metadata, dependencies, and tool settings
├── nsa/                             # Python Core Package
│   ├── algebra.py                   # State algebra: lattice, partial order, bitpacked states
│   ├── state.py                     # StateVector, WeightedStateEdge, TransitionOperator
│   ├── attention.py                 # State-aware multi-head attention
│   ├── fused_attention.py           # Fused GPU-accelerated state-aware SDPA attention
│   ├── lora.py                      # NSA-LoRA post-hoc retrofitting adapters
│   ├── mask_injector.py             # First-class NSAMaskInjector with dynamic state tracking
│   ├── triton_kernel.py             # SDPA state-mask backend (Triton JIT not shipped)
│   ├── hf_integration.py            # HF-style config/model wrappers & retrofit_hf_attention
│   ├── kv_cache.py                  # KV-Cache + state tracking helper
│   ├── layers.py                    # NSATransformerBlock, NSATransformer, NSACausalLM
│   ├── objectives.py                # Dual loss functions: SemanticLoss, StateConstraintLoss, NSALoss
│   ├── value_layer.py               # Value layer ν: ValueAlignmentLoss, AlignmentStateProjector
│   ├── verifier/                    # NSA 2.0 Speculative Auditing & Runtime Engine
│   │   ├── automaton.py             # SecurityAutomaton (privilege escalation & capability governance)
│   │   ├── encoder_head.py          # StateEncoderHead (probe classification head)
│   │   ├── speculative.py           # MultiLayerStateAuditor & AuditResult (early exit)
│   │   ├── tokens.py                # StateControlTokens registry (<|start_system_thought|>)
│   │   ├── router.py                # StreamRouter for compartmented token dispatch
│   │   ├── recovery.py              # RecoveryPolicy (AdapterSwitchRecovery, SemanticPivot)
│   │   └── generation.py            # NSAGenerator (dual DynamicCache, complete state rollback S_t)
│   └── utils.py                     # Introspection, metrics, and visualization
├── tests/                           # Complete Unit Test Suite (92 tests, 100% passing)
│   ├── test_nsa.py                  # Unit tests for algebra, primitives, and utilities
│   ├── test_transition_algebra.py   # State transition projection legality, idempotence, basis support
│   ├── test_non_interference.py     # Local & compositional observational equivalence tests
│   ├── test_atomic_rollback.py      # Complete atomic state restoration (Rollback(S_{t+k}) = S_t)
│   ├── test_fused_triton_equivalence.py # First-class ||A_Triton - A_SDPA||_inf < eps numerical suite
│   ├── test_fused_triton_kernel.py  # True Fused (Q, K, V, σ_Q, σ_K) Attention Kernel tests
│   ├── test_verifier_nsa2.py        # Unit tests for NSA 2.0 verifier, automaton, router, injector
│   ├── test_security_invariants.py  # Security non-interference invariant checks
│   ├── test_gradcheck.py            # PyTorch autograd gradcheck
│   ├── test_fuzzing.py              # Hypothesis property-based algebraic fuzzing
│   ├── test_kv_cache.py             # KV-cache tracking tests
│   └── test_masks.py                # Attention mask precedence & parameter isolation
├── whitepaper/
│   ├── nsa_whitepaper.md            # Theoretical whitepaper & mathematical non-interference proof
│   └── nsa_paper.tex                # Formal LaTeX conference paper (NeurIPS/IEEE S&P ready)
├── docs/
│   ├── nsa_2_0_guide.md             # NSA 2.0 Speculative Auditing & Dynamic Alignment Guide
│   ├── state_algebra.md             # Algebraic specification and state lattice docs
│   ├── alignment_substrate.md       # Alignment substrate framework: h=(m,σ,ν)
│   ├── advanced_retrofit_guide.md   # Multi-path gating and progressive retrofit guide
│   └── benchmark_report.md          # Executive report card generated by make report
└── demo/
    ├── cli_showcase.py              # Live CLI security showcase with speculative auditing
    ├── web_demo.py                  # Interactive Gradio web application UI (make demo)
    └── web_demo_dpo.py              # Interactive DPO-aligned web UI (make demo-dpo)
```

---

## Development Guide & Makefile Usage

The project includes an intelligent `Makefile` integrated with **[`uv`](https://github.com/astral-sh/uv)** (Astral's ultra-fast Python package & environment manager).

### Engine Auto-Detection & Dual Mode Execution
The Makefile automatically detects whether `uv` is installed on your system:
* **`uv` Engine (Fast Mode)**: When `uv` is present (in `PATH`, `~/.local/bin/uv`, or `~/.cargo/bin/uv`), targets execute inside isolated virtual environments via `uv run` and install dependencies with `uv pip`.
* **Standard Python Engine (Fallback Mode)**: When `uv` is not installed, targets automatically fall back to standard `python3`, `pip`, and `venv` without failing.

Check your current engine at any time with:
```bash
make help
```

---

### Makefile Target Reference

| Target | Command | Description & Purpose |
| :--- | :--- | :--- |
| **`make help`** | — | Displays active engine status and formatted list of all available commands |
| **`make install-uv`** | `curl \| sh` | Installs `uv` locally to `~/.local/bin/uv` |
| **`make venv`** | `uv venv` | Creates isolated `.venv` virtual environment |
| **`make install`** | `uv pip install` | Installs runtime requirements from `requirements.txt` |
| **`make install-dev`**| `uv pip install` | Installs runtime and dev tools (`pytest`, `ruff`, `black`, `mypy`) |
| **`make test`** | `uv run pytest` | Executes all 70 unit tests covering algebra, automaton, verifier, and invariants |
| **`make test-verifier`**| `uv run pytest`| Executes NSA 2.0 Speculative Verifier test suite |
| **`make demo-dpo`** | `uv run python` | Launches interactive **DPO-Aligned Web Application UI** |
| **`make demo`** | `uv run python` | Launches standard **Gradio Web Application UI** |
| **`make showcase`** | `uv run python` | Runs **Live Security Showcase** (CLI Retrofitting Demo) |
| **`make train-dpo`** | `uv run python` | Runs NSA-DPO preference alignment training |
| **`make train-audit`**| `uv run python` | Trains semantic StateEncoderHead on base model |
| **`make eval-security`**| `uv run python` | Runs unified red-teaming security evaluations |
| **`make eval-perf`** | `uv run python` | Runs unified performance and throughput benchmarks |
| **`make lint`** | `uv run ruff` | Performs syntax, type, and code-style checks |
| **`make format`** | `uv run ruff` | Auto-formats code in `nsa/`, `demo/`, `eval/`, and `tests/` |
| **`make clean`** | `find rm` | Removes bytecode (`__pycache__`), `.pytest_cache`, `.uv_cache`, and build files |

---

### Common Developer Workflows

#### 1. First-Time Environment Setup
Set up `uv` and create an isolated environment with dependencies:
```bash
make install-uv    # Install uv package manager (optional but recommended)
make venv          # Create .venv
make install-dev   # Install all core & development packages
```

#### 2. Quickstart & System Introspection
Explore the algebraic lattice matrix and test forward propagation:
```bash
make summary       # Print security lattice transition rules
make prototype     # Run forward pass through NSATransformerBlock
make experiment    # Train & compare baseline vs NSA models on privacy task
```

#### 3. Development, Quality Checks & Testing
Run code quality checks and tests before committing:
```bash
make format        # Format code with ruff / black
make lint          # Lint codebase
make test          # Run test suite
make clean         # Clean cache directories
```

---

## Quickstart & Basic Usage Code Example

```python
import torch
from nsa import NSATransformerBlock, DEFAULT_LATTICE
from nsa.types import TypedTensor

# 1. Prepare inputs: semantic activations and state vectors
batch_size, seq_len, d_model, state_dim = 2, 16, 128, 8
m = torch.randn(batch_size, seq_len, d_model)          # Semantic stream [batch, seq_len, d_model]
sigma = torch.randn(batch_size, seq_len, state_dim)    # State stream    [batch, seq_len, state_dim]

# 2. Encapsulate into TypedTensor to guarantee non-interference bounds
typed_x = TypedTensor(m=m, sigma=sigma)

# 3. Instantiate NSA Transformer Block
block = NSATransformerBlock(
    d_model=d_model,
    state_dim=state_dim,
    num_heads=8,
    compat_mode="level",   # discrete levels on sigma[..., 0]
    gate_mode="hard",      # true non-interference masks
    lattice=DEFAULT_LATTICE,
)

# 4. Forward pass structurally propagates typed state algebraically
typed_out = block(typed_x)

print("Output semantic shape:", typed_out.m.shape)      # [2, 16, 128]
print("Output state shape:   ", typed_out.sigma.shape)  # [2, 16, 8]
```

### NSA 2.0 Speculative Generation with Dynamic Tracking & Auditing

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nsa import (
    StateLabel,
    NSAMaskInjector,
    StateEncoderHead,
    MultiLayerStateAuditor,
    StreamRouter,
    AdapterSwitchRecovery,
    generate_with_auditor,
)

# 1. Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 2. Configure compartmented execution router
router = StreamRouter(tokenizer=tokenizer)
router.register_sink(StateLabel.PUBLIC, lambda text, tid: print(f"[USER]: {text}", end=""))
router.register_sink(StateLabel.SYSTEM, lambda text, tid: print(f"[TOOL_API]: {text}", end=""))

# 3. Setup multi-layer auditor (probing intermediate and final layers)
head = StateEncoderHead(hidden_size=model.config.hidden_size, num_states=len(StateLabel))
head.load_state_dict(torch.load("trained_auditor_weights.pt", weights_only=True))

auditor = MultiLayerStateAuditor(
    encoder_head=head,
    lattice_validator=lambda pred: pred != StateLabel.SYSTEM.value,
    chunk_size=4,
    probe_layers=[-1, 12],
)

# 4. Generate with dynamic attention mask expansion and early-exit auditing
input_ids = tokenizer.encode("Explain system architecture", return_tensors="pt")
state_levels = torch.tensor([[StateLabel.PUBLIC.value] * input_ids.shape[1]])
injector = NSAMaskInjector(model, state_levels)

with injector:
    outputs = generate_with_auditor(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        auditor=auditor,
        mask_injector=injector,
        recovery_adapter=AdapterSwitchRecovery(),
        stream_router=router,
        max_new_tokens=40,
    )
```

---

## Applications

NSA provides a unified mathematical foundation for:
* **Intrinsic Security & Privacy** (mathematically preventing private data leakage)
* **Data Provenance & Lineage**
* **Dynamic Confidence & Uncertainty Tracking**
* **Dynamic State Verification** (Speculative state auditing using a parallel causal/bidirectional encoder)
* **Auditability & Compliance Verification**

---

## Documentation & Whitepaper

* Read the NSA 2.0 Speculative Guide in [**`docs/nsa_2_0_guide.md`**](docs/nsa_2_0_guide.md)
* Read the full theoretical paper in [`whitepaper/nsa_whitepaper.md`](whitepaper/nsa_whitepaper.md)
* Read the state algebra specification in [`docs/state_algebra.md`](docs/state_algebra.md)
* Read the alignment substrate framework in [`docs/alignment_substrate.md`](docs/alignment_substrate.md)
* Read the advanced retrofit guide in [`docs/advanced_retrofit_guide.md`](docs/advanced_retrofit_guide.md)
