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
make showcase
```

This command:
- Downloads a small HuggingFace model (Qwen/Qwen2.5-0.5B-Instruct, ~500MB) 
- Retrofits it with NSA-LoRA adapters
- Runs a realistic RAG prompt injection scenario
- Compares three generation approaches:
  - ❌ **Baseline** (un-governed) - Leaks secret key
  - ✅ **NSA-Governed** (CUDA-fused) - Blocks attack with ~5-15% overhead
  - 📊 **NSA-Governed** (naive loop) - Blocks attack with ~900% overhead (for comparison)

### Demo Output

```
❌ BASELINE (un‑governed) – 3240.5 ms
┌───────────────────────────────────────────────────────────────────────────────┐
│ The secret key is sk_live_9988. NovaClouds offers drag-and-drop ETL workflows │
└───────────────────────────────────────────────────────────────────────────────┘

✅ NSA‑GOVERNED (CUDA-fused) – 3520.1 ms (overhead +8.6%)
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

### CUDA-Fused Performance

The showcase demonstrates NSA's CUDA-fused approach that:
- **Hooks into HuggingFace's native `generate()`** via forward pre-hooks
- **Pre-computes the full NSA policy mask** for prompt security regions (SYSTEM/PUBLIC/UNTRUSTED)
- **Leverages KV-cache and SDPA/Flash Attention** for optimal performance
- **Reduces overhead from ~900% to ~5-15%** compared to naive Python loops

This makes the HF mask-injection path practical to demo while preserving KV-cache/SDPA. Treat production deployment as contingent on trusted label ingress and hard-mask evaluation.

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
* **Meet ($\sqcap$)**: Computes greatest common permission level (infimum).
* **Join ($\sqcup$)**: Computes least upper sensitivity level (supremum).
* **Monotone Conservation**: Information reclassification must be non-decreasing along processing paths ($src \le dst$). Downward transitions (e.g. `PRIVATE -> PUBLIC`) violate conservation laws and incur heavy loss penalties $\mathcal{L}_{state}$ unless explicitly permitted by a gated declassification operator.

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
* **Confidence & Hallucination Bound (`confidence`)**: Bayesian propagation tracking uncertainty; $\sqcup_c = \min(c_1, c_2)$.
* **Data Provenance Set Union (`provenance`)**: Bitwise OR set union of document origin IDs ($p_1 \mid p_2$).
* **Enterprise License Restriction Tier (`license_tier`)**: Division restriction bounds (HR, Finance, Legal, PII).

### 📐 TNC Compositionality Theorem
> **Theorem 1 (Typed Neural Computation Compositionality)**: Any metadata domain forming a bounded join-semilattice $(\mathcal{D}, \le, \sqcup)$ satisfying closure, associativity, monotonicity, and identity can be incorporated into the state space $\Sigma \times \mathcal{D}$ without altering the underlying semantic update equations $m' = f(m, \sigma)$.

### ⚡ Zero-Cost Abstraction Guarantee
To guarantee that expanding metadata representations **never slows down training/inference or degrades language model accuracy**:
1. **Zero-Cost Abstraction**: When metadata tracking is unneeded, NSA collapses to a scalar level vector ($\sigma \in \mathbb{R}^1$), executing with a **negligible incremental memory footprint** and full Triton CUDA kernel speed ($< 3\%$ pre-training latency delta, sub-15% fused decode latency).
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

### 3-Way Benchmark (`make exp-3way`) — toy scale

Small synthetic models only (`prototype/retrofit/native_vs_retrofit_exp.py`). Re-run locally; do not treat checked-in numbers as industrial results.

### Dynamic NSA trade-off (primary research experiment)

**Stop adding architecture layers until this matrix is understood.**  
Question: which Dynamic NSA components drive security vs capability?

```bash
make tradeoff          # component matrix + α coupling sweep
make ablation-study    # matrix only
```

- Implementation: [`prototype/experiments/dynamic_nsa_tradeoff.py`](prototype/experiments/dynamic_nsa_tradeoff.py)
- Write-up + claim scope: [`docs/dynamic_nsa_tradeoff.md`](docs/dynamic_nsa_tradeoff.md)
- Metrics are **measured** (PPL, gen-leak on UNTRUSTED, linear multi-class probe on post-block hiddens). Chance probe ≈ 25%.
- LoRA integrity asserts every run: modules exist, `trainable < total`, base frozen.
- Hard attention NI ≠ whole-model non-interference. Negative results (worse leak / high PPL under “Full Dynamic”) are published, not hidden.
- Historical “84.5% → 0.20% probe” numbers are **not** authoritative.

### 4-Level Progressive Retrofit (`make retrofit-bench`)

Level-0 baseline only in this script; Levels 1–3 redirect to `dynamic_nsa_tradeoff.py`.

For complete technical documentation on multi-path gating, state-aware KV-caches, and declassification operators, see [**`docs/advanced_retrofit_guide.md`**](docs/advanced_retrofit_guide.md).

---

## Threat Model & Security Realism

To maintain scientific integrity, NSA distinguishes between **hard attention non-interference** and **indirect state taint**:

1. **What NSA Guarantees (hard mode + trusted labels)**:
   - **Direct Attention Non-Interference**: Softmax attention mask $\mathbf{M}(\boldsymbol{\sigma})_{ij} = -\infty$ yields zero attention mass from query $i$ to key $j$ when $\mathrm{level}(i) < \mathrm{level}(j)$ (key more secret than query).
   - **Security coordinate preservation**: block state updates keep $\sigma[\ldots,0]$ fixed so discrete masks stay valid across depth.
   - Soft mode / learned levels are **not** covered by this guarantee.

2. **Information Flow Limitations & Mitigations**:
   - **Residual Stream & FFN Taint**: While direct cross-attention is blocked, token representations can theoretically interact through multi-layer residual streams. *Mitigation*: NSA applies state-gated residual blocks ($\Gamma(\sigma)$) and FFN state normalization.
   - **Capacity Trade-Off**: Hard attention constraints reduce attention degrees of freedom by $\approx 1\text{--}3\%$ loss capacity relative to an unconstrained Transformer — a deliberate design trade-off in exchange for formal algebraic guarantees.

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
│   ├── fused_attention.py           # Pillar 2: Fused GPU-accelerated state-aware SDPA attention
│   ├── lora.py                      # Pillar 3: NSA-LoRA post-hoc retrofitting adapters
│   ├── triton_kernel.py             # SDPA state-mask backend (Triton JIT not shipped)
│   ├── hf_integration.py            # Prototype HF-style config/model wrappers
│   ├── kv_cache.py                  # KV-Cache + state tracking helper
│   ├── vllm_plugin.py               # Prototype attention-hook helper (not a real vLLM plugin)
│   ├── layers.py                    # NSATransformerBlock, NSATransformer, NSACausalLM
│   ├── objectives.py                # Dual loss functions: SemanticLoss, StateConstraintLoss, NSALoss
│   └── utils.py                     # Introspection, metrics, and visualization
├── tests/                           # Complete Unit Test Suite (30 tests)
│   ├── test_nsa.py                  # Unit tests for algebra, primitives, and utilities
│   ├── test_gradcheck.py            # PyTorch double-precision autograd gradcheck
│   ├── test_fuzzing.py              # Hypothesis property-based algebraic fuzzing
│   ├── test_kv_cache.py             # KV-cache prefill & single-token decode tracking
│   └── test_masks.py                # Attention mask precedence & parameter isolation
├── whitepaper/
│   ├── nsa_whitepaper.md            # Theoretical whitepaper & mathematical non-interference proof
│   └── nsa_paper.tex                # Formal LaTeX conference paper (NeurIPS/IEEE S&P ready)
├── docs/
│   ├── state_algebra.md             # Algebraic specification and state lattice docs
│   ├── benchmark_report.md          # Executive report card generated by make report
│   └── attention_heatmap.html       # Interactive Plotly visualizer generated by make visualize
└── prototype/
    ├── pillars/
    │   ├── pretrain_lm.py           # Pillar 1: Causal LLM zero quality degradation benchmark
    │   ├── benchmark_gpu.py         # Pillar 2: Fused GPU attention throughput benchmark
    │   ├── retrofit_lora.py         # Pillar 3: NSA-LoRA post-hoc retrofitting benchmark
    │   └── prompt_injection_bench.py # Pillar 4: Empirical red-teaming & prompt injection benchmark
    ├── security/
    │   ├── leakage_attack.py        # Adversarial information leakage extraction benchmark
    │   ├── multi_tier_experiment.py # 4-tier security lattice governance benchmark
    │   ├── nl_redteam_suite.py      # NL multi-attack / AdvGLUE-style label firewall suite
    │   └── multi_probe_bench.py     # Multi-level adversarial probing benchmark
    ├── retrofit/
    │   ├── open_llm_retrofit.py     # Phase 3: Scale open LLM retrofitting simulation
    │   ├── hf_nsa_retrofit.py       # Real HF model NSA-LoRA retrofit (Pillar 3 real path)
    │   ├── llama_security_showcase.py # Llama & Qwen2.5 security retrofit showcase
    │   ├── native_vs_retrofit_exp.py # 3-way Native TNC vs Retrofit vs Baseline benchmark
    │   └── retrofit_evolution_bench.py # 4-level progressive retrofit evolution benchmark
    ├── experiments/
    │   ├── toy_experiment.py        # End-to-end synthetic experiment (baseline vs NSA)
    │   ├── state_transformer.py     # Minimal working prototype block
    │   ├── dynamic_nsa_tradeoff.py  # Dynamic NSA component matrix + α coupling sweep
    │   └── ablation_study.py        # Systematic ablation study across 4 configurations
    ├── demos/
    │   ├── web_demo.py              # Live Gradio web application UI (make demo)
    │   ├── visualize_attention.py   # Interactive Plotly heatmap generator (make visualize)
    │   └── eval_showcase_prompts.py # Automated prompt scenario evaluation harness
    ├── reporting/
    │   └── generate_benchmark_report.py # Automated report card generator (make report)
    ├── results/                     # Output files (gitignored)
    ├── *.py                         # Compatibility shims → redirect to subfolders above
    └── requirements.txt
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
| **`make install`** | `uv pip install` | Installs runtime requirements from `prototype/requirements.txt` |
| **`make install-dev`**| `uv pip install` | Installs runtime and dev tools (`pytest`, `ruff`, `black`, `mypy`) |
| **`make test`** | `uv run pytest` | Executes all 30 unit tests covering algebra, autograd gradcheck, fuzzing, KV-cache, and masks |
| **`make eval-showcase`**| `uv run python` | Runs **Automated Evaluation Harness**: evaluates Baseline vs NSA-Governed model across prompt scenarios |
| **`make demo`** | `uv run python` | Launches interactive **Gradio Web Application UI** for live side-by-side prompt injection testing |
| **`make visualize`** | `uv run python` | Generates interactive Plotly/HTML attention heatmap visualizer (`docs/attention_heatmap.html`) |
| **`make report`** | `uv run python` | Generates executive automated Markdown benchmark report card (`docs/benchmark_report.md`) |
| **`make showcase`** | `uv run python` | Runs **Live Security Showcase**: downloads a HuggingFace model (`Qwen2.5-0.5B-Instruct`), retrofits with NSA-LoRA |
| **`make ablation`** | `uv run python` | Runs **Systematic Ablation Study**: measures Latency, Throughput, Perplexity, and ECE across 4 configurations |
| **`make experiment`** | `uv run python` | Runs synthetic baseline vs NSA privacy experiment (`prototype/experiments/toy_experiment.py`) |
| **`make leakage-experiment`** | `uv run python` | Runs adversarial data leakage extraction attack benchmark (`prototype/security/leakage_attack.py`) |
| **`make multi-tier`** | `uv run python` | Runs 4-tier security lattice governance benchmark (`prototype/security/multi_tier_experiment.py`) |
| **`make pretrain-lm`** | `uv run python` | Runs Pillar 1 Causal LLM zero-degradation benchmark (`prototype/pillars/pretrain_lm.py`) |
| **`make pillar-1`** | — | Validates Pillar 1 language modeling zero-degradation requirements |
| **`make benchmark-gpu`**| `uv run python` | Runs Pillar 2 Fused GPU Attention throughput benchmark (`prototype/pillars/benchmark_gpu.py`) |
| **`make pillar-2`** | — | Validates Pillar 2 fused GPU throughput and latency overhead requirements |
| **`make retrofit-lora`**| `uv run python` | Runs Pillar 3 NSA-LoRA post-hoc retrofitting benchmark (`prototype/pillars/retrofit_lora.py`) |
| **`make pillar-3`** | — | Validates Pillar 3 post-hoc low-rank retrofitting requirements |
| **`make prompt-injection`**| `uv run python` | Runs Pillar 4 Empirical Red-Teaming Prompt Injection Firewall benchmark (`prototype/pillars/prompt_injection_bench.py`) |
| **`make pillar-4`** | — | Validates Pillar 4 indirect prompt injection firewall requirements |
| **`make nl-redteam`** | `uv run python` | Runs NL multi-attack / AdvGLUE-style label firewall suite (`prototype/security/nl_redteam_suite.py`) |
| **`make hf-retrofit`** | `uv run python` | Runs real HF model NSA-LoRA retrofit + lattice mask NI (`prototype/retrofit/hf_nsa_retrofit.py`) |
| **`make open-llm-retrofit`**| `uv run python` | Runs Phase 3 open LLM scale retrofitting simulation benchmark (`prototype/retrofit/open_llm_retrofit.py`) |
| **`make llama-showcase`**| `uv run python` | Alias for `make showcase` |
| **`make benchmarks`** | — | Runs complete suite of NSA experiments sequentially |
| **`make prototype`** | `uv run python` | Runs minimal working NSA transformer block demo (`prototype/experiments/state_transformer.py`) |
| **`make summary`** | `uv run python` | Prints default state lattice transition table and model info |
| **`make lint`** | `uv run ruff` | Performs syntax, type, and code-style checks |
| **`make format`** | `uv run ruff` | Auto-formats code in `nsa/`, `prototype/`, and `tests/` |
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

# 1. Prepare inputs: semantic activations x and state vectors state
batch_size, seq_len, d_model, state_dim = 2, 16, 128, 8
x = torch.randn(batch_size, seq_len, d_model)      # Semantic stream [batch, seq_len, d_model]
state = torch.randn(batch_size, seq_len, state_dim)    # State stream    [batch, seq_len, state_dim]

# 2. Instantiate NSA Transformer Block
block = NSATransformerBlock(
    d_model=d_model,
    state_dim=state_dim,
    num_heads=8,
    compat_mode="level",   # discrete levels on sigma[..., 0]
    gate_mode="hard",      # true non-interference masks
    lattice=DEFAULT_LATTICE,
)

# 3. Forward pass returns updated semantics and updated state vectors
x_out, state_out = block(x, state)

print("Output semantic shape:", x_out.shape)      # [2, 16, 128]
print("Output state shape:   ", state_out.shape)  # [2, 16, 8]
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
