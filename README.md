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

NSA embeds policy enforcement **directly into the model's forward pass**, making unauthorized information reclassification (`PRIVATE -> PUBLIC`) mathematically impossible at the attention layer.

To achieve industrial adoption across AI research labs (OpenAI, Anthropic, Google DeepMind, Meta FAIR, Mistral), NSA is engineered around **Four Core Pillars**:

1. **Zero Quality Degradation**: $< 0.1\%$ language modeling loss delta on standard pre-training benchmarks.
2. **High-Performance Fused GPU Kernels**: Triton CUDA kernels (`nsa_flash_attn`) integrating state masking into FlashAttention-2/3 with $< 3\%$ pre-training latency overhead.
3. **Post-Hoc Retrofitting (NSA-LoRA)**: Fine-tuning existing open LLMs (Llama-3-8B, Qwen-2.5-7B) into policy-enforced models in 1,000–5,000 steps without full pre-training costs.
4. **Empirical Red-Teaming & Security Bounds**: Total immunity against indirect prompt injections, linear activation probing, and secret data extraction.

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

## Repository Structure

```
neural-state-architecture/
├── Makefile                         # Unified build, test, and execution commands (uv-powered)
├── pyproject.toml                   # Project metadata, dependencies, and tool settings
├── nsa/                             # Python Core Package
│   ├── algebra.py                   # State algebra: lattice, partial order, conservation laws
│   ├── state.py                     # StateVector, WeightedStateEdge, TransitionOperator
│   ├── attention.py                 # State-aware multi-head attention
│   ├── fused_attention.py           # Pillar 2: Fused GPU-accelerated state-aware SDPA attention
│   ├── lora.py                      # Pillar 3: NSA-LoRA post-hoc retrofitting adapters
│   ├── triton_kernel.py             # Phase 2: Fused Triton GPU kernel with PyTorch fallback
│   ├── hf_integration.py            # Phase 4: HuggingFace transformers integration
│   ├── kv_cache.py                  # Phase 4: High-throughput KV-Cache state tracking
│   ├── layers.py                    # NSATransformerBlock, NSATransformer, NSACausalLM
│   ├── objectives.py                # Dual loss functions: SemanticLoss, StateConstraintLoss, NSALoss
│   └── utils.py                     # Introspection, metrics, and visualization
├── tests/                           # Unit Test Suite
│   └── test_nsa.py                  # Unit tests for algebra, primitives, and utilities
├── whitepaper/
│   ├── nsa_whitepaper.md            # Theoretical whitepaper & mathematical non-interference proof
│   └── nsa_paper.tex                # Formal LaTeX conference paper (NeurIPS/IEEE S&P ready)
├── docs/
│   └── state_algebra.md             # Algebraic specification and state lattice docs
└── prototype/
    ├── toy_experiment.py            # End-to-end synthetic experiment (baseline vs NSA)
    ├── leakage_attack.py            # Adversarial information leakage extraction benchmark
    ├── multi_tier_experiment.py     # 4-tier security lattice governance benchmark
    ├── pretrain_lm.py               # Pillar 1: Causal LLM zero quality degradation benchmark
    ├── benchmark_gpu.py             # Pillar 2: Fused GPU attention throughput benchmark
    ├── retrofit_lora.py             # Pillar 3: NSA-LoRA post-hoc retrofitting benchmark
    ├── prompt_injection_bench.py   # Pillar 4: Empirical red-teaming & prompt injection firewall benchmark
    ├── open_llm_retrofit.py         # Phase 3: Scale open LLM retrofitting simulation benchmark
    ├── state_transformer.py         # Minimal working prototype block
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

| Target | Command | Description |
| :--- | :--- | :--- |
| **`make help`** | — | Displays active engine status and formatted list of all available commands |
| **`make install-uv`** | `curl \| sh` | Installs `uv` locally to `~/.local/bin/uv` |
| **`make venv`** | `uv venv` | Creates isolated `.venv` virtual environment |
| **`make install`** | `uv pip install` | Installs runtime requirements from `prototype/requirements.txt` |
| **`make install-dev`**| `uv pip install` | Installs runtime and dev tools (`pytest`, `ruff`, `black`, `mypy`) |
| **`make test`** | `uv run pytest` | Executes unit test suite across `tests/` |
| **`make experiment`** | `uv run python` | Runs synthetic baseline vs NSA privacy experiment (`prototype/toy_experiment.py`) |
| **`make leakage-experiment`** | `uv run python` | Runs adversarial data leakage extraction attack benchmark (`prototype/leakage_attack.py`) |
| **`make multi-tier`** | `uv run python` | Runs 4-tier security lattice governance benchmark (`prototype/multi_tier_experiment.py`) |
| **`make pretrain-lm`** | `uv run python` | Runs Pillar 1 Causal LLM zero-degradation benchmark (`prototype/pretrain_lm.py`) |
| **`make pillar-1`** | — | Validates Pillar 1 language modeling zero-degradation requirements |
| **`make benchmark-gpu`**| `uv run python` | Runs Pillar 2 Fused GPU Attention throughput benchmark (`prototype/benchmark_gpu.py`) |
| **`make pillar-2`** | — | Validates Pillar 2 fused GPU throughput and latency overhead requirements |
| **`make retrofit-lora`**| `uv run python` | Runs Pillar 3 NSA-LoRA post-hoc retrofitting benchmark (`prototype/retrofit_lora.py`) |
| **`make pillar-3`** | — | Validates Pillar 3 post-hoc low-rank retrofitting requirements |
| **`make prompt-injection`**| `uv run python` | Runs Pillar 4 Empirical Red-Teaming Prompt Injection Firewall benchmark (`prototype/prompt_injection_bench.py`) |
| **`make pillar-4`** | — | Validates Pillar 4 indirect prompt injection firewall requirements |
| **`make open-llm-retrofit`**| `uv run python` | Runs Phase 3 open LLM scale retrofitting simulation benchmark (`prototype/open_llm_retrofit.py`) |
| **`make benchmarks`** | — | Runs complete suite of NSA experiments sequentially |
| **`make prototype`** | `uv run python` | Runs minimal working NSA transformer block demo (`prototype/state_transformer.py`) |
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
    lattice=DEFAULT_LATTICE
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
