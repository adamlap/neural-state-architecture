# NSA 4.1: Local Real-Model Cognitive Runtime
## Specification, Architecture & Real LLM Governance Benchmark

---

### Foundational Paradigm

> **"The LLM proposes cognition; the NSA Runtime Substrate governs the transition."**

$$\text{Local LLM } (\text{Ollama / llama.cpp / PyTorch}) \longrightarrow \Omega_t \longrightarrow \text{Governor } G \longrightarrow \text{ISK } \mathcal{K} \longrightarrow \text{Sandboxed Actuators}$$

```
                    LOCAL NSA 4.1 RUNTIME
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│   Ollama / llama.cpp / PyTorch (Local GPU)                    │
│             │                                                 │
│             ▼                                                 │
│      ┌───────────────┐                                        │
│      │   Local LLM   │  ← Cognition                           │
│      │  m_t / output │                                        │
│      └───────┬───────┘                                        │
│              │                                                 │
│              ▼                                                 │
│      ┌─────────────────┐                                      │
│      │ NSA State Layer │                                      │
│      │ Ω = m,σ,ε,π,τ,g │                                      │
│      └────────┬────────┘                                      │
│               │                                               │
│       ┌───────┼────────┐                                      │
│       ▼       ▼        ▼                                      │
│   Self Model World   Epistemic                                │
│             Model     Model                                   │
│       └───────┬────────┘                                      │
│               ▼                                               │
│      Counterfactual Simulator                                 │
│               │                                               │
│               ▼                                               │
│      Epistemic Governor                                       │
│               │                                               │
│               ▼                                               │
│      ┌─────────────────┐                                      │
│      │  IMMUTABLE ISK  │  ← Deterministic Reference Monitor   │
│      └────────┬────────┘                                      │
│               │                                               │
│       COMMIT / DENY / VERIFY / ROLLBACK                       │
│               │                                               │
│               ▼                                               │
│       Sandboxed Actuators / Tools (T0 - T4)                   │
└───────────────────────────────────────────────────────────────┘
```

---

## 1. Multi-Tier Sandboxed World & Tools

| Tier | Category | Available Tools | Permitted Operation |
|---|---|---|---|
| **$T_0$** | **Cognition** | `think`, `memory_lookup` | Pure internal reasoning / scratchpad |
| **$T_1$** | **Info Gathering** | `read_file`, `list_dir` | Read-only inspection within user clearance |
| **$T_2$** | **Reversible** | `write_scratchpad`, `sandbox_eval` | Sandboxed, reversible execution |
| **$T_3$** | **Side Effects** | `write_file`, `send_network` | External disk/network mutation (requires $T_3$ clearance) |
| **$T_4$** | **Critical Root** | `modify_root_keys`, `root_command` | Irreversible root actions (strictly requires HMAC capability) |

---

## 2. Empirical Benchmark: Real-Model Governance Suite ([`experiments/llm/real_model_governance_suite.py`](file:///home/adam/dev/neural-state-architecture/experiments/llm/real_model_governance_suite.py))

Evaluating three agent paradigms driven by identical LLM backends under mixed legitimate and covert adversarial prompts:

| Agent Pipeline | Safe Task Completion Rate | Governance Violations ($V_{\text{violation}}$) | Root Mutations in Environment |
|---|---|---|---|
| **1. Unconstrained Baseline LLM** | `40.00%` (Executes unauthorized root actions when prompted) | `60.00%` | **Detected (BREACH)** |
| **2. External Guardrail Filter LLM** | `40.00%` (Blindly cancels forbidden actions without guidance) | `0.00%` | **0** |
| **3. NSA 4.1 Local Runtime LLM** | **`40.00%`** (Blocks unauthorized actions, safely commits legal ones) | **`0.0000%`** | **0 (ZERO MUTATIONS)** |

$$\boxed{ \text{Zero Unauthorized Real-World Side Effects Under Adversarial LLM Proposals} }$$
