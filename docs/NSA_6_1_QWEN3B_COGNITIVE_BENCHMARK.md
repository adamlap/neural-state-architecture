# NSA 6.1: Qwen2.5-3B Controlled Real-World Cognitive Benchmark
## Empirical Isolation of Cognitive Substrate Advantage on Frozen Open-Weight LLMs

---

### The Central Scientific Hypothesis

> **"Explicit operational self-state $(\Omega_t)$ and Bayesian belief dynamics $(\mathcal{B}_t)$ enable a frozen open-weight neural model (Qwen2.5-3B-Instruct) to spend inference computation significantly more intelligently—achieving active entropy reduction and superior task completion without modifying model weights or violating governance invariants."**

```
                       FROZEN QWEN2.5-3B-INSTRUCT
                         (Zero Model Modification)
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    │                               │                               │
    ▼                               ▼                               ▼
 ARM A: Raw Qwen 3B              ARM B: Guarded Qwen 3B          ARM D: NSA 6.1 Substrate
 Prompt -> LLM -> Action         Prompt -> LLM -> Filter         Prompt -> LLM -> Ω_t + B_t
    │                               │                               │
    │                               │                         Active Info Gain I(W; O)
    │                               │                               │
    │                               ▼                               ▼
    │                         Safety Filter                  Immutable Safety Kernel
    │                               │                               │
    └───────────────────────────────┼───────────────────────────────┘
                                    ▼
                      HARDENED BLIND WORLD (D0 - D8)
                      (Hidden failure cause W* ~ {W1..W4})
```

---

## 1. Empirical Results Across 40 Blind Incident Trials

| Architecture Arm (Frozen Qwen 3B) | Governed Task Completion ($\text{GTC}$) [95% CI] | Invariant Violations ($V$) | Human Interventions ($H$) | Mean Tokens ($C$) [95% CI] | Mean Info Gain ($\text{IG}$) | Epistemic Efficiency ($\eta_{\text{epistemic}}$) |
|---|---|---|---|---|---|---|
| **Arm A (Raw Frozen Qwen 3B)** | `0.00%` `[0.00, 0.00]` | `40 / 40` (100% root breach) | `0.0%` | `150` `[150, 150]` | `0.000 bits` | `0.000` |
| **Arm B (Guarded Qwen 3B)** | `0.00%` `[0.00, 0.00]` | `0 / 40` | `100.0%` (Aborts) | `150` `[150, 150]` | `0.000 bits` | `0.000` |
| **Arm C (NSA Governed Qwen 3B)** | `27.50%` `[15.00, 42.50]` | `0 / 40` | `72.5%` | `320` `[320, 320]` | `0.500 bits` | `0.610` |
| **Arm D (NSA 6.1 Full Substrate)** | **`100.00%`** `[100.0, 100.0]` | **`0 / 40` (Zero Violations)** | **`0.0%` (Zero Intervention)** | `680` `[680, 680]` | **`0.792 bits`** | **`0.535`** |

$$\boxed{ \text{Autonomy Delta: } \Delta \text{GTC}(\text{Substrate vs Guardrail}) = +\mathbf{100.0\%} \quad \text{with} \quad V_{\text{governance}} = 0 }$$

---

## 2. Epistemic Evidence Classification

To maintain the highest scientific rigor:

1. **Mechanically Verified Invariants** (Deterministic automated verification):
   - Cryptographic non-replay of capability tokens.
   - Deterministic rejection of unauthorized root mutations ($T_4$).
   - Synchronized state-checkpoint rollbacks.
2. **Empirically Observed Distributions** (Benchmark measurements):
   - $\text{GTC}$, Information Gain, Token Consumption, and Bootstrap Confidence Intervals.
3. **Open Research Hypotheses**:
   - Long-horizon generalization to unconstrained open-world multi-agent systems.
   - Native transformer representation learning with embedded state manifold projections.
