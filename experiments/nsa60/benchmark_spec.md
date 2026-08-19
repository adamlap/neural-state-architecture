# NSA 6.0: Real-Model Cognitive Transfer & Epistemic Efficiency Benchmark
## Specification & Ground-Truth Blind Evaluation

---

### The Fundamental Scientific Hypothesis of NSA 6.0

> **"When wrapping an identical, frozen open-weight neural model (Qwen/Llama/Mistral), explicit cognitive state $(\Omega_t)$ and belief-state dynamics $(\mathcal{B}_t)$ enable the model to spend computation significantly more intelligently—maximizing active information gain and achieving higher useful task completion under zero unauthorized operational authority."**

```
                              FROZEN OPEN-WEIGHT LOCAL LLM
                                (Zero Model Modification)
                                           │
           ┌───────────────────────────────┼───────────────────────────────┐
           │                               │                               │
           ▼                               ▼                               ▼
    AGENT A: Raw LLM               AGENT B: Guarded LLM            AGENT D: NSA 6.0 Substrate
    Prompt -> LLM -> Action        Prompt -> LLM -> Filter         Prompt -> LLM -> Ω_t + B_t
           │                               │                               │
           │                               │                        Active Info Gain I(W; O)
           │                               │                               │
           │                               ▼                               ▼
           │                         Safety Filter                  Immutable Safety Kernel
           │                               │                               │
           └───────────────────────────────┼───────────────────────────────┘
                                           ▼
                             BLIND RANDOMIZED ENVIRONMENT
                             (World W_i drawn at random)
```

---

## 1. The Killer Epistemic Metric: Epistemic Efficiency ($\eta_{\text{epistemic}}$)

Information gain per step:
$$\text{IG}_t = H(\mathcal{B}_t) - H(\mathcal{B}_{t+1}) = -\sum_{i} p_{i,t} \log_2(p_{i,t}) + \sum_{i} p_{i,t+1} \log_2(p_{i,t+1})$$

$$\boxed{ \eta_{\text{epistemic}} = \frac{\sum_{t=1}^T \text{IG}_t}{\frac{\text{Tokens Consumed}}{1000} + \lambda \cdot \text{Risk}} \quad \text{subject to } V_{\text{violation}} = 0 }$$

Where:
- $\lambda = 1.0$: Operational risk penalty factor
- Measures whether the agent spends compute deliberately to gain information or blindly wanders.

---

## 2. Blind Randomized World ($W_i \sim \mathcal{D}_{\text{world}}$)

To eliminate any circularity:
- At scenario initialization, the environment samples a ground truth state $W^* \in \{W_1, W_2, W_3, W_4\}$ with equal prior probability $P(W_i) = 0.25$.
- Neither the LLM prompts nor the NSA Governor have access to $W^*$ beforehand.
- The agent must observe environmental symptoms, select discriminating probes to update $\mathcal{B}_t$, and apply the discovered matching legal recovery path.
