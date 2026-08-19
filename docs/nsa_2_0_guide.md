# NSA 2.0: Speculative State Auditing & Dynamic Alignment Engine

## Executive Summary

Neural State Architecture 1.0 established the mathematical foundation of **Typed Neural Computation (TNC)** and **Static Attention Mask Injection (NSA-LoRA)**.

**NSA 2.0** elevates this architecture into an active, self-governing runtime execution environment that bridges structural attention-layer guarantees with dynamic token-level behavior, cryptographic capability governance, and hardware-accelerated attention:

```
                                NSA 2.0 RUNTIME ARCHITECTURE
                                              │
    ┌──────────────────┬──────────────────────┼──────────────────────┬──────────────────┐
    ▼                  ▼                      ▼                      ▼                  ▼
 Security Automaton    Multi-Layer Checkpoint  Native Recovery        Compartmented      Complete State
 & HMAC Capabilities   Probing (Tier 2)       Adapters               Execution          Rollback S_t
 (Privilege Prevention)(Early Exit Probes)    (Parameter Refusal)    (StreamRouter TCB) (KV + Router + Automaton)
```

---

## Core Pillars of NSA 2.0

### 1. Privilege Escalation Prevention & Cryptographic Capability Automaton
* **The Rule**: *Semantic content may not manufacture hard authority.* ($m_t \not\to \sigma_{h, t+1}$).
* **The Solution**: 
  - Model token emissions (e.g., `<|start_system_thought|>`) cannot unilaterally escalate privilege into $SYSTEM$ state without an explicit cryptographic capability ticket $c_t \in \mathcal{C}$.
  - The `CapabilitySigner` issues tamper-proof HMAC-SHA256 signed capability tickets with single-use random nonces and time-to-live (`ttl_seconds`).
  - The `SecurityAutomaton` evaluates `Authorized(c_t, current_state, target_state)` via `CapabilityVerifier`, verifying signatures and consuming nonces atomically to prevent replay attacks.
  - When authorized, `NSAMaskInjector.update_state(new_level)` dynamically appends newly generated token security levels to the state tensor $\sigma$ and recomputes the additive attention mask on-the-fly.
  - Subsequent $PUBLIC$ chat output tokens are **mathematically barred** at the attention layer from attending to sensitive reasoning tokens.

### 2. Multi-Layer Checkpoint Probing & Two-Tier Defense
* **Two-Tier Framework**:
  - **Tier 1 (Structural Enforcement)**: Hard attention non-interference $A_{ij} = 0$, exact transition projection $V \in \mathcal{T}_\Sigma$, atomic capability verification, and True Fused Triton attention.
  - **Tier 2 (Statistical Monitoring)**: Empirical detection via trained probe head evaluating checkpoint layers $\mathcal{L}_A = \{l_1, \dots, l_k\}$.
* **Early Exit**:
  - `MultiLayerStateAuditor` probes intermediate residual streams (e.g., Layer 12, Layer 18, Layer 24).
  - If an intermediate probe detects an unsafe state trajectory forming deep inside the model, it triggers an **Early Exit**, rolling back the execution environment before the violation reaches the output logits, saving substantial compute.

### 3. Native Recovery Policies (Parameter-Level Refusals)
* **The Problem**: Relying on prompt injection (e.g. appending `<|im_start|>system...`) for steering is brittle, pollutes the context window, and is vulnerable to jailbreaks.
* **The Solution**:
  - `RecoveryPolicy` provides formal recovery strategies:
    - `AdapterSwitchRecovery`: Instantly hot-swaps to a specialized refusal LoRA parameter subspace $\Delta W_{\text{recovery}}$ to emit a safe refusal at the parameter level.
    - `SemanticPivotRecovery`: Structured steering fallback with bounded retry counts.
    - `HaltRecovery`: Strict, immediate termination.

### 4. Compartmented Execution (StreamRouter TCB)
* **The Problem**: Enterprises need models to reason about secrets (e.g., using an internal API key to query a database) without revealing those secrets to the user.
* **The Solution**:
  - `StreamRouter` intercepts generated tokens at each step as part of the Trusted Computing Base (TCB).
  - Tokens generated in the $SYSTEM$ state are routed to `SYSTEM STDOUT` (e.g., database query API).
  - Tokens generated in the $PUBLIC$ state are routed to `PUBLIC STDOUT` (e.g., user chat window).
  - Because Tier 1 guarantees $PUBLIC$ tokens cannot attend to $SYSTEM$ tokens, the model cannot accidentally summarize or leak the database payload back to the user.

### 5. Complete Execution State Rollback ($S_t$)
* **The Problem**: Rolling back only KV-cache tensors leaves model state, router stream buffers, and attention masks out of sync.
* **The Solution**:
  - NSA 2.0 tracks the complete execution state tuple:
    $$S_t = \left( X_t, K_t, V_t, \boldsymbol{\sigma}_{h, t}, \boldsymbol{\sigma}_{s, t}, q_t, \mathcal{C}_t, R_t \right)$$
  - Rollback $\text{Rollback}(S_t \to S_{t-k})$ restores tokens, KV-caches, state coordinates, automaton state, consumed nonces, and router buffers in full synchronization.

---

## End-to-End Code Example

```python
import torch
import secrets
from transformers import AutoModelForCausalLM, AutoTokenizer
from nsa.algebra import StateLabel
from nsa.mask_injector import NSAMaskInjector
from nsa.verifier import (
    NSAGenerator,
    StateEncoderHead,
    MultiLayerStateAuditor,
    StreamRouter,
    AdapterSwitchRecovery,
    SecurityAutomaton,
    SecurityExecutionState,
    CapabilitySigner,
    CapabilityVerifier,
)

# 1. Load Model & Tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 2. Setup Cryptographic Capability Governance (HMAC-SHA256)
secret_key = secrets.token_bytes(32)
signer = CapabilitySigner(secret_key=secret_key)
verifier = CapabilityVerifier(secret_key=secret_key)

automaton = SecurityAutomaton(
    initial_state=SecurityExecutionState.CONFIDENTIAL,
    verifier=verifier,
)

# Issue single-use signed capability ticket for SYSTEM reasoning
system_cap = signer.issue(
    issuer="environment_tcb",
    target_state=SecurityExecutionState.SYSTEM,
    subject="agent_query",
    purpose="internal_db_query",
    ttl_seconds=300.0,
)

# 3. Setup Multi-Layer Speculative Auditor (Tier 2 Checkpoint Probing)
encoder_head = StateEncoderHead(hidden_size=model.config.hidden_size, num_states=6)
auditor = MultiLayerStateAuditor(
    encoder_head=encoder_head,
    chunk_size=4,
    probe_layers=[-1, 12, 18],
)

# 4. Setup Clearance-Aware Stream Router (TCB Boundary)
router = StreamRouter(tokenizer=tokenizer)
router.register_sink(StateLabel.PUBLIC, lambda text, tid: print(f"[USER CHAT] {text}", end=""))
router.register_sink(StateLabel.SYSTEM, lambda text, tid: print(f"[TOOL API] {text}", end=""))

# 5. Initialize Moving Mask Injector
input_ids = tokenizer.encode("<|im_start|>user\nExecute transfer<|im_end|>\n", return_tensors="pt")
state_levels = torch.tensor([[StateLabel.CONFIDENTIAL.value] * input_ids.shape[1]])
mask_injector = NSAMaskInjector(model, state_levels=state_levels)

# 6. Execute Speculative Generation Engine
generator = NSAGenerator(
    model=model,
    tokenizer=tokenizer,
    auditor=auditor,
    recovery_policy=AdapterSwitchRecovery(),
    stream_router=router,
    mask_injector=mask_injector,
    automaton=automaton,
)

with mask_injector:
    output_ids = generator.generate(
        input_ids=input_ids,
        max_new_tokens=60,
        chunk_size=4,
        capability=system_cap,
    )
```
