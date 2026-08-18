# NSA 2.0: Speculative State Auditing & Dynamic Alignment Engine

## Executive Summary

Neural State Architecture 1.0 established the mathematical foundation of **Typed Neural Computation (TNC)** and **Static Attention Mask Injection (NSA-LoRA)**.

**NSA 2.0** elevates this architecture into an active, self-governing runtime execution environment that bridges structural attention-layer guarantees with dynamic token-level behavior:

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

---

## Core Pillars of NSA 2.0

### 1. Dynamic State Tracking (The "Moving Mask")
* **The Problem**: In static NSA, prompt tokens were statically labeled ($SYSTEM$ prompt = Level 5, $PUBLIC$ user query = Level 1). However, as models generate text, their internal reasoning state shifts.
* **The Solution**: 
  - Tokens like `<|start_system_thought|>` and `<|end_system_thought|>` allow models to autonomously enter and exit high-clearance scratchpad states.
  - `NSAMaskInjector.update_state(new_level)` dynamically appends newly generated token security levels to the state tensor $\sigma$ and recomputes the additive attention mask on-the-fly.
  - Subsequent $PUBLIC$ chat output tokens are **mathematically barred** at the attention layer from attending to those sensitive reasoning tokens.

### 2. Multi-Layer Auditing & Deep Probing
* **The Problem**: Waiting until the final layer ($L_{final}$) to detect policy violations means the model has already committed to leaking sensitive data.
* **The Solution**:
  - `MultiLayerStateAuditor` probes intermediate residual streams (e.g., Layer 12, Layer 18, Layer 24).
  - If an intermediate probe detects an unsafe state trajectory forming deep inside the model, it triggers an **Early Exit**, rolling back the KV-cache before the violation reaches the output logits, saving substantial compute.

### 3. Native Recovery Policies (Weight-Level Adapters)
* **The Problem**: Relying on prompt injection (e.g. appending `<|im_start|>system...`) for steering is brittle and pollutes the context window.
* **The Solution**:
  - `RecoveryPolicy` provides formal recovery strategies:
    - `AdapterSwitchRecovery`: Instantly hot-swaps to a specialized refusal LoRA adapter to emit a safe refusal at the parameter level.
    - `SemanticPivotRecovery`: Structured steering fallback with bounded retry counts.
    - `HaltRecovery`: Strict, immediate termination.

### 4. Compartmented Execution (Clearance-Aware Stream Routing)
* **The Problem**: Enterprises need models to reason about secrets (e.g., using an internal API key to query a database) without revealing those secrets to the user.
* **The Solution**:
  - `StreamRouter` intercepts generated tokens at each step.
  - Tokens generated in the $SYSTEM$ state are routed to `SYSTEM STDOUT` (e.g., database query API).
  - Tokens generated in the $PUBLIC$ state are routed to `PUBLIC STDOUT` (e.g., user chat window).
  - Because Phase 1 guarantees $PUBLIC$ tokens cannot attend to $SYSTEM$ tokens, the model cannot accidentally summarize or leak the database payload back to the user.

---

## Architecture Flow Diagram

```
                              User Prompt + RAG Document
                                         │
                                         ▼
                            ┌────────────────────────┐
                            │    NSAMaskInjector     │
                            │   (Attention Hooks)    │
                            └────────────┬───────────┘
                                         │
                        Autoregressive Forward Step
                                         │
       ┌─────────────────────────────────┴─────────────────────────────────┐
       │                                                                   │
       ▼                                                                   ▼
┌──────────────┐                                            ┌───────────────────────────────┐
│ Control Tag  │                                            │     Multi-Layer Probing       │
│  Detection   │                                            │ (Layers 12, 18, 24 Residuals) │
└──────┬───────┘                                            └──────────────┬────────────────┘
       │ <|start_system_thought|>                                          │
       ▼                                                                   ▼
┌──────────────┐                                            ┌───────────────────────────────┐
│ Dynamic Mask │                                            │   SpeculativeStateAuditor     │
│  Expansion   │                                            │  (Lattice Violation Check)    │
└──────────────┘                                            └──────────────┬────────────────┘
                                                                           │
                                                    ┌──────────────────────┴──────────────────────┐
                                                    ▼                                             ▼
                                              [Valid Chunk]                               [Lattice Violation]
                                                    │                                             │
                                                    ▼                                             ▼
                                     ┌─────────────────────────────┐               ┌─────────────────────────────┐
                                     │        StreamRouter         │               │     KV-Cache Rollback       │
                                     │  ┌───────────────────────┐  │               │   (Drop Violating Tokens)   │
                                     │  │ SYSTEM -> Tool API    │  │               └──────────────┬──────────────┘
                                     │  │ PUBLIC -> User Chat   │  │                              │
                                     │  └───────────────────────┘  │                              ▼
                                     └─────────────────────────────┘               ┌─────────────────────────────┐
                                                                                   │       RecoveryPolicy        │
                                                                                   │ (Hot-Swap Refusal Adapter)  │
                                                                                   └─────────────────────────────┘
```

---

## Code Example

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

# 1. Load Model and Tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# 2. Configure Stream Router for Compartmented Execution
router = StreamRouter(tokenizer=tokenizer)
router.register_sink(StateLabel.PUBLIC, lambda text, tid: print(f"[USER CHAT]: {text}", end=""))
router.register_sink(StateLabel.SYSTEM, lambda text, tid: print(f"[BACKEND TOOL]: {text}", end=""))

# 3. Setup Multi-Layer Auditor & Recovery Policy
head = StateEncoderHead(hidden_size=model.config.hidden_size, num_states=len(StateLabel))
head.load_state_dict(torch.load("trained_auditor_weights.pt", weights_only=True))

auditor = MultiLayerStateAuditor(
    encoder_head=head,
    lattice_validator=lambda pred: pred != StateLabel.SYSTEM.value,
    chunk_size=4,
    probe_layers=[-1, 12], # Check intermediate layer 12 and final layer
)

# 4. Generate with Dynamic Mask Injection
input_ids = tokenizer.encode("Explain the database schema.", return_tensors="pt")
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
        max_new_tokens=60,
    )
```
