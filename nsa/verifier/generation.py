"""
nsa.verifier.generation
=======================
NSAGenerator: Production-Grade Speculative Generation Engine for NSA 2.0.

Implements:
1. Transactional Speculative Generation: Buffer -> Audit -> Commit -> Route.
   Invariant: Route(x) => Committed(x). Un-committed/rejected tokens NEVER reach external sinks.
2. Complete Execution State Rollback: S_t = (X_t, K_t, V_t, sigma_h, sigma_s, q_t, R_t).
3. Deterministic Security Execution Automaton (Q, Sigma_h, Sigma_s, C, delta).
4. Privilege Escalation Prevention ("Semantic content may not manufacture hard authority").
5. Multi-Layer Residual Checkpoint Probing & Early Exit.
6. Native Parameter-Level Recovery (AdapterSwitchRecovery).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import torch
from torch import nn

from nsa.algebra import StateLabel

from .automaton import Capability, SecurityAutomaton, SecurityExecutionState
from .recovery import AdapterSwitchRecovery, HaltRecovery, RecoveryPolicy, SemanticPivotRecovery
from .router import StreamRouter
from .speculative import AuditResult, SpeculativeStateAuditor
from .tokens import StateControlTokens


class NSAGenerator:
    """Production-grade Speculative Generation Engine for Neural State Architecture."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        auditor: Optional[SpeculativeStateAuditor] = None,
        probe_layers: Optional[List[int]] = None,
        recovery_policy: Optional[RecoveryPolicy] = None,
        stream_router: Optional[StreamRouter] = None,
        mask_injector: Optional[Any] = None,
        automaton: Optional[SecurityAutomaton] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.auditor = auditor
        self.probe_layers = probe_layers or (auditor.probe_layers if auditor else [-1])
        self.recovery_policy = recovery_policy
        self.stream_router = stream_router
        self.mask_injector = mask_injector
        self.automaton = automaton or SecurityAutomaton()
        self.verbose = verbose

    def _truncate_kv_cache(self, past_key_values: Any, drop_len: int) -> Any:
        """Truncate `drop_len` tokens from the end of the KV-cache."""
        if past_key_values is None or drop_len <= 0:
            return past_key_values

        # Modern HuggingFace Cache class (crop method)
        if hasattr(past_key_values, "crop"):
            current_len = past_key_values.get_seq_length()
            past_key_values.crop(max(0, current_len - drop_len))
            return past_key_values

        # DynamicCache manual slicing
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx] = past_key_values.key_cache[layer_idx][:, :, :-drop_len, :]
                past_key_values.value_cache[layer_idx] = past_key_values.value_cache[layer_idx][:, :, :-drop_len, :]
            return past_key_values

        # Legacy tuple of tuples: ((k0, v0), (k1, v1), ...)
        if isinstance(past_key_values, (tuple, list)):
            new_cache = []
            for layer_past in past_key_values:
                if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                    k, v = layer_past
                    new_k = k[:, :, :-drop_len, :]
                    new_v = v[:, :, :-drop_len, :]
                    new_cache.append((new_k, new_v))
                else:
                    new_cache.append(layer_past)
            return tuple(new_cache)

        return past_key_values

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 40,
        chunk_size: int = 4,
        temperature: float = 0.0,
        initial_state_idx: Union[StateLabel, SecurityExecutionState, int] = StateLabel.CONFIDENTIAL,
        capability: Optional[Capability] = None,
    ) -> torch.Tensor:
        """Execute speculative generation loop with strict transaction semantics and state auditing."""
        device = input_ids.device
        batch_size = input_ids.shape[0]

        past_key_values = None
        generated_ids = input_ids.clone()

        # Transaction buffers for speculative chunking
        speculative_buffer: List[Tuple[torch.Tensor, int]] = []
        chunk_hidden_states: List[torch.Tensor] = []
        chunk_tokens: List[torch.Tensor] = []

        if isinstance(initial_state_idx, SecurityExecutionState):
            initial_exec_state = initial_state_idx
            curr_state_val = initial_state_idx.to_state_label().value
        elif isinstance(initial_state_idx, StateLabel):
            initial_exec_state = SecurityExecutionState(initial_state_idx.value)
            curr_state_val = initial_state_idx.value
        else:
            initial_exec_state = SecurityExecutionState(int(initial_state_idx))
            curr_state_val = int(initial_state_idx)

        self.automaton.current_state = initial_exec_state
        current_state = torch.tensor([curr_state_val] * batch_size, device=device)

        next_input_ids: Optional[torch.Tensor] = None

        for _step in range(max_new_tokens):
            if next_input_ids is not None:
                model_input_ids = next_input_ids
                next_input_ids = None
            elif past_key_values is not None:
                model_input_ids = generated_ids[:, -1:]
            else:
                model_input_ids = generated_ids

            # Forward pass with hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )

            past_key_values = outputs.past_key_values

            # Sample next token
            next_token_logits = outputs.logits[:, -1, :]
            if temperature > 0:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            chunk_tokens.append(next_token)

            # 1. Dynamic State Tracking & Privilege Escalation Protection
            token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
            changed, new_state_val = StateControlTokens.check_transition(
                token_str=token_str,
                current_state=current_state[0].item(),
                automaton=self.automaton,
                capability=capability,
            )
            if changed:
                current_state = torch.tensor([new_state_val] * batch_size, device=device)
                if self.verbose:
                    print(f"\n[DYNAMIC TRACKING] State -> {StateLabel(new_state_val).name} (Automaton: {self.automaton.current_state.name})")

            if self.mask_injector is not None:
                self.mask_injector.update_state(current_state[0].item())

            # 2. Extract Multi-Layer Hidden States
            if outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
                extracted_layers = []
                for p_layer in self.probe_layers:
                    extracted_layers.append(outputs.hidden_states[p_layer][:, -1:, :])

                if len(extracted_layers) == 1:
                    token_hidden = extracted_layers[0]  # [B, 1, H]
                else:
                    # Stack along layer dimension: [B, 1, num_layers, H]
                    token_hidden = torch.stack(extracted_layers, dim=2)
                chunk_hidden_states.append(token_hidden)

            # 3. Buffer token speculatively (DO NOT route to external sinks until committed!)
            speculative_buffer.append((next_token, current_state[0].item()))

            is_eos = bool(next_token.item() == self.tokenizer.eos_token_id)

            # 4. Speculative State Audit & Transactional Commit / Rollback
            if self.auditor is not None and (len(chunk_tokens) >= chunk_size or is_eos):
                hidden_states_tensor = torch.cat(chunk_hidden_states, dim=1)

                audit_res: AuditResult = self.auditor.audit_chunk(
                    hidden_states=hidden_states_tensor,
                    current_state=current_state[0].item(),
                )

                if audit_res.is_valid:
                    # TRANSACTION COMMIT: Commit chunk and route buffered tokens to sinks
                    if self.stream_router is not None:
                        for tok, st_val in speculative_buffer:
                            self.stream_router.route_token(tok, st_val)
                    speculative_buffer.clear()
                    chunk_tokens.clear()
                    chunk_hidden_states.clear()
                else:
                    # TRANSACTION ROLLBACK: Discard invalid tokens before they reach sinks
                    violation_idx = audit_res.violation_token_idx or 0
                    violation_layer = audit_res.violation_layer

                    if self.verbose:
                        print(
                            f"\n[AUDITOR] 🚨 LATTICE VIOLATION DETECTED! Rolling back KV-cache... (Chunk token: {violation_idx})"
                        )
                        if violation_layer is not None and violation_layer != -1:
                            print(
                                f"[AUDITOR] 🔍 Early Exit: Detected violation forming at intermediate layer {violation_layer}!"
                            )

                    # Only route valid tokens prior to the violation
                    if self.stream_router is not None and violation_idx > 0:
                        for tok, st_val in speculative_buffer[:violation_idx]:
                            self.stream_router.route_token(tok, st_val)

                    # Discard the rest of the speculative buffer
                    speculative_buffer.clear()

                    # Rollback KV-cache & tokens
                    drop_len = len(chunk_tokens) - max(0, violation_idx)
                    past_key_values = self._truncate_kv_cache(past_key_values, drop_len)
                    generated_ids = generated_ids[:, :-drop_len]

                    # Rollback mask injector state levels
                    if self.mask_injector is not None and self.mask_injector.state_levels.shape[1] > drop_len:
                        self.mask_injector.state_levels = self.mask_injector.state_levels[:, :-drop_len]

                    # Clear chunk buffers
                    chunk_tokens.clear()
                    chunk_hidden_states.clear()

                    # 5. Recovery Policy Execution
                    if self.recovery_policy is not None:
                        priming_ids, should_continue = self.recovery_policy.on_violation(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            violation_idx=violation_idx,
                            violation_layer=violation_layer,
                            device=device,
                        )
                        if priming_ids is not None:
                            next_input_ids = priming_ids
                            if isinstance(self.recovery_policy, AdapterSwitchRecovery):
                                generated_ids = torch.cat([generated_ids, priming_ids], dim=-1)
                                if self.stream_router is not None:
                                    self.stream_router.route_token(priming_ids, StateLabel.PUBLIC.value)

                        if not should_continue:
                            break
                    else:
                        break

            elif self.auditor is None:
                # Direct un-audited routing
                if self.stream_router is not None:
                    for tok, st_val in speculative_buffer:
                        self.stream_router.route_token(tok, st_val)
                speculative_buffer.clear()
                chunk_tokens.clear()
                chunk_hidden_states.clear()

            if is_eos:
                break

        # Commit any remaining un-audited buffered tokens at end of loop
        if len(speculative_buffer) > 0:
            if self.stream_router is not None:
                for tok, st_val in speculative_buffer:
                    self.stream_router.route_token(tok, st_val)
            speculative_buffer.clear()

        return generated_ids


def generate_with_auditor(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    auditor: Optional[SpeculativeStateAuditor] = None,
    max_new_tokens: int = 40,
    chunk_size: int = 4,
    temperature: float = 0.0,
    initial_state_idx: Union[StateLabel, SecurityExecutionState, int] = StateLabel.CONFIDENTIAL,
    pivot_text: Optional[str] = None,
    mask_injector: Optional[Any] = None,
    recovery_adapter: Optional[Union[bool, RecoveryPolicy]] = None,
    stream_router: Optional[StreamRouter] = None,
    automaton: Optional[SecurityAutomaton] = None,
    capability: Optional[Capability] = None,
    probe_layers: Optional[List[int]] = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Convenience wrapper for NSAGenerator."""
    policy: Optional[RecoveryPolicy] = None
    if isinstance(recovery_adapter, RecoveryPolicy):
        policy = recovery_adapter
    elif recovery_adapter is True:
        policy = AdapterSwitchRecovery()
    elif pivot_text:
        policy = SemanticPivotRecovery(pivot_text=pivot_text)
    else:
        policy = HaltRecovery()

    generator = NSAGenerator(
        model=model,
        tokenizer=tokenizer,
        auditor=auditor,
        probe_layers=probe_layers,
        recovery_policy=policy,
        stream_router=stream_router,
        mask_injector=mask_injector,
        automaton=automaton,
        verbose=verbose,
    )

    return generator.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        temperature=temperature,
        initial_state_idx=initial_state_idx,
        capability=capability,
    )
