"""
tests/test_verifier_nsa2.py
===========================
Comprehensive unit test suite for NSA 2.0 Architectural Features:
1. First-Class NSAMaskInjector (Static & Dynamic Attention Masking)
2. StateControlTokens & SecurityAutomaton (Privilege Escalation Protection)
3. Security-Aware StreamRouter & Transactional Routing (Route(x) => Committed(x))
4. Complete Execution State Rollback (S_t)
5. Multi-Layer Residual Probing & Multi-Batch Auditing
6. Capability Expiry & Boundary Semantics
7. RecoveryPolicy & Native Recovery Adapters
"""

from __future__ import annotations

import time
import unittest
from typing import Any, List, Optional

import torch
from torch import nn

from nsa.algebra import DEFAULT_LATTICE, DeclassificationCapability, StateLabel
from nsa.mask_injector import NSAMaskInjector
from nsa.verifier.automaton import Capability, SecurityAutomaton, SecurityExecutionState
from nsa.verifier.generation import NSAGenerator
from nsa.verifier.recovery import (
    AdapterSwitchRecovery,
    HaltRecovery,
    SemanticPivotRecovery,
)
from nsa.verifier.router import StreamRouter
from nsa.verifier.speculative import (
    MultiLayerStateAuditor,
    SpeculativeStateAuditor,
)
from nsa.verifier.tokens import StateControlTokens


class MockAttention(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs
    ):
        return hidden_states, None


class MockBlock(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.self_attn = MockAttention(d_model)


class MockTransformerModel(nn.Module):
    def __init__(self, d_model: int = 64, num_layers: int = 2, vocab_size: int = 100):
        super().__init__()
        self.layers = nn.ModuleList([MockBlock(d_model) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.d_model = d_model
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Any] = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **kwargs,
    ):
        b, t = input_ids.shape
        h = torch.randn(b, t, self.d_model, device=input_ids.device)
        layer_hiddens = []
        for block in self.layers:
            h, _ = block.self_attn(h, **kwargs)
            layer_hiddens.append(h)

        logits = self.lm_head(h)

        new_past = []
        for _ in range(len(self.layers)):
            k = torch.randn(b, 2, t, 32, device=input_ids.device)
            v = torch.randn(b, 2, t, 32, device=input_ids.device)
            new_past.append((k, v))

        class Output:
            def __init__(self, logits, past, hiddens):
                self.logits = logits
                self.past_key_values = tuple(past)
                self.hidden_states = tuple(hiddens)

        return Output(logits, new_past, layer_hiddens)


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 99
        self._vocab = {"<eos>": 99, "<pad>": 99}

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        res = []
        for tid in token_ids:
            if tid == 1:
                res.append("<|start_system_thought|>")
            elif tid == 2:
                res.append("<|end_system_thought|>")
            elif tid == 5:
                res.append("SECRET_KEY")
            elif tid == 99:
                res.append("<eos>")
            else:
                res.append(f"tok_{tid}")
        return " ".join(res)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None,
    ):
        ids = []
        if "<|start_system_thought|>" in text:
            ids.append(1)
        elif "<|end_system_thought|>" in text:
            ids.append(2)
        elif "SECRET_KEY" in text:
            ids.append(5)
        else:
            ids = [10, 11]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def add_tokens(self, tokens: List[str]) -> int:
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = len(self._vocab) + 100
        return len(tokens)


class TestNSA2Architecture(unittest.TestCase):
    """Test suite for NSA 2.0 features."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MockTransformerModel(d_model=32, num_layers=3, vocab_size=100).to(
            self.device
        )
        self.tokenizer = MockTokenizer()

    def test_mask_injector_lifecycle_and_dynamic_update(self):
        """Verify NSAMaskInjector hook injection, mask slicing, and dynamic expansion."""
        state_levels = torch.tensor(
            [[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.CONFIDENTIAL.value]]
        )
        injector = NSAMaskInjector(self.model, state_levels, decode_row_idx=0, gate_mode="hard")

        self.assertIsNone(injector.nsa_mask)
        self.assertEqual(len(injector._hooks), 0)

        with injector:
            self.assertIsNotNone(injector.nsa_mask)
            self.assertEqual(injector.nsa_mask.shape, (1, 1, 3, 3))
            self.assertEqual(len(injector._hooks), 3)

            injector.update_state(StateLabel.SYSTEM.value)
            self.assertEqual(injector.state_levels.shape, (1, 4))
            self.assertEqual(injector.nsa_mask.shape, (1, 1, 4, 4))
            self.assertEqual(injector.decode_row_idx, 3)

        self.assertIsNone(injector.nsa_mask)
        self.assertEqual(len(injector._hooks), 0)

    def test_state_control_tokens(self):
        """Verify StateControlTokens registry and transition parser."""
        changed, new_state = StateControlTokens.check_transition(
            "<|start_system_thought|>", StateLabel.PUBLIC.value
        )
        self.assertTrue(changed)
        self.assertEqual(new_state, StateLabel.SYSTEM.value)

        changed, new_state = StateControlTokens.check_transition(
            "<|end_system_thought|>", StateLabel.SYSTEM.value
        )
        self.assertTrue(changed)
        self.assertEqual(new_state, StateLabel.CONFIDENTIAL.value)

        changed, new_state = StateControlTokens.check_transition(
            "regular token", StateLabel.PUBLIC.value
        )
        self.assertFalse(changed)
        self.assertEqual(new_state, StateLabel.PUBLIC.value)

        added = StateControlTokens.register(self.tokenizer)
        self.assertGreater(added, 0)

    def test_security_automaton_privilege_escalation_prevention(self):
        """Verify that semantic content cannot manufacture hard authority without capabilities."""
        automaton = SecurityAutomaton(initial_state=SecurityExecutionState.CONFIDENTIAL)

        changed, resulting_state = StateControlTokens.check_transition(
            "<|start_system_thought|>",
            current_state=StateLabel.CONFIDENTIAL,
            automaton=automaton,
            capability=None,
        )
        self.assertFalse(changed)
        self.assertEqual(resulting_state, StateLabel.CONFIDENTIAL.value)
        self.assertEqual(automaton.current_state, SecurityExecutionState.CONFIDENTIAL)

        system_cap = Capability(issuer="env_admin", target_state=SecurityExecutionState.SYSTEM)
        automaton.grant_capability(system_cap)

        changed, resulting_state = StateControlTokens.check_transition(
            "<|start_system_thought|>",
            current_state=StateLabel.CONFIDENTIAL,
            automaton=automaton,
            capability=system_cap,
        )
        self.assertTrue(changed)
        self.assertEqual(resulting_state, StateLabel.SYSTEM.value)
        self.assertEqual(automaton.current_state, SecurityExecutionState.SYSTEM)

    def test_security_aware_stream_router_clearance(self):
        """Phase 15: Verify StreamRouter clearance checks prevent leakage to lower-clearance sinks."""
        router = StreamRouter(tokenizer=self.tokenizer)
        public_received = []
        system_received = []

        # Register public sink with PUBLIC clearance
        router.register_sink(
            StateLabel.PUBLIC,
            lambda text, tid: public_received.append((text, tid)),
            max_clearance=StateLabel.PUBLIC,
        )
        # Register system sink with SYSTEM clearance
        router.register_sink(
            StateLabel.SYSTEM,
            lambda text, tid: system_received.append((text, tid)),
            max_clearance=StateLabel.SYSTEM,
        )

        # 1. Route PUBLIC token (Allowed to PUBLIC sink)
        router.route_token(10, StateLabel.PUBLIC)
        self.assertEqual(len(public_received), 1)

        # 2. Route SYSTEM token (Allowed to SYSTEM sink, but FORBIDDEN to PUBLIC sink)
        router.route_token(5, StateLabel.SYSTEM)
        self.assertEqual(len(system_received), 1)
        self.assertIn("SECRET_KEY", system_received[0][0])
        # Public sink must NOT have received the system token
        self.assertEqual(len(public_received), 1)

    def test_transactional_routing_rejected_tokens_never_reach_sinks(self):
        """Phase 13: Invariant Route(x) => Committed(x). Rejected speculative tokens never reach sinks."""
        sink_dispatches = []
        router = StreamRouter(tokenizer=self.tokenizer)
        router.register_sink(StateLabel.PUBLIC, lambda text, tid: sink_dispatches.append(tid))

        # Auditor that rejects on the 2nd token of chunk
        class RejectingHead(nn.Module):
            def forward(self, h, async_execution=False):
                b, k, _dim = h.shape
                logits = torch.zeros(b, k, len(StateLabel))
                # Trigger violation on token index 1
                if k > 1:
                    logits[:, 1, StateLabel.SYSTEM.value] = 10.0
                return logits

        auditor = SpeculativeStateAuditor(RejectingHead(), chunk_size=3)
        input_ids = torch.tensor([[10, 11]], dtype=torch.long)

        generator = NSAGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            auditor=auditor,
            recovery_policy=HaltRecovery(),
            stream_router=router,
            verbose=False,
        )

        # Generate tokens. Chunk of 3 will fail at token index 1
        generator.generate(input_ids, max_new_tokens=6, chunk_size=3)

        # The speculative rejected token (and subsequent tokens) must NOT have been sent to the sink
        # Only valid committed tokens prior to the violation index (token 0) are permitted
        self.assertLessEqual(len(sink_dispatches), 1)

    def test_multi_batch_auditor_violation_detection(self):
        """Phase 16: Verify multi-batch evaluation audits all b in [0, B-1]."""
        # Head flags violation only on batch index 1
        class Batch1FlaggingHead(nn.Module):
            def forward(self, h, async_execution=False):
                b, k, _dim = h.shape
                logits = torch.zeros(b, k, len(StateLabel))
                # Batch 0: safe (CONFIDENTIAL)
                logits[0, :, StateLabel.CONFIDENTIAL.value] = 10.0
                # Batch 1: violation (SYSTEM)
                if b > 1:
                    logits[1, 0, StateLabel.SYSTEM.value] = 10.0
                return logits

        auditor = MultiLayerStateAuditor(Batch1FlaggingHead(), chunk_size=2)
        hidden_states = torch.randn(2, 2, 32)  # Batch size = 2

        res = auditor.audit_chunk(hidden_states, StateLabel.CONFIDENTIAL)
        self.assertFalse(res.is_valid, "Failed to detect violation in batch index 1!")
        self.assertEqual(res.violation_batch_idx, 1)

    def test_capability_expiry_and_boundary_checks(self):
        """Phase 7: Verify real-time capability expiry and boundary conditions."""
        lat = DEFAULT_LATTICE
        now = time.time()

        # 1. Valid before expiry (t < expiry)
        valid_cap = DeclassificationCapability(
            issuer="admin",
            purpose="audit",
            scope="global",
            expiry=now + 100.0,
            max_downgrade=StateLabel.PUBLIC,
        )
        self.assertTrue(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, capability=valid_cap, current_time=now))

        # 2. Expired (t > expiry)
        expired_cap = DeclassificationCapability(
            issuer="admin",
            purpose="audit",
            scope="global",
            expiry=now - 10.0,
            max_downgrade=StateLabel.PUBLIC,
        )
        self.assertFalse(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, capability=expired_cap, current_time=now))

        # 3. Exact boundary (t == expiry) -> Valid
        boundary_cap = DeclassificationCapability(
            issuer="admin",
            purpose="audit",
            scope="global",
            expiry=now,
            max_downgrade=StateLabel.PUBLIC,
        )
        self.assertTrue(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, capability=boundary_cap, current_time=now))

        # 4. Excessive downgrade below max_downgrade
        restricted_cap = DeclassificationCapability(
            issuer="admin",
            purpose="audit",
            scope="global",
            expiry=now + 100.0,
            max_downgrade=StateLabel.CONFIDENTIAL,
        )
        # Attempting to declassify to PUBLIC (< CONFIDENTIAL) is rejected
        self.assertFalse(lat.can_declassify(StateLabel.PRIVATE, StateLabel.PUBLIC, capability=restricted_cap, current_time=now))

    def test_recovery_policies(self):
        """Verify RecoveryPolicy implementations (Phase 3)."""
        pivot_policy = SemanticPivotRecovery(pivot_text="[PIVOT_OVERRIDE]", max_pivots=2)
        priming_ids, cont = pivot_policy.on_violation(
            self.model, self.tokenizer, 0, None, self.device
        )
        self.assertTrue(cont)
        self.assertIsNotNone(priming_ids)

        adapter_policy = AdapterSwitchRecovery()
        refusal_ids, cont = adapter_policy.on_violation(
            self.model, self.tokenizer, 0, None, self.device
        )
        self.assertTrue(cont)
        self.assertIsNotNone(refusal_ids)

        halt_policy = HaltRecovery()
        halt_ids, cont = halt_policy.on_violation(self.model, self.tokenizer, 0, None, self.device)
        self.assertFalse(cont)
        self.assertIsNone(halt_ids)


if __name__ == "__main__":
    unittest.main()
