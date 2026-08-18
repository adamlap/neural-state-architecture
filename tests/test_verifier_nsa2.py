"""
tests/test_verifier_nsa2.py
===========================
Comprehensive unit test suite for NSA 2.0 Architectural Features:
1. First-Class NSAMaskInjector (Static & Dynamic Attention Masking)
2. StateControlTokens (Dynamic Tag Recognition & Registry)
3. StreamRouter (Clearance-Aware Multi-Stream Dispatch)
4. RecoveryPolicy & Native Recovery Adapters
5. SpeculativeStateAuditor & MultiLayerStateAuditor (Early-Exit Residual Probing)
6. NSAGenerator Speculative Rollback Engine
"""

import unittest
from typing import Any, List, Optional

import torch
from torch import nn

from nsa.algebra import StateLabel
from nsa.mask_injector import NSAMaskInjector
from nsa.verifier.encoder_head import StateEncoderHead
from nsa.verifier.generation import NSAGenerator
from nsa.verifier.recovery import (
    AdapterSwitchRecovery,
    HaltRecovery,
    SemanticPivotRecovery,
)
from nsa.verifier.router import StreamRouter
from nsa.verifier.speculative import (
    AuditResult,
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
        # Return hidden_states unchanged
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

        # Mock tuple past_key_values
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
        self, text: str, add_special_tokens: bool = False, return_tensors: Optional[str] = None
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
        self.model = MockTransformerModel(d_model=32, num_layers=3, vocab_size=100).to(self.device)
        self.tokenizer = MockTokenizer()

    def test_mask_injector_lifecycle_and_dynamic_update(self):
        """Verify NSAMaskInjector hook injection, mask slicing, and dynamic expansion."""
        state_levels = torch.tensor(
            [[StateLabel.PUBLIC.value, StateLabel.SYSTEM.value, StateLabel.CONFIDENTIAL.value]]
        )
        injector = NSAMaskInjector(self.model, state_levels, decode_row_idx=0, gate_mode="hard")

        # Before enter, nsa_mask is None
        self.assertIsNone(injector.nsa_mask)
        self.assertEqual(len(injector._hooks), 0)

        with injector:
            self.assertIsNotNone(injector.nsa_mask)
            self.assertEqual(injector.nsa_mask.shape, (1, 1, 3, 3))
            self.assertEqual(len(injector._hooks), 3)

            # Test dynamic update_state (Phase 1)
            injector.update_state(StateLabel.SYSTEM.value)
            self.assertEqual(injector.state_levels.shape, (1, 4))
            self.assertEqual(injector.nsa_mask.shape, (1, 1, 4, 4))
            self.assertEqual(injector.decode_row_idx, 3)

        # After exit, hooks and mask are cleaned up
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

        # Register tokens
        added = StateControlTokens.register(self.tokenizer)
        self.assertGreater(added, 0)

    def test_stream_router_clearance_routing(self):
        """Verify StreamRouter clearance-aware multi-stream dispatching (Phase 4)."""
        router = StreamRouter(tokenizer=self.tokenizer)
        public_received = []
        system_received = []

        router.register_sink(
            StateLabel.PUBLIC, lambda text, tid: public_received.append((text, tid))
        )
        router.register_sink(
            StateLabel.SYSTEM, lambda text, tid: system_received.append((text, tid))
        )

        # Route a public token (id=10)
        router.route_token(10, StateLabel.PUBLIC)
        # Route a system token (id=5 -> SECRET_KEY)
        router.route_token(5, StateLabel.SYSTEM)

        self.assertEqual(len(public_received), 1)
        self.assertEqual(public_received[0][1], 10)

        self.assertEqual(len(system_received), 1)
        self.assertEqual(system_received[0][1], 5)
        self.assertIn("SECRET_KEY", system_received[0][0])

        self.assertEqual(router.get_stream_tokens(StateLabel.SYSTEM), [5])
        self.assertEqual(router.get_stream_tokens(StateLabel.PUBLIC), [10])

    def test_recovery_policies(self):
        """Verify RecoveryPolicy implementations (Phase 3)."""
        # Semantic pivot
        pivot_policy = SemanticPivotRecovery(pivot_text="[PIVOT_OVERRIDE]", max_pivots=2)
        priming_ids, cont = pivot_policy.on_violation(
            self.model, self.tokenizer, 0, None, self.device
        )
        self.assertTrue(cont)
        self.assertIsNotNone(priming_ids)

        # Adapter switch
        adapter_policy = AdapterSwitchRecovery()
        refusal_ids, cont = adapter_policy.on_violation(
            self.model, self.tokenizer, 0, None, self.device
        )
        self.assertTrue(cont)
        self.assertIsNotNone(refusal_ids)

        # Halt
        halt_policy = HaltRecovery()
        halt_ids, cont = halt_policy.on_violation(self.model, self.tokenizer, 0, None, self.device)
        self.assertFalse(cont)
        self.assertIsNone(halt_ids)

    def test_speculative_auditor_and_multi_layer_early_exit(self):
        """Verify MultiLayerStateAuditor early-exit detection across intermediate layers (Phase 2)."""
        encoder_head = StateEncoderHead(
            hidden_size=32, num_states=len(StateLabel), use_bidirectional=False
        )

        # Custom validator: flags SYSTEM as violation
        def validator(pred: int) -> bool:
            return pred != StateLabel.SYSTEM.value

        auditor = MultiLayerStateAuditor(
            encoder_head, lattice_validator=validator, chunk_size=4, probe_layers=[-1, 1, 2]
        )

        # Test single layer 3D tensor
        hidden_3d = torch.randn(1, 4, 32)
        res_3d = auditor.audit_chunk(hidden_3d, StateLabel.CONFIDENTIAL)
        self.assertIsInstance(res_3d, AuditResult)

        # Test multi-layer 4D tensor with early exit
        # Create a mock head that outputs SYSTEM for layer index 1
        class MockLayerFlaggingHead(nn.Module):
            def forward(self, h, async_execution=False):
                b, k, _dim = h.shape
                logits = torch.zeros(b, k, len(StateLabel))
                # Set high score on SYSTEM (5)
                logits[:, :, StateLabel.SYSTEM.value] = 10.0
                return logits

        flagging_auditor = MultiLayerStateAuditor(
            MockLayerFlaggingHead(),
            lattice_validator=validator,
            chunk_size=4,
            probe_layers=[0, 12, 24],
        )

        hidden_4d = torch.randn(1, 4, 3, 32)  # [batch, K, num_layers, hidden_size]
        res_4d = flagging_auditor.audit_chunk(hidden_4d, StateLabel.CONFIDENTIAL)
        self.assertFalse(res_4d.is_valid)
        self.assertEqual(res_4d.violation_token_idx, 0)
        self.assertEqual(res_4d.violation_layer, 0)

        # Test tuple unpacking compatibility
        is_valid, violation_idx, predicted, layer = res_4d
        self.assertFalse(is_valid)
        self.assertEqual(violation_idx, 0)

    def test_nsa_generator_integration(self):
        """Verify NSAGenerator end-to-end loop with dynamic tracking, router, and recovery."""
        encoder_head = StateEncoderHead(
            hidden_size=32, num_states=len(StateLabel), use_bidirectional=False
        )
        auditor = SpeculativeStateAuditor(encoder_head, chunk_size=2)
        router = StreamRouter(tokenizer=self.tokenizer)

        input_ids = torch.tensor([[10, 11]], dtype=torch.long)
        state_levels = torch.tensor(
            [[StateLabel.CONFIDENTIAL.value, StateLabel.CONFIDENTIAL.value]]
        )
        injector = NSAMaskInjector(self.model, state_levels)

        generator = NSAGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            auditor=auditor,
            recovery_policy=AdapterSwitchRecovery(),
            stream_router=router,
            mask_injector=injector,
            verbose=False,
        )

        out_ids = generator.generate(input_ids, max_new_tokens=4, chunk_size=2)
        self.assertGreater(out_ids.shape[1], input_ids.shape[1])


if __name__ == "__main__":
    unittest.main()
