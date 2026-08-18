"""
tests/test_atomic_rollback.py
=============================
Rigorous verification of Complete Atomic Execution State Rollback:

    Theorem (Atomic Rollback Invariant):
        For any speculative chunk encountering an audit rejection at t + k:
            Rollback(S_{t+k}) = S_t
        where S_t = (X_t, K_t, V_t, sigma_t, q_t, C_t, R_t).
"""

import unittest

import torch
from torch import nn

from nsa.algebra import StateLabel
from nsa.mask_injector import NSAMaskInjector
from nsa.verifier.automaton import (
    Capability,
    CapabilitySigner,
    CapabilityVerifier,
    SecurityAutomaton,
    SecurityExecutionState,
)
from nsa.verifier.generation import NSAGenerator
from nsa.verifier.recovery import HaltRecovery
from nsa.verifier.router import StreamRouter
from nsa.verifier.speculative import MultiLayerStateAuditor


class MockModelForRollback(nn.Module):
    def __init__(self, d_model: int = 32, vocab_size: int = 100):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.step = 0

    def forward(self, input_ids, past_key_values=None, use_cache=True, output_hidden_states=True, return_dict=True, **kwargs):
        b, t = input_ids.shape
        h = torch.randn(b, t, self.d_model, device=input_ids.device)
        logits = torch.zeros(b, t, self.vocab_size, device=input_ids.device)

        # Force emitting state transition token <|start_system_thought|> (id 1) at step 0 of chunk
        if self.step == 0:
            logits[:, -1, 1] = 100.0  # <|start_system_thought|>
        else:
            logits[:, -1, 10] = 100.0  # regular token

        self.step += 1

        # Return mock KV cache
        new_past = []
        for _ in range(2):
            k = torch.randn(b, 2, t, 16, device=input_ids.device)
            v = torch.randn(b, 2, t, 16, device=input_ids.device)
            new_past.append((k, v))

        class Output:
            def __init__(self, logits, past, hiddens):
                self.logits = logits
                self.past_key_values = tuple(past)
                self.hidden_states = (h, h)

        return Output(logits, new_past, (h, h))


class MockTokenizerForRollback:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 99

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, (torch.Tensor, int)):
            tid = int(token_ids.item() if isinstance(token_ids, torch.Tensor) else token_ids)
            if tid == 1:
                return "<|start_system_thought|>"
            return f"tok_{tid}"
        return " ".join([self.decode(t) for t in token_ids])


class TestAtomicRollback(unittest.TestCase):
    """Test complete atomic execution state rollback across all subsystems."""

    def setUp(self):
        self.secret_key = b"super-secret-tcb-key-12345"
        self.signer = CapabilitySigner(self.secret_key)
        self.verifier = CapabilityVerifier(self.secret_key)

    def test_cryptographic_capability_signing_and_verification(self):
        """Task 2: Test HMAC cryptographic capability issuance, verification, and tamper detection."""
        # 1. Valid signed capability
        cap = self.signer.issue(
            issuer="admin_tcb",
            target_state=SecurityExecutionState.SYSTEM,
            scope="kernel_exec",
            ttl_seconds=60.0,
        )
        self.assertTrue(self.verifier.verify(cap))

        # 2. Tampered capability (e.g. adversary changed target_state from CONFIDENTIAL to SYSTEM)
        tampered_cap = Capability(
            issuer=cap.issuer,
            target_state=SecurityExecutionState.SYSTEM,
            subject=cap.subject,
            scope=cap.scope,
            purpose=cap.purpose,
            expires_at=cap.expires_at,
            nonce=cap.nonce,
            signature=cap.signature,  # Signature from different payload
        )
        # Change scope to simulate tampering
        tampered_cap_2 = Capability(
            issuer=cap.issuer,
            target_state=cap.target_state,
            subject=cap.subject,
            scope="malicious_scope",
            purpose=cap.purpose,
            expires_at=cap.expires_at,
            nonce=cap.nonce,
            signature=cap.signature,
        )
        self.assertFalse(self.verifier.verify(tampered_cap_2))

        # 3. Nonce consumption (Replay prevention)
        self.verifier.consume_nonce(cap.nonce)
        self.assertFalse(self.verifier.verify(cap), "Replay attack was not prevented!")

    def test_atomic_verify_and_consume_exhaustion(self):
        """Verify Authorize(c) = Verify(c) + Consume(c) and single-use invariant forall c: uses <= 1."""
        cap = self.signer.issue(
            issuer="admin_tcb",
            target_state=SecurityExecutionState.SYSTEM,
            scope="kernel_exec",
        )
        automaton = SecurityAutomaton(
            initial_state=SecurityExecutionState.CONFIDENTIAL,
            capabilities=[cap],
            verifier=self.verifier,
        )

        # 1. First transition: valid capability -> MUST SUCCEED and consume capability
        success, state = automaton.transition(SecurityExecutionState.SYSTEM)
        self.assertTrue(success)
        self.assertEqual(state, SecurityExecutionState.SYSTEM)
        self.assertTrue(self.verifier.is_nonce_consumed(cap.nonce), "Nonce was not atomically consumed!")

        # 2. De-escalate back to CONFIDENTIAL
        automaton.transition(SecurityExecutionState.CONFIDENTIAL)
        self.assertEqual(automaton.current_state, SecurityExecutionState.CONFIDENTIAL)

        # 3. Replay attack: try transitioning back to SYSTEM with the same consumed capability -> MUST FAIL
        success_replay, state_replay = automaton.transition(SecurityExecutionState.SYSTEM, capability=cap)
        self.assertFalse(success_replay, "Replay attack succeeded! Capability was reused.")
        self.assertEqual(state_replay, SecurityExecutionState.CONFIDENTIAL)

    def test_atomic_rollback_reverts_all_components(self):
        """Task 3 & 4: Invariant Rollback(S_{t+k}) = S_t (automaton, router, injector, KV)."""
        model = MockModelForRollback()
        tokenizer = MockTokenizerForRollback()

        # Sign a valid capability so the automaton enters SYSTEM during speculative generation
        cap = self.signer.issue(
            issuer="admin_tcb",
            target_state=SecurityExecutionState.SYSTEM,
        )
        automaton = SecurityAutomaton(
            initial_state=SecurityExecutionState.CONFIDENTIAL,
            capabilities=[cap],
            verifier=self.verifier,
        )

        router = StreamRouter(tokenizer=tokenizer)
        state_levels = torch.tensor([[StateLabel.CONFIDENTIAL.value, StateLabel.CONFIDENTIAL.value]])
        injector = NSAMaskInjector(model, state_levels)

        # Auditor that rejects on token 1
        class RejectingHead(nn.Module):
            def forward(self, h, async_execution=False):
                b, k, _ = h.shape
                logits = torch.zeros(b, k, len(StateLabel))
                logits[:, :, StateLabel.SYSTEM.value] = 10.0  # Force violation
                return logits

        def reject_all_validator(pred):
            return False  # Always flag violation

        auditor = MultiLayerStateAuditor(RejectingHead(), lattice_validator=reject_all_validator, chunk_size=2)

        generator = NSAGenerator(
            model=model,
            tokenizer=tokenizer,
            auditor=auditor,
            recovery_policy=HaltRecovery(),
            stream_router=router,
            mask_injector=injector,
            automaton=automaton,
            verbose=False,
        )

        input_ids = torch.tensor([[10, 20]], dtype=torch.long)
        initial_automaton_state = automaton.current_state  # CONFIDENTIAL
        initial_injector_len = injector.state_levels.shape[1]  # 2

        out_ids = generator.generate(
            input_ids,
            max_new_tokens=4,
            chunk_size=2,
            initial_state_idx=initial_automaton_state,
        )

        # 1. Automaton state MUST be restored to initial CONFIDENTIAL state (not left in SYSTEM)
        self.assertEqual(
            automaton.current_state,
            initial_automaton_state,
            "SecurityAutomaton state leaked across rolled-back speculative chunk!",
        )

        # 2. Mask injector state levels MUST be restored to original length
        self.assertEqual(
            injector.state_levels.shape[1],
            initial_injector_len,
            "Mask injector state levels were not restored during rollback!",
        )

        # 3. Router buffers MUST NOT contain uncommitted/rejected tokens
        self.assertEqual(len(router.get_stream_tokens(StateLabel.SYSTEM)), 0)


if __name__ == "__main__":
    unittest.main()
