"""
tests/test_adversarial_suite.py
===============================
Automated verification of the NSA 2.0 Adversarial Red-Team Suite.

Asserts 0.00% Attack Success Rate (ASR) across all 6 threat vectors:
1. Semantic Privilege Escalation
2. Cryptographic Capability Forgery
3. Parameter Tampering & Downgrade
4. Nonce Replay & Ticket Reuse
5. Rollback State Desynchronization
6. State Laundering & Sink Leakage
"""

import unittest
from prototype.security.adversarial_suite import AdversarialBenchmarkSuite


class TestAdversarialSecuritySuite(unittest.TestCase):
    """Adversarial security validation test suite."""

    def setUp(self):
        self.suite = AdversarialBenchmarkSuite()

    def test_vector_1_semantic_privilege_escalation(self):
        res = self.suite.attack_1_semantic_privilege_escalation(trials=20)
        self.assertEqual(res.asr, 0.0, f"Vector 1 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)

    def test_vector_2_capability_forgery(self):
        res = self.suite.attack_2_capability_forgery(trials=20)
        self.assertEqual(res.asr, 0.0, f"Vector 2 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)

    def test_vector_3_parameter_tampering(self):
        res = self.suite.attack_3_parameter_tampering(trials=20)
        self.assertEqual(res.asr, 0.0, f"Vector 3 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)

    def test_vector_4_nonce_replay(self):
        res = self.suite.attack_4_nonce_replay(trials=20)
        self.assertEqual(res.asr, 0.0, f"Vector 4 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)

    def test_vector_5_rollback_desynchronization(self):
        res = self.suite.attack_5_rollback_desynchronization(trials=10)
        self.assertEqual(res.asr, 0.0, f"Vector 5 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)

    def test_vector_6_state_laundering(self):
        res = self.suite.attack_6_state_laundering_and_sink_leakage(trials=20)
        self.assertEqual(res.asr, 0.0, f"Vector 6 Vulnerable! ASR={res.asr}%")
        self.assertEqual(res.successes, 0)


if __name__ == "__main__":
    unittest.main()
