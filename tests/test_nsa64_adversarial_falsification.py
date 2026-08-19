"""
tests/test_nsa64_adversarial_falsification.py
=============================================
Unit and Stress Tests for NSA 6.4 Adversarial Scientific Falsification Suite.
"""

from experiments.nsa64.environments.adversarial_environments import (
    AdversarialClass,
    AdversarialFalsificationWorld,
)
from experiments.nsa64.falsification_suite import (
    NSA64FalsificationRunner,
    run_full_falsification_suite,
)


def test_adversarial_world_generation_across_classes():
    for env_cls in AdversarialClass:
        world = AdversarialFalsificationWorld(env_class=env_cls, seed=42)
        assert world.env_class == env_cls
        assert world.spec.safe_discriminating_probe is not None
        assert len(world.available_tools) >= 4


def test_class_b_deceptive_high_entropy_probe_defense():
    runner = NSA64FalsificationRunner()
    res = runner.run_class_b_deceptive_test(trials=4, seed=42)
    assert res["unconstrained_search_violations"] == 4
    assert res["nsa_violations"] == 0
    assert res["nsa_recovery_rate"] == 1.0
    assert res["falsification_outcome"] == "FALSIFICATION_RESISTED"


def test_full_falsification_suite_runner():
    report = run_full_falsification_suite(trials=2, seed=42)
    assert report["suite"] == "NSA 6.4 Adversarial Scientific Falsification Suite"
    assert report["overall_status"] == "FALSIFICATION_RESISTED"
