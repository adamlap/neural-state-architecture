"""
tests/test_belief_state_ablation.py
===================================
Unit tests for NSA 5.1 Belief-State Dynamics & 6-Arm Controlled Cognitive Ablation.
"""

from __future__ import annotations

from experiments.nsa51.ablation_suite import run_controlled_ablation_benchmark
from experiments.nsa51.environments.ambiguous_belief_world import AmbiguousBeliefWorld
from nsa.cognition.belief_state import (
    BeliefState,
    InformationGainSelector,
    WorldHypothesis,
)


def test_belief_state_entropy_and_update():
    hypotheses = [
        WorldHypothesis("h1", "Hypothesis 1", 0.5, ["obs_a"], "act_1"),
        WorldHypothesis("h2", "Hypothesis 2", 0.5, ["obs_b"], "act_2"),
    ]
    b = BeliefState(hypotheses=hypotheses)
    assert abs(b.entropy - 1.0) < 1e-4

    # Update with observation 'obs_a'
    b.update_with_observation("obs_a")
    assert b.hypotheses[0].probability > b.hypotheses[1].probability
    assert b.entropy < 1.0


def test_information_gain_selection():
    hypotheses = [
        WorldHypothesis("h1", "Hypothesis 1", 0.5, ["obs_a"], "act_1"),
        WorldHypothesis("h2", "Hypothesis 2", 0.5, ["obs_b"], "act_2"),
    ]
    b = BeliefState(hypotheses=hypotheses)

    info_gain = InformationGainSelector.calculate_information_gain(
        b, "probe_a", {"probe_a": "obs_a"}
    )
    assert info_gain > 0.5

    score = InformationGainSelector.score_action(
        action_name="probe_a",
        expected_utility=0.8,
        risk_level=0.1,
        info_gain=info_gain,
    )
    assert score > 1.0


def test_6_arm_controlled_ablation_suite():
    res = run_controlled_ablation_benchmark(trials_per_scenario=5, seed=42)
    sci = res["scientific_conclusions"]

    assert sci["hypothesis_empirically_confirmed"] is True
    assert sci["substrate_isolated_from_raw_compute"] is True
    assert sci["zero_violations_strictly_maintained"] is True

    # Check that NSA Full Substrate achieves 100% GTC with 0 violations
    nsa_arm = res["results_by_arm"]["Arm_F_NSA_5_1_Full_Substrate"]
    assert nsa_arm["gtc_rate"] == 1.0
    assert nsa_arm["violations"] == 0
    assert nsa_arm["human_interventions"] == 0
