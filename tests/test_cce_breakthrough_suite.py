from experiments.breakthrough.cce_breakthrough_suite import (
    authority_invariance,
    evaluate_predictor,
    four_way_cognition,
    run_suite,
    state_ablation,
)


def test_predictor_is_evaluated_on_held_out_trajectory():
    result = evaluate_predictor(7, epochs=20)
    assert result["held_out"]["mse"] >= 0.0
    assert result["held_out"]["persistence_mse"] >= 0.0
    assert "beats_persistence" in result


def test_four_way_control_has_all_conditions():
    result = four_way_cognition(7)
    assert set(result["mse_lower_is_better"]) == {
        "stateless", "persistent", "clocked_cce", "continuous_predictive_cce"
    }


def test_state_ablation_is_complete():
    result = state_ablation(7)
    assert set(result["mse_lower_is_better"]) == {
        "state_dims_0", "state_dims_2", "state_dims_4", "state_dims_6", "state_dims_8"
    }


def test_hard_authority_never_transfers_to_continuous_state():
    result = authority_invariance(ticks=25)
    assert result["zero_violation"]
    assert result["hard_state_unchanged"]


def test_suite_emits_all_scientific_gates():
    report = run_suite([7], predictor_epochs=10)
    assert set(report["gates"]) == {
        "predictor_beats_persistence_all_seeds",
        "continuous_beats_stateless_mean",
        "continuous_beats_persistent_mean",
        "authority_zero_violation",
    }
    assert report["overall_status"] in {"PASS", "RESEARCH_GATE_NOT_YET_MET"}
