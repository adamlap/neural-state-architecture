from experiments.live.cce_matched_cognitive_loop_study import _extract_feedback


def test_extract_feedback_accepts_typed_finite_shape():
    proposal = _extract_feedback(
        '{"working_delta":[0.1,0.0,-0.1,0.2],"goal_delta":[0,0,0,0],"confidence":0.8}',
        4,
    )
    assert proposal is not None
    assert proposal.source == "ollama"
    assert proposal.confidence == 0.8


def test_extract_feedback_rejects_wrong_dimension_and_malformed_json():
    assert _extract_feedback('{"working_delta":[1,2]}', 4) is None
    assert _extract_feedback('not json', 4) is None


def test_extract_feedback_does_not_create_hard_authority_path():
    proposal = _extract_feedback(
        '{"working_delta":[0,0,0,0],"confidence":1.0}',
        4,
    )
    assert proposal is not None
    assert not hasattr(proposal, "authority")
