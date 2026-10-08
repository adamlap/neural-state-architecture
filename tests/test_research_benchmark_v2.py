from experiments.research_v2.selective_memory_benchmark import build_memory, make_episode, retrieve, extract


def test_memory_retrieval_preserves_latest_superseding_value():
    observations, query, expected, _ = make_episode(7, "supersession", 100)
    store = build_memory(observations)
    items = retrieve(store, query, 3)
    assert items
    assert items[-1].content["value"] == expected
    assert any(item.content["kind"] == "fact" for item in items[:-1])


def test_memory_retrieval_is_constant_size_with_long_distraction_stream():
    observations, query, _, _ = make_episode(17, "recall", 1000)
    store = build_memory(observations)
    items = retrieve(store, query, 3)
    assert len(items) == 1
    assert len(store.items) == 1001


def test_interference_does_not_change_target_retrieval():
    observations, query, expected, _ = make_episode(37, "interference", 1000)
    store = build_memory(observations)
    items = retrieve(store, query, 3)
    assert items[-1].content["value"] == expected


def test_extract_is_strict_and_case_insensitive():
    assert extract("VALUE=V007_001") == "V007_001"
    assert extract("answer: value=v007_001") == "V007_001"
    assert extract("no structured answer") is None
