from experiments.research_v3.temporal_memory_benchmark import (
    build_append_memory,
    build_temporal_memory,
    make_episode,
    append_retrieve,
    temporal_retrieve,
)


def test_temporal_current_value_excludes_superseded_version():
    observations, query, expected, key, _ = make_episode(7, "supersession_current", 1000)
    store = build_temporal_memory(observations)
    items = temporal_retrieve(store, [key], history=False, limit=3)
    assert len(items) == 1
    assert items[0].content["value"] == expected


def test_temporal_history_returns_previous_value():
    observations, query, expected, key, history = make_episode(17, "supersession_history", 1000)
    store = build_temporal_memory(observations)
    items = temporal_retrieve(store, [key], history=history, limit=3)
    assert items
    assert items[0].content["value"] == expected


def test_temporal_memory_stays_bounded_with_1000_distractors():
    observations, _, _, key, _ = make_episode(37, "recall", 1000)
    store = build_temporal_memory(observations)
    items = temporal_retrieve(store, [key], history=False, limit=3)
    assert len(store.store.items) == 1001
    assert len(items) == 1


def test_append_memory_keeps_superseded_history_for_current_queries():
    observations, _, _, key, _ = make_episode(73, "supersession_current", 250)
    store = build_append_memory(observations)
    items = append_retrieve(store, [key], history=False, limit=3)
    assert len(items) == 1
    assert items[0].content["value"].endswith("_CURRENT")


def test_temporal_multi_key_retrieval_is_exactly_bounded():
    observations, _, _, _, _ = make_episode(137, "multi_current", 1000)
    store = build_temporal_memory(observations)
    keys = [x for x in ("ITEM_00", "ITEM_01", "ITEM_02") if store.current(x)]
    items = store.retrieve(keys, limit=3)
    assert len(items) == 3
