from experiments.research_v3.temporal_memory_benchmark import (
    build_append_memory,
    build_temporal_memory,
    make_episode,
    append_retrieve,
    temporal_retrieve,
    temporal_stale,
)
from experiments.research_common.episodes import target_occurrences
from nsa.memory.temporal import TemporalMemoryStore


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
    # append-only baseline deliberately exposes the superseded version too
    assert len(items) == 2
    assert not items[0].content["value"].endswith("_CURRENT")
    assert items[-1].content["value"].endswith("_CURRENT")


def test_temporal_multi_key_retrieval_is_exactly_bounded():
    observations, query, expected, _, _ = make_episode(137, "multi_current", 1000)
    store = build_temporal_memory(observations)
    keys = [o.split()[1] for o in observations[:3]]
    items = store.retrieve(keys, limit=3)
    assert len(items) == 3
    assert {i.content["value"] for i in items} == set(expected.split("|"))


def test_unique_generator_never_repeats_target_key():
    for d in (0, 10, 64, 200, 1000):
        for task in ("recall", "interference", "supersession_current"):
            observations, _, _, key, _ = make_episode(5, task, d)
            allowed = 2 if task == "supersession_current" else 1
            assert len(target_occurrences(observations, key)) == allowed


def test_legacy_generator_reproduces_wraparound():
    observations, _, _, key, _ = make_episode(5, "recall", 200, generator="legacy")
    assert len(target_occurrences(observations, key)) > 1


def test_temporal_stale_is_measured_not_assumed():
    store = TemporalMemoryStore()
    store = store.remember("K", {"value": "A"}, kind="fact")
    store = store.remember("K", {"value": "B"}, kind="update")
    assert not temporal_stale(store.retrieve(["K"]))
    store = store.remember("K", {"value": "A"}, kind="fact")
    assert temporal_stale(store.retrieve(["K"]))
