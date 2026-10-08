from datetime import datetime, timedelta, timezone

import pytest

from nsa.memory import MemoryItem, MemoryStore, TemporalMemoryStore


def test_memory_is_append_only():
    item = MemoryItem("m1", "hello", "fact", provenance_ids=("claim-1",))
    store = MemoryStore().write(item)
    assert store.items == (item,)
    with pytest.raises(ValueError):
        store.write(item)


def test_expired_memory_is_not_active():
    item = MemoryItem(
        "m1",
        "temporary",
        "observation",
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    assert MemoryStore().write(item).active() == ()


def test_memory_retains_provenance_reference():
    item = MemoryItem("m1", {"x": 1}, "derived", provenance_ids=("c1", "c2"))
    assert item.provenance_ids == ("c1", "c2")


def test_temporal_memory_supersedes_without_deleting_history():
    memory = TemporalMemoryStore()
    memory = memory.remember("temperature", {"value": "20"}, provenance_ids=("obs-1",))
    memory = memory.remember("temperature", {"value": "25"}, kind="update", provenance_ids=("obs-2",))
    current = memory.current("temperature")
    assert current is not None
    assert current.content["value"] == "25"
    assert current.content["supersedes_id"] == "tm-00000000"
    assert len(memory.history("temperature")) == 2


def test_temporal_retrieval_returns_only_canonical_versions():
    memory = TemporalMemoryStore()
    memory = memory.remember("a", {"value": "old"})
    memory = memory.remember("a", {"value": "new"}, kind="update")
    memory = memory.remember("b", {"value": "other"})
    items = memory.retrieve(["a", "b"], limit=3)
    assert {item.content["value"] for item in items} == {"new", "other"}
    assert all(item.content["value"] != "old" for item in items)


def test_temporal_render_is_bounded():
    memory = TemporalMemoryStore()
    for i in range(1000):
        memory = memory.remember(f"item-{i}", {"value": str(i)})
    rendered = memory.render(["item-1", "item-500", "item-999"], limit=2)
    assert rendered.count("\n") == 1
