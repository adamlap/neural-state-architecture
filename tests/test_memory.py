from datetime import datetime, timedelta, timezone

import pytest

from nsa.memory import MemoryItem, MemoryStore


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
