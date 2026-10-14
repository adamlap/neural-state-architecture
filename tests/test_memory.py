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


def test_temporal_render_labels_current_and_history_semantics():
    memory = TemporalMemoryStore()
    memory = memory.remember("status", {"value": "old"})
    memory = memory.remember("status", {"value": "new"}, kind="update")

    current = memory.render(["status"])
    assert "status: new" in current
    assert "VERSION=2" in current
    assert "STATUS=CURRENT" in current
    assert "old" not in current

    history = memory.render(["status"], mode="history")
    assert "status: old" in history
    assert "VERSION=1" in history
    assert "STATUS=SUPERSEDED" in history
    assert "CURRENT_VALUE" not in history and "new" not in history
    assert "CURRENT_VALUE=new" in memory.render(["status"], mode="history", include_current=True)


def test_temporal_store_flags_reassertion_of_a_superseded_value():
    """A later plain fact that restates an old value silently becomes CURRENT under
    last-write-wins; the store now records that instead of hiding it."""
    memory = TemporalMemoryStore()
    memory = memory.remember("k", {"value": "old"})
    memory = memory.remember("k", {"value": "new"}, kind="update")
    assert memory.conflicts() == ()
    memory = memory.remember("k", {"value": "old"}, kind="fact")
    flagged = memory.conflicts()
    assert len(flagged) == 1
    assert flagged[0].content["version"] == 3
    assert flagged[0].content["reasserts_version"] == 1
    assert flagged[0].content["overrides_update"] is True
    assert memory.current("k").content["value"] == "old"  # semantics unchanged: last write wins


def test_temporal_store_does_not_flag_repeating_the_current_value_or_new_values():
    memory = TemporalMemoryStore().remember("k", {"value": "a"}).remember("k", {"value": "a"}).remember("k", {"value": "b"})
    assert memory.conflicts() == ()


def test_temporal_store_lookup_matches_a_full_scan_and_is_case_insensitive():
    memory = TemporalMemoryStore()
    for i in range(300):
        memory = memory.remember(f"Key-{i % 7}", {"value": i})
    for k in range(7):
        scan = [it for it in memory.store.items if it.content["memory_key"].lower() == f"key-{k}"]
        assert memory.versions(f"KEY-{k}") == tuple(scan)
        assert memory.current(f"key-{k}") is scan[-1]
    assert memory.current("absent") is None


def test_temporal_store_respects_expiry_for_current_and_versioning():
    from datetime import datetime, timedelta, timezone
    memory = TemporalMemoryStore().remember("k", {"value": "a"})
    assert memory.current("k").content["version"] == 1
    expired = memory.store.items[0]
    object.__setattr__(expired, "expires_at", datetime.now(timezone.utc) - timedelta(seconds=1))
    assert memory.current("k") is None
    assert memory.remember("k", {"value": "b"}).current("k").content["version"] == 1


def test_memory_store_still_rejects_duplicate_ids_after_many_writes():
    import pytest
    from nsa.memory import MemoryItem, MemoryStore
    store = MemoryStore()
    for i in range(50):
        store = store.write(MemoryItem(f"m{i}", {"v": i}, "fact"))
    with pytest.raises(ValueError, match="duplicate"):
        store.write(MemoryItem("m7", {}, "fact"))
    assert len(store.items) == 50 and store == MemoryStore(store.items)


def test_untrusted_write_cannot_supersede_a_trusted_value():
    from nsa.algebra import IntegrityLabel

    memory = TemporalMemoryStore()
    memory = memory.remember("owner", {"value": "alice"})
    memory = memory.remember("owner", {"value": "mallory"}, kind="update", integrity=IntegrityLabel.UNTRUSTED)

    assert memory.current("owner").content["value"] == "alice"
    assert "mallory" not in memory.render(["owner"])
    assert "mallory" not in memory.render(["owner"], mode="history")
    assert [i.content["value"] for i in memory.history("owner")] == ["alice"]
    # audit trail keeps it, flagged, and it never entered the version chain
    (blocked,) = memory.quarantined()
    assert blocked.content["value"] == "mallory" and blocked.content["blocked_version"] == 1
    assert len(memory.store.items) == 2


def test_trusted_write_still_supersedes_and_untrusted_can_update_untrusted():
    from nsa.algebra import IntegrityLabel

    memory = TemporalMemoryStore().remember("k", {"value": "u1"}, integrity=IntegrityLabel.UNTRUSTED)
    memory = memory.remember("k", {"value": "u2"}, kind="update", integrity=IntegrityLabel.UNTRUSTED)
    assert memory.current("k").content["value"] == "u2" and not memory.quarantined()
    memory = memory.remember("k", {"value": "t1"}, kind="update")          # trusted wins
    assert memory.current("k").content["value"] == "t1" and memory.current("k").content["version"] == 3
    memory = memory.remember("k", {"value": "u3"}, kind="update", integrity=IntegrityLabel.UNTRUSTED)
    assert memory.current("k").content["value"] == "t1" and len(memory.quarantined()) == 1


def test_quarantine_survives_rebuilding_the_index_from_items():
    from nsa.algebra import IntegrityLabel

    memory = TemporalMemoryStore().remember("k", {"value": "a"})
    memory = memory.remember("k", {"value": "evil"}, integrity=IntegrityLabel.UNTRUSTED)
    rebuilt = TemporalMemoryStore(store=memory.store)
    assert rebuilt.current("k").content["value"] == "a" and len(rebuilt.quarantined()) == 1


def test_default_integrity_is_trusted_so_existing_callers_are_unchanged():
    memory = TemporalMemoryStore().remember("k", {"value": "a"}).remember("k", {"value": "b"}, kind="update")
    assert memory.current("k").content["value"] == "b" and not memory.quarantined()
