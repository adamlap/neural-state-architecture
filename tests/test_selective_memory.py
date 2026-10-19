import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone

from nsa.memory.model import MemoryItem, MemoryStore
from nsa.server import proxy


def _runtime(max_items=256, ttl=3600.0):
    runtime = proxy.NSAProxyRuntime.__new__(proxy.NSAProxyRuntime)
    runtime.memory_store = MemoryStore()
    runtime._memory_lock = threading.Lock()
    runtime._system_one_last_tick = None
    runtime.MEMORY_MAX_ITEMS = max_items
    runtime.MEMORY_TTL_SECONDS = ttl
    return runtime


def _chat(n):
    msgs = []
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append({"role": "assistant", "content": f"answer {i}"})
    return msgs


def test_repeating_the_same_conversation_adds_no_duplicates():
    runtime = _runtime()
    msgs = _chat(10)
    for turn in range(1, 11):
        runtime._remember_selectively(msgs[: turn * 2], msgs[turn * 2 - 2]["content"])
    contents = [i.content for i in runtime.memory_store.items]
    assert len(contents) == len(set(contents)) == 20
    count = len(runtime.memory_store.items)
    runtime._remember_selectively(msgs, "x")
    assert len(runtime.memory_store.items) == count


def test_same_text_with_different_roles_is_stored_separately():
    runtime = _runtime()
    runtime._remember_selectively([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hi"}], "hi")
    assert len(runtime.memory_store.items) == 2


def test_ids_are_stable_across_processes():
    code = (
        "import threading\n"
        "from nsa.memory.model import MemoryStore\n"
        "from nsa.server import proxy\n"
        "r = proxy.NSAProxyRuntime.__new__(proxy.NSAProxyRuntime)\n"
        "r.memory_store = MemoryStore(); r._memory_lock = threading.Lock(); r._system_one_last_tick = None\n"
        "r._remember_selectively([{'role': 'user', 'content': 'hello'}], 'hello')\n"
        "print(r.memory_store.items[0].memory_id)\n"
    )
    ids = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() for _ in range(2)}
    assert len(ids) == 1 and next(iter(ids)).startswith("chat-")


def test_store_size_is_capped_keeping_newest():
    runtime = _runtime(max_items=5)
    msgs = _chat(20)
    for turn in range(1, 21):
        runtime._remember_selectively(msgs[: turn * 2], "")
    items = runtime.memory_store.items
    assert len(items) == 5
    assert items[-1].content == "answer 19"


def test_items_get_expiry_and_expired_items_are_pruned_and_can_return():
    runtime = _runtime(ttl=60.0)
    msg = [{"role": "user", "content": "remember me"}]
    runtime._remember_selectively(msg, "")
    item = runtime.memory_store.items[0]
    assert item.expires_at is not None and item.expires_at > item.created_at
    expired = MemoryItem(item.memory_id, item.content, item.kind, created_at=item.created_at,
                         expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    runtime.memory_store = MemoryStore((expired,))
    runtime._remember_selectively(msg, "")
    assert len(runtime.memory_store.items) == 1 and runtime.memory_store.items[0].expires_at > datetime.now(timezone.utc)


def test_memory_store_prune_unchanged_returns_same_object():
    store = MemoryStore().write(MemoryItem("a", "x", "k"))
    assert store.prune() is store
    assert store.prune(max_items=0).items == ()
