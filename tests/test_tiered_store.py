import asyncio

from nsa.residency.tiered_store import TieredRegionStore


class Source:
    def __init__(self):
        self.loads = []

    def load_region(self, region):
        self.loads.append(region)
        return {region: "payload"}

    def size_of(self, region):
        return {"a": 3, "b": 3}[region]


def test_tiered_region_store_uses_one_hot_cache():
    async def scenario():
        source = Source()
        store = TieredRegionStore(source, capacity_bytes=3)
        assert await store.get("a") == {"a": "payload"}
        assert await store.get("a") == {"a": "payload"}
        assert source.loads == ["a"]
        assert store.metrics.hits == 1
        assert store.resident_bytes == 3
        await store.get("b")
        assert source.loads == ["a", "b"]
        assert store.metrics.evictions == 1

    asyncio.run(scenario())


def test_tiered_store_prefetches_from_durable_source():
    async def scenario():
        source = Source()
        store = TieredRegionStore(source, capacity_bytes=6)
        await store.prefetch(["a", "b"])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert sorted(source.loads) == ["a", "b"]
        assert store.resident_bytes == 6

    asyncio.run(scenario())
