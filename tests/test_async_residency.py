import asyncio

from nsa.residency.async_manager import AsyncResidencyManager


def test_async_manager_deduplicates_loads_and_tracks_hits():
    async def run():
        calls = []

        async def loader(name):
            calls.append(name)
            await asyncio.sleep(0)
            return f"value:{name}"

        manager = AsyncResidencyManager(loader, lambda _: 10, capacity_bytes=20)

        first, second = await asyncio.gather(
            manager.get("a"),
            manager.get("a"),
        )
        third = await manager.get("a")

        assert first == second == third == "value:a"
        assert calls == ["a"]
        assert manager.metrics.requests == 3
        assert manager.metrics.misses == 2
        assert manager.metrics.hits == 1

    asyncio.run(run())


def test_async_manager_evicts_to_admit_new_region():
    async def run():
        async def loader(name):
            return name

        manager = AsyncResidencyManager(loader, lambda _: 10, capacity_bytes=20)
        await manager.get("a", next_use=None)
        await manager.get("b", next_use=10)
        await manager.get("c", next_use=20)

        assert manager.resident_bytes <= 20
        assert manager.metrics.evictions == 1

    asyncio.run(run())


def test_async_manager_prefetches():
    async def run():
        async def loader(name):
            await asyncio.sleep(0)
            return name

        manager = AsyncResidencyManager(loader, lambda _: 5, capacity_bytes=20)
        await manager.prefetch(["a", "b"])
        await asyncio.sleep(0.01)

        assert manager.metrics.prefetches == 2
        assert manager.resident_bytes == 10

    asyncio.run(run())
