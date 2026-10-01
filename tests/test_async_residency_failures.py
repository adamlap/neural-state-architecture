import asyncio

import pytest

from nsa.residency.async_manager import AsyncResidencyManager


def test_loader_failure_is_propagated_and_counted():
    async def scenario():
        async def loader(_):
            raise OSError("disk failure")

        manager = AsyncResidencyManager(loader, lambda _: 1, capacity_bytes=4)
        with pytest.raises(OSError, match="disk failure"):
            await manager.get("bad")
        assert manager.metrics.load_errors == 1
        assert manager.resident_bytes == 0

    asyncio.run(scenario())


def test_oversized_region_is_rejected_before_admission():
    async def scenario():
        manager = AsyncResidencyManager(lambda _: b"x", lambda _: 9, capacity_bytes=8)
        with pytest.raises(MemoryError, match="requires 9 bytes"):
            await manager.get("too-large")
        assert manager.resident_bytes == 0

    asyncio.run(scenario())


def test_prefetch_failure_does_not_create_unhandled_background_error():
    async def scenario():
        async def loader(_):
            raise RuntimeError("prefetch failed")

        manager = AsyncResidencyManager(loader, lambda _: 1, capacity_bytes=4)
        await manager.prefetch(["bad"])
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert manager.metrics.failed_prefetches == 1
        assert manager.resident_bytes == 0

    asyncio.run(scenario())
