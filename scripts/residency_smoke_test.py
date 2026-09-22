"""Fast structural smoke test for the residency subsystem (no model required).

Run:
  python scripts/residency_smoke_test.py
"""
from __future__ import annotations

from time import monotonic

from nsa.residency import (
    ActiveResidencyController,
    HeuristicResidencyPredictor,
    MemoryTier,
    NeuralRegion,
    NeuralResidencyManager,
    OnlineResidencyPredictor,
    ResidencyEvent,
    ResidencyPolicy,
    ResidencyTrace,
)

MIB = 1024 * 1024


def main() -> None:
    policy = ResidencyPolicy(vram_budget_bytes=1024 * MIB, ram_budget_bytes=2048 * MIB)
    predictor = OnlineResidencyPredictor()
    manager = NeuralResidencyManager(policy=policy, predictor=predictor, trace=ResidencyTrace(max_events=64))
    manager.register(
        [NeuralRegion(f"layer.{i}", size_bytes=64 * MIB, layer_index=i) for i in range(3)]
    )

    # Execute the layers the way the decoder hook does, feeding the predictor.
    for _ in range(5):
        previous = None
        for i in range(3):
            predictor.observe(previous, f"layer.{i}")
            previous = f"layer.{i}"
    manager.current_region = "layer.0"

    # tick(): a learned transition produces a real placement task.
    calls: list[tuple[str, MemoryTier]] = []
    controller = ActiveResidencyController(
        manager, load_fn=lambda region_id, tier: calls.append((region_id, tier)), lookahead=2
    )
    tasks = controller.tick({"tags": []})
    assert tasks, "expected a learned transition to produce a residency task"
    assert any(region_id == "layer.1" for region_id, _ in calls)
    assert manager.current_region == "layer.0", "a transfer must not move the execution position"

    # prefetch_async(): a warmed region is followed by an execution -> measured hit.
    warmed: list[str] = []
    async_controller = ActiveResidencyController(
        manager, load_fn=lambda *_: None, lookahead=2,
        prefetch_fn=lambda region_id: warmed.append(region_id) or 64 * MIB,
    )
    async_controller.prefetch_async({"tags": []})
    async_controller.wait()
    assert "layer.1" in warmed
    manager.record_event(
        ResidencyEvent(monotonic() + 0.01, "layer.1", "execute", MemoryTier.RAM, MemoryTier.RAM, 0, 1.0, "smoke")
    )
    metrics = manager.trace.metrics()
    assert metrics["prefetch_completed"] >= 1, metrics
    assert metrics["prefetch_hits"] >= 1 and 0.0 < metrics["prefetch_hit_rate"] <= 1.0, metrics
    assert metrics["bytes_prefetched"] >= 64 * MIB, metrics

    # An execution that nothing prefetched is never counted as a hit.
    solo = ResidencyTrace()
    solo.record(ResidencyEvent(1.0, "layer.0", "execute", MemoryTier.RAM, MemoryTier.RAM, 0, 1.0, "smoke"))
    assert solo.metrics()["prefetch_hits"] == 0

    heuristic = HeuristicResidencyPredictor()
    heuristic.observe_transition(None, "layer.0")
    heuristic.observe_transition("layer.0", "layer.1")
    scores = heuristic.predict(list(manager.regions.values()), {"tags": []}, current_region="layer.0")
    assert scores["layer.1"] > scores["layer.2"]

    controller.shutdown()
    async_controller.shutdown()
    print("residency smoke test: PASS")


if __name__ == "__main__":
    main()
