"""Fast structural smoke test for the residency subsystem.

Run:
  python scripts/residency_smoke_test.py
"""
from __future__ import annotations

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


def main() -> None:
    policy = ResidencyPolicy(vram_budget_bytes=100, ram_budget_bytes=200)
    predictor = OnlineResidencyPredictor()
    manager = NeuralResidencyManager(policy=policy, predictor=predictor)
    manager.register(
        [
            NeuralRegion("layer.0", size_bytes=50, layer_index=0),
            NeuralRegion("layer.1", size_bytes=50, layer_index=1),
            NeuralRegion("layer.2", size_bytes=50, layer_index=2),
        ]
    )

    for _ in range(5):
        predictor.observe("layer.0", "layer.1")
        predictor.observe("layer.1", "layer.2")
    manager.current_region = "layer.0"

    calls: list[tuple[str, MemoryTier]] = []

    def load(region_id: str, tier: MemoryTier) -> None:
        calls.append((region_id, tier))

    controller = ActiveResidencyController(manager, load_fn=load, lookahead=2)
    tasks = controller.tick({"tags": ["transformer"]})

    assert tasks, "expected a learned transition to produce a residency task"
    assert any(region_id == "layer.1" for region_id, _ in calls)

    trace = ResidencyTrace(max_events=8)
    trace.record(
        ResidencyEvent(
            timestamp=1.0,
            region_id="layer.1",
            action="prefetch",
            source=MemoryTier.NVME,
            destination=MemoryTier.RAM,
            bytes_moved=50,
            latency_ms=1.0,
            reason="smoke",
        )
    )
    trace.record(
        ResidencyEvent(
            timestamp=2.0,
            region_id="layer.1",
            action="prefetch-complete",
            source=MemoryTier.NVME,
            destination=MemoryTier.RAM,
            bytes_moved=50,
            latency_ms=1.0,
            reason="smoke",
        )
    )
    trace.record(
        ResidencyEvent(
            timestamp=3.0,
            region_id="layer.1",
            action="execute",
            source=MemoryTier.RAM,
            destination=MemoryTier.RAM,
            bytes_moved=0,
            latency_ms=1.0,
            reason="smoke",
        )
    )

    metrics = trace.metrics()
    assert metrics["prefetches"] == 1
    assert metrics["prefetch_hits"] == 1
    assert metrics["prefetch_hit_rate"] == 1.0

    heuristic = HeuristicResidencyPredictor()
    heuristic.observe_transition(None, "layer.0")
    heuristic.observe_transition("layer.0", "layer.1")
    scores = heuristic.predict(
        manager.regions.values(),
        {"tags": []},
        current_region="layer.0",
    )
    assert scores["layer.1"] > scores["layer.2"]

    controller.shutdown()
    print("residency smoke test: PASS")


if __name__ == "__main__":
    main()
