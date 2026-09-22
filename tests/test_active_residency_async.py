import threading
import unittest

from nsa.residency import (
    ActiveResidencyController, MemoryTier, NeuralRegion, NeuralResidencyManager,
    ResidencyPolicy,
)


def _manager(**policy):
    manager = NeuralResidencyManager(
        ResidencyPolicy(vram_budget_bytes=1024 * 1024, ram_budget_bytes=1024 * 1024, **policy)
    )
    manager.register([NeuralRegion("a", size_bytes=10), NeuralRegion("b", size_bytes=10)])
    manager.predictor.observe_transition("a", "b")
    manager.current_region = "a"
    return manager


def _actions(manager, region="b"):
    return [e.action for e in manager.events if e.region_id == region]


class TestAsyncPrefetch(unittest.TestCase):
    def test_prefetch_does_not_claim_residency(self):
        manager = _manager()
        calls = []
        controller = ActiveResidencyController(
            manager,
            load_fn=lambda rid, tier: calls.append(("load", rid, tier)),
            lookahead=2,
            prefetch_fn=lambda rid: calls.append(("prefetch", rid)),
        )
        tasks = controller.prefetch_async({"tags": []})
        controller.wait()
        self.assertTrue(any(task.region_id == "b" for task in tasks))
        self.assertIn(("prefetch", "b"), calls)
        self.assertNotIn("load", [c[0] for c in calls])
        self.assertEqual(manager.states["b"].value, "cold")
        controller.shutdown()

    def test_instant_prefetch_never_deadlocks(self):
        """Regression: add_done_callback ran inline under a non-reentrant lock."""
        manager = _manager()
        controller = ActiveResidencyController(
            manager, load_fn=lambda rid, tier: None, lookahead=2, prefetch_fn=lambda rid: 1
        )
        done = threading.Event()

        def hammer():
            for _ in range(500):
                controller.prefetch_async({"tags": []})
            controller.wait()
            done.set()

        thread = threading.Thread(target=hammer, daemon=True)
        thread.start()
        self.assertTrue(done.wait(20), "prefetch_async deadlocked")
        controller.shutdown()

    def test_prefetch_records_completion_with_actual_bytes_in_host_ram(self):
        manager = _manager()
        controller = ActiveResidencyController(manager, lambda r, t: None, prefetch_fn=lambda rid: 4096)
        controller.prefetch_async({"tags": []}); controller.wait(); controller.shutdown()
        complete = [e for e in manager.events if e.action == "prefetch-complete"]
        self.assertEqual(len(complete), 1)
        self.assertEqual(complete[0].bytes_moved, 4096)
        self.assertEqual(complete[0].destination, MemoryTier.RAM)

    def test_prefetch_that_warms_nothing_is_skipped_not_completed(self):
        manager = _manager()
        controller = ActiveResidencyController(manager, lambda r, t: None, prefetch_fn=lambda rid: 0)
        controller.prefetch_async({"tags": []}); controller.wait(); controller.shutdown()
        self.assertIn("prefetch-skipped", _actions(manager))
        self.assertNotIn("prefetch-complete", _actions(manager))

    def test_prefetch_failure_is_recorded_and_does_not_raise(self):
        manager = _manager()

        def boom(rid):
            raise OSError("disk gone")

        controller = ActiveResidencyController(manager, lambda r, t: None, prefetch_fn=boom)
        controller.prefetch_async({"tags": []})
        controller.wait()  # must not raise
        controller.shutdown()
        errors = [e for e in manager.events if e.action == "prefetch-error"]
        self.assertEqual(len(errors), 1)
        self.assertIn("disk gone", errors[0].reason)

    def test_ineligible_regions_are_not_scheduled(self):
        manager = _manager()
        calls = []
        controller = ActiveResidencyController(
            manager, lambda r, t: None, prefetch_fn=lambda rid: calls.append(rid) or 1,
            prefetch_eligible=lambda rid: False,
        )
        controller.prefetch_async({"tags": []}); controller.wait(); controller.shutdown()
        self.assertEqual(calls, [])

    def test_prefetch_after_shutdown_is_a_noop(self):
        manager = _manager()
        calls = []
        controller = ActiveResidencyController(manager, lambda r, t: None, prefetch_fn=lambda rid: calls.append(rid) or 1)
        controller.shutdown()
        controller.prefetch_async({"tags": []})  # must not raise "cannot schedule new futures"
        self.assertEqual(calls, [])

    def test_tick_load_does_not_move_execution_position(self):
        manager = _manager()
        controller = ActiveResidencyController(manager, load_fn=lambda rid, tier: None)
        controller.tick({"tags": []})
        self.assertEqual(manager.states["b"].value, "resident")
        self.assertEqual(manager.current_region, "a")
        controller.shutdown()

    def test_retain_or_evict_keeps_executing_region(self):
        manager = _manager()
        manager.record_resident("a", MemoryTier.RAM)
        controller = ActiveResidencyController(manager, lambda r, t: None)
        retained = controller.retain_or_evict(manager.plan({"tags": []}))
        self.assertIn("a", retained)
        self.assertEqual(manager.states["a"].value, "resident")
        controller.shutdown()


if __name__ == "__main__":
    unittest.main()
