import unittest

from nsa.residency import MemoryTier, NeuralRegion, NeuralResidencyManager, ResidencyPolicy, ActiveResidencyController


class TestAsyncPrefetch(unittest.TestCase):
    def test_prefetch_does_not_claim_residency(self):
        manager = NeuralResidencyManager(
            ResidencyPolicy(vram_budget_bytes=1024 * 1024, ram_budget_bytes=1024 * 1024)
        )
        manager.register([NeuralRegion("a", size_bytes=10), NeuralRegion("b", size_bytes=10)])
        manager.predictor.observe_transition("a", "b")
        manager.current_region = "a"
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
        self.assertEqual(manager.states["b"].value, "cold")
        controller.shutdown()


if __name__ == "__main__":
    unittest.main()
