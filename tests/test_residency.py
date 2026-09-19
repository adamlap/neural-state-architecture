"""Unit tests for NSA neural virtual memory."""
import unittest
from nsa.residency import MemoryTier,NeuralRegion,NeuralResidencyManager,ResidencyPolicy,ResidencyState,HeuristicResidencyPredictor
from nsa.residency.cache import CacheEntry,ResidencyCache

class TestResidencyCache(unittest.TestCase):
    def test_lru_capacity(self):
        cache=ResidencyCache(100)
        a=NeuralRegion("a",size_bytes=60); b=NeuralRegion("b",size_bytes=60)
        self.assertEqual(cache.put(CacheEntry(a,MemoryTier.VRAM,60)),[])
        evicted=cache.put(CacheEntry(b,MemoryTier.VRAM,60))
        self.assertEqual([e.region.region_id for e in evicted],["a"])
        self.assertEqual(cache.bytes_used,60)

class TestResidencyPolicy(unittest.TestCase):
    def test_high_future_probability_prefers_vram(self):
        policy=ResidencyPolicy(vram_budget_bytes=1024*1024,ram_budget_bytes=4*1024*1024)
        decision=policy.decide(NeuralRegion("layer.1",size_bytes=64*1024),0.9,0.95)
        self.assertEqual(decision.desired_tier,MemoryTier.VRAM); self.assertTrue(decision.prefetch)
    def test_cold_region_goes_to_nvme(self):
        policy=ResidencyPolicy(vram_budget_bytes=1024*1024,ram_budget_bytes=4*1024*1024)
        self.assertEqual(policy.decide(NeuralRegion("layer.1",size_bytes=64*1024),0,0).desired_tier,MemoryTier.NVME)

class TestResidencyPredictor(unittest.TestCase):
    def test_transition_learning(self):
        p=HeuristicResidencyPredictor()
        p.observe_transition("layer.0","layer.1"); p.observe_transition("layer.0","layer.1"); p.observe_transition("layer.0","layer.2")
        scores=p.predict([NeuralRegion("layer.1"),NeuralRegion("layer.2")],{},current_region="layer.0")
        self.assertGreater(scores["layer.1"],scores["layer.2"])

class TestResidencyManager(unittest.TestCase):
    def test_state_and_snapshot(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=200))
        m.register([NeuralRegion("a",size_bytes=80),NeuralRegion("b",size_bytes=80)])
        m.record_resident("a",MemoryTier.VRAM)
        s=m.snapshot()
        self.assertEqual(s.states["a"],ResidencyState.RESIDENT); self.assertEqual(s.tiers["a"],MemoryTier.VRAM)
        self.assertEqual(s.bytes_by_tier[MemoryTier.VRAM],80)
    def test_plan_uses_state_tags(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=1024,ram_budget_bytes=4096))
        m.register([NeuralRegion("python",size_bytes=100,semantic_tags=("python",)),NeuralRegion("vision",size_bytes=100,semantic_tags=("vision",))])
        p=m.plan({"tags":["python"]})
        self.assertEqual(p[0].region_id,"python")

class TestSelectiveBackend(unittest.TestCase):
    def test_mock_backend_does_not_require_transformers(self):
        from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend
        backend=SelectiveStorageTransformersBackend(mode="mock")
        out=backend.generate("hello")
        self.assertIn("action",out.text)
