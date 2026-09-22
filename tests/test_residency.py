"""Unit tests for NSA neural virtual memory."""
import unittest
from nsa.residency import MemoryTier,NeuralRegion,NeuralResidencyManager,OnlineResidencyPredictor,ResidencyPolicy,ResidencyState,HeuristicResidencyPredictor
from nsa.residency.cache import CacheEntry,ResidencyCache
from nsa.residency.predictor import state_tags

class TestResidencyCache(unittest.TestCase):
    def test_lru_capacity(self):
        cache=ResidencyCache(100)
        a=NeuralRegion("a",size_bytes=60); b=NeuralRegion("b",size_bytes=60)
        self.assertEqual(cache.put(CacheEntry(a,MemoryTier.VRAM,60)),[])
        evicted=cache.put(CacheEntry(b,MemoryTier.VRAM,60))
        self.assertEqual([e.region.region_id for e in evicted],["a"])
        self.assertEqual(cache.bytes_used,60)
    def test_oversize_entry_is_rejected_without_flushing_the_cache(self):
        cache=ResidencyCache(100)
        cache.put(CacheEntry(NeuralRegion("small",size_bytes=50),MemoryTier.VRAM,50))
        rejected=cache.put(CacheEntry(NeuralRegion("huge",size_bytes=500),MemoryTier.VRAM,500))
        self.assertEqual([e.region.region_id for e in rejected],["huge"])
        self.assertTrue(cache.contains("small")); self.assertFalse(cache.contains("huge"))
        self.assertEqual(cache.bytes_used,50)
    def test_reput_replaces_and_never_evicts_itself(self):
        cache=ResidencyCache(100)
        region=NeuralRegion("a",size_bytes=80)
        cache.put(CacheEntry(region,MemoryTier.VRAM,80))
        self.assertEqual(cache.put(CacheEntry(region,MemoryTier.VRAM,80)),[])
        self.assertEqual(cache.bytes_used,80)

class TestResidencyPolicy(unittest.TestCase):
    def test_high_future_probability_prefers_vram(self):
        policy=ResidencyPolicy(vram_budget_bytes=1024*1024,ram_budget_bytes=4*1024*1024)
        decision=policy.decide(NeuralRegion("layer.1",size_bytes=64*1024),0.9,0.95)
        self.assertEqual(decision.desired_tier,MemoryTier.VRAM); self.assertTrue(decision.prefetch)
    def test_cold_region_goes_to_nvme(self):
        policy=ResidencyPolicy(vram_budget_bytes=1024*1024,ram_budget_bytes=4*1024*1024)
        self.assertEqual(policy.decide(NeuralRegion("layer.1",size_bytes=64*1024),0,0).desired_tier,MemoryTier.NVME)

    def test_unknown_relevance_is_not_scored_as_irrelevant(self):
        policy=ResidencyPolicy(vram_budget_bytes=1024*1024*1024,ram_budget_bytes=2*1024**3)
        region=NeuralRegion("layer.1",size_bytes=64*1024*1024)  # 6% of the VRAM budget
        unknown=policy.decide(region,None,1.0)
        known_irrelevant=policy.decide(region,0.0,1.0)
        self.assertTrue(unknown.prefetch)          # certain transition, no tag evidence
        self.assertFalse(known_irrelevant.prefetch)  # tags say: not relevant
        self.assertIn("unknown",unknown.reason)
    def test_size_penalty_can_still_veto_a_huge_region(self):
        policy=ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=200)
        self.assertFalse(policy.decide(NeuralRegion("big",size_bytes=100),None,0.6).prefetch)

class TestResidencyPredictor(unittest.TestCase):
    def test_transition_learning(self):
        p=HeuristicResidencyPredictor()
        p.observe_transition("layer.0","layer.1"); p.observe_transition("layer.0","layer.1"); p.observe_transition("layer.0","layer.2")
        scores=p.predict([NeuralRegion("layer.1"),NeuralRegion("layer.2")],{},current_region="layer.0")
        self.assertGreater(scores["layer.1"],scores["layer.2"])
    def test_deterministic_transition_is_full_probability(self):
        for predictor,observe in ((HeuristicResidencyPredictor(),"observe_transition"),(OnlineResidencyPredictor(),"observe")):
            getattr(predictor,observe)("layer.0","layer.1")
            scores=predictor.predict([NeuralRegion("layer.0"),NeuralRegion("layer.1")],{"tags":[]},current_region="layer.0")
            self.assertAlmostEqual(scores["layer.1"],1.0)
            self.assertEqual(scores["layer.0"],0.0)
    def test_unrelated_tags_do_not_dilute_transition_evidence(self):
        p=OnlineResidencyPredictor(); p.observe("a","b")
        scores=p.predict([NeuralRegion("a"),NeuralRegion("b")],{"tags":["never-seen"]},current_region="a")
        self.assertAlmostEqual(scores["b"],1.0)
    def test_tags_and_transitions_blend(self):
        p=HeuristicResidencyPredictor(); p.observe_transition("a","b"); p.observe_state_tags(["code"],"c")
        scores=p.predict([NeuralRegion("b"),NeuralRegion("c")],{"tags":["code"]},current_region="a")
        self.assertAlmostEqual(scores["b"],0.65); self.assertAlmostEqual(scores["c"],0.35)
    def test_malformed_tags_are_tolerated(self):
        self.assertEqual(state_tags({"tags":None}),()); self.assertEqual(state_tags({"tags":"solo"}),("solo",))
        self.assertEqual(state_tags({"tags":5}),()); self.assertEqual(state_tags(None),())
        OnlineResidencyPredictor().observe("a","b",{"tags":None})

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
    def test_tier_move_does_not_double_count(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100))
        m.register([NeuralRegion("r",size_bytes=60)])
        m.record_resident("r",MemoryTier.RAM); m.record_resident("r",MemoryTier.VRAM)
        self.assertEqual(m.ram_cache.bytes_used,0); self.assertEqual(m.vram_cache.bytes_used,60)
        self.assertEqual(m.snapshot().bytes_by_tier[MemoryTier.RAM],0)
    def test_oversize_region_is_rejected_and_prior_state_kept(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100))
        m.register([NeuralRegion("big",size_bytes=500),NeuralRegion("ok",size_bytes=50)])
        self.assertTrue(m.record_resident("ok",MemoryTier.VRAM))
        self.assertFalse(m.record_resident("big",MemoryTier.VRAM))
        self.assertEqual(m.states["ok"],ResidencyState.RESIDENT)
        self.assertEqual(m.states["big"],ResidencyState.COLD)
        self.assertEqual(m.events[-1].action,"reject")
    def test_capacity_eviction_marks_victim_cold(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100))
        m.register([NeuralRegion("a",size_bytes=60),NeuralRegion("b",size_bytes=60)])
        m.record_resident("a",MemoryTier.VRAM); m.record_resident("b",MemoryTier.VRAM)
        self.assertEqual(m.states["a"],ResidencyState.COLD); self.assertEqual(m.states["b"],ResidencyState.RESIDENT)
        self.assertEqual(m.snapshot().bytes_by_tier[MemoryTier.VRAM],60)
    def test_residency_transfer_does_not_change_execution_position(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100))
        m.register([NeuralRegion("a",size_bytes=10),NeuralRegion("b",size_bytes=10)])
        m.current_region="a"; m.record_resident("b",MemoryTier.VRAM)
        self.assertEqual(m.current_region,"a")
    def test_event_log_is_bounded(self):
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100),max_events=5)
        m.register([NeuralRegion("a",size_bytes=10)])
        for _ in range(50): m.record_resident("a",MemoryTier.RAM)
        self.assertEqual(len(m.events),5)
    def test_snapshot_is_json_serialisable(self):
        import json
        m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=100,ram_budget_bytes=100))
        m.register([NeuralRegion("a",size_bytes=10)]); m.record_resident("a",MemoryTier.RAM)
        json.dumps(m.snapshot().to_dict())

class TestPrefetchReachesThreshold(unittest.TestCase):
    """Regression: a deterministic transition on a realistically sized layer must be prefetched."""
    def test_qwen_3b_sized_layer_is_prefetched_by_both_predictors(self):
        layer_bytes=147*1024*1024
        for predictor,observe in ((HeuristicResidencyPredictor(),"observe_transition"),(OnlineResidencyPredictor(),"observe")):
            m=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=4*1024**3,ram_budget_bytes=8*1024**3),predictor=predictor)
            m.register([NeuralRegion(f"layer.{i}",size_bytes=layer_bytes,dependencies=(f"layer.{i-1}",) if i else ()) for i in range(3)])
            getattr(predictor,observe)("layer.0","layer.1")
            m.current_region="layer.0"
            decisions={d.region_id:d for d in m.plan({"tags":[]})}
            self.assertTrue(decisions["layer.1"].prefetch,decisions["layer.1"])
            self.assertFalse(decisions["layer.2"].prefetch)

class TestSelectiveBackend(unittest.TestCase):
    def test_mock_backend_does_not_require_transformers(self):
        from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend
        backend=SelectiveStorageTransformersBackend(mode="mock")
        out=backend.generate("hello")
        self.assertIn("action",out.text)
