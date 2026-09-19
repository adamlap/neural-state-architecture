"""Tests for active and learned residency."""
import unittest
from nsa.residency import MemoryTier, NeuralRegion, NeuralResidencyManager, ResidencyPolicy, ActiveResidencyController, OnlineResidencyPredictor

class TestActiveResidency(unittest.TestCase):
    def test_prefetch_executes_selected_loads(self):
        manager=NeuralResidencyManager(ResidencyPolicy(vram_budget_bytes=1024*1024,ram_budget_bytes=1024*1024))
        manager.register([NeuralRegion("a",size_bytes=10),NeuralRegion("b",size_bytes=10)])
        calls=[]
        controller=ActiveResidencyController(manager,lambda rid,tier:calls.append((rid,tier)),lookahead=2)
        manager.predictor.observe_transition(None,"a") if hasattr(manager.predictor,"observe_transition") else None
        tasks=controller.tick({"tags":[]})
        self.assertEqual(tasks,[])
        manager.predictor.observe_transition("a","b")
        manager.current_region="a"
        tasks=controller.tick({"tags":[]})
        self.assertTrue(any(x[0]=="b" for x in calls))

class TestLearnedPredictor(unittest.TestCase):
    def test_transition_signal(self):
        p=OnlineResidencyPredictor()
        p.observe("a","b"); p.observe("a","b"); p.observe("a","c")
        regions=[NeuralRegion("b"),NeuralRegion("c")]
        scores=p.predict(regions,{},current_region="a")
        self.assertGreater(scores["b"],scores["c"])

if __name__=="__main__": unittest.main()
