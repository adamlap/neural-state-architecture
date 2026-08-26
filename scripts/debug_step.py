from experiments.nsa61.environments.hardened_blind_world import HardenedBlindWorldEnvironment
from experiments.nsa62.agents.frozen_llm_agents import FrozenLLMBenchmarkHarness
from nsa.runtime.inference.transformers import PyTorchTransformersBackend
from nsa.runtime.inference.base import BackendMode

world = HardenedBlindWorldEnvironment(difficulty_tier='D3', seed=42)
backend = PyTorchTransformersBackend(mode=BackendMode.MOCK)
harness = FrozenLLMBenchmarkHarness(backend=backend)
print('Active world:', world.active_world.world_id)
steps = harness.run_arm_d_nsa_closed_loop(world)
for s in steps:
    print('Step:', s.action_taken, 'rec:', s.is_recovered, 'viol:', s.is_violation, 'ig:', s.information_gain)
print('Recovered:', world.state_db.get('recovered'))
