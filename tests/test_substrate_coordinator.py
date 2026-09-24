from nsa.core.state import CanonicalState
from nsa.core.substrate_coordinator import NeuralSubstrateCoordinator
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.policy import ResidencyPolicy
from nsa.residency.types import MemoryTier, NeuralRegion


def _coordinator():
    manager = NeuralResidencyManager(
        ResidencyPolicy(vram_budget_bytes=100, ram_budget_bytes=200)
    )
    manager.register([
        NeuralRegion("a", size_bytes=60),
        NeuralRegion("b", size_bytes=60),
        NeuralRegion("c", size_bytes=60),
    ])
    return NeuralSubstrateCoordinator(CanonicalState(), manager)


def test_router_compute_residency_loop_and_resource_constraints():
    coordinator = _coordinator()
    coordinator.observe_routing(["b"], [0.95])
    transition = coordinator.plan({"tags": []})
    assert transition.candidates
    allocations = {a.region_id: a for a in transition.allocations}
    assert allocations["b"].probability >= 0.95
    assert allocations["b"].tier == MemoryTier.VRAM
    assert sum(a.tier == MemoryTier.VRAM for a in transition.allocations) <= 1


def test_execution_updates_state_and_predictor_without_authority_change():
    coordinator = _coordinator()
    before = coordinator.state.hard
    coordinator.commit_execution(["a", "b"])
    assert coordinator.state.step == 1
    assert coordinator.state.hard == before
    assert coordinator.residency.current_region == "b"


def test_apply_allocations_only_changes_residency():
    coordinator = _coordinator()
    before = coordinator.state.hard
    transition = coordinator.plan({"tags": []})
    coordinator.apply_resource_allocations(transition.allocations)
    assert coordinator.state.hard == before


def test_resource_pressure_is_bounded():
    coordinator = _coordinator()
    coordinator.residency.record_resident("a", MemoryTier.VRAM)
    coordinator.residency.record_resident("b", MemoryTier.RAM)
    assert 0.0 <= coordinator.resource_pressure() <= 1.0
