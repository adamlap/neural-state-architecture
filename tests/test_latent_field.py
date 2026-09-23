"""Tests for continuous latent cognitive field and sub-inference thought vectors."""
import torch
from nsa.cognition.latent_field import LatentCognitiveField, LatentFieldConfig, LatentThoughtVector
from nsa.core.state import SoftState


def test_latent_field_initialization():
    config = LatentFieldConfig(dimension=64, max_norm=5.0)
    field = LatentCognitiveField(config)

    assert field.current.dimension == 64
    assert field.current.step == 0
    assert field.current.energy == 0.0

    soft = field.project_soft_state()
    assert isinstance(soft, SoftState)
    assert 0.0 <= soft.uncertainty <= 1.0
    assert 0.0 <= soft.risk <= 1.0
    assert 0.0 <= soft.confidence <= 1.0


def test_latent_field_continuous_ticks():
    field = LatentCognitiveField(LatentFieldConfig(dimension=32, integration_rate=0.25))

    sensory_inp = torch.randn(32)
    for _ in range(10):
        vec = field.tick(sensory_input=sensory_inp)

    assert vec.step == 10
    assert vec.energy > 0.0
    assert len(field._history) == 10


def test_lipschitz_energy_bounding():
    config = LatentFieldConfig(dimension=16, max_norm=4.0)
    field = LatentCognitiveField(config)

    # Inject massive perturbation that exceeds max_norm
    massive_impulse = torch.ones(16) * 100.0
    field.inject_thought_perturbation(massive_impulse)

    assert field.current.energy <= 4.0001


def test_deliberation_trigger_threshold():
    config = LatentFieldConfig(dimension=32, deliberation_threshold=0.60)
    field = LatentCognitiveField(config)

    # Initial state should not require deliberation
    initial_delib = field.requires_deliberation()

    # Align current vector strongly with uncertainty projection to trigger deliberation
    field._current_vec = field._proj_uncertainty * 10.0
    assert field.requires_deliberation() is True