import torch

from nsa.cognition.latent_field import LatentCognitiveField, LatentFieldConfig


def test_latent_field_accepts_batch_and_flat_inputs():
    field = LatentCognitiveField(LatentFieldConfig(dimension=8, max_norm=5.0))
    out = field.tick(torch.ones(2, 4))
    assert out.vector.shape == (8,)
    assert out.energy <= 5.0

    field.inject_thought_perturbation(torch.ones(3, 2, 2))
    assert field.current.vector.shape == (8,)
