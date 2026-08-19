"""Smoke/invariant tests for the first NSA cognitive experiment."""

import torch

from experiments.self_state.model import BaselineEvidenceModel, ExplicitSelfStateModel, parameter_count
from experiments.self_state.task import make_batch


def test_models_accept_sequential_evidence():
    x, y = make_batch(8, steps=5, generator=torch.Generator().manual_seed(1))
    baseline = BaselineEvidenceModel(hidden=32)
    explicit = ExplicitSelfStateModel(hidden=28)
    baseline_out = baseline(x)
    explicit_out = explicit(x)
    assert baseline_out["logits"].shape == y.shape
    assert explicit_out["logits"].shape == y.shape
    assert explicit_out["self_state"].shape == (8, 7)
    assert explicit_out["state_trace"].shape == (8, 5, 7)


def test_explicit_state_path_is_causally_ablatable():
    x, _ = make_batch(8, steps=5, generator=torch.Generator().manual_seed(2))
    model = ExplicitSelfStateModel(hidden=28)
    normal = model(x, state_scale=1.0)["logits"]
    ablated = model(x, state_scale=0.0)["logits"]
    assert not torch.allclose(normal, ablated)


def test_parameter_budgets_are_close():
    baseline = parameter_count(BaselineEvidenceModel(hidden=32))
    explicit = parameter_count(ExplicitSelfStateModel(hidden=28))
    ratio = explicit / baseline
    assert 0.9 <= ratio <= 1.1
