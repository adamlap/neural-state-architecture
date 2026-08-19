import torch
from nsa.cognitive import NSACognitiveLM


def make_model():
    return NSACognitiveLM(vocab_size=64, d_model=32, state_dim=8, num_layers=1, num_heads=4, max_seq_len=8, dropout=0.0)


def test_cognitive_outputs_are_finite():
    torch.manual_seed(1)
    out = make_model()(torch.randint(0, 64, (2, 8)))
    assert out["logits"].shape == (2, 8, 64)
    assert out["state"].shape == (2, 8, 8)
    assert out["base_state"].shape == (2, 8, 8)
    assert out["prediction_error"].shape == (2, 8, 8)
    assert out["regulation_delta"].shape == (2, 8, 8)
    assert out["capability"].shape == (2, 8, 1)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["state"]).all()


def test_self_state_feedback_is_ablatable_and_causal():
    torch.manual_seed(2)
    model = make_model()
    tokens = torch.randint(0, 64, (2, 8))
    enabled = model(tokens, self_state_feedback=True)
    disabled = model(tokens, self_state_feedback=False)
    assert torch.allclose(enabled["base_state"], disabled["base_state"])
    assert torch.allclose(enabled["base_hidden"], disabled["base_hidden"])
    assert torch.allclose(disabled["error_signal"], torch.zeros_like(disabled["error_signal"]))
    assert torch.allclose(disabled["regulation_delta"], torch.zeros_like(disabled["regulation_delta"]))
    assert not torch.allclose(enabled["state"], disabled["state"])
    assert not torch.allclose(enabled["logits"], disabled["logits"])
    # The hard security coordinate is immutable under self-regulation.
    assert torch.allclose(enabled["state"][..., 0], enabled["base_state"][..., 0])
    assert torch.isfinite(enabled["logits"]).all()


def test_prediction_starts_without_future_information():
    torch.manual_seed(3)
    out = make_model()(torch.randint(0, 64, (1, 8)))
    assert torch.allclose(out["predicted_state"][:, 0], torch.zeros_like(out["predicted_state"][:, 0]))


def test_regulation_is_bounded_and_security_immutable():
    torch.manual_seed(4)
    model = make_model()
    out = model(torch.randint(0, 64, (2, 8)), self_state_feedback=True)
    delta = out["regulation_delta"]
    assert float(delta.abs().max()) <= model.state_regulator.max_delta + 1e-6
    assert torch.allclose(delta[..., 0], torch.zeros_like(delta[..., 0]))
