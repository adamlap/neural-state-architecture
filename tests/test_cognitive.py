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
    assert not torch.allclose(enabled["state"], disabled["state"])
    assert not torch.allclose(enabled["logits"], disabled["logits"])
    # The hard security coordinate is immutable under self-regulation.
    assert torch.allclose(enabled["state"][..., 0], enabled["base_state"][..., 0])
    assert torch.isfinite(enabled["logits"]).all()


def test_prediction_starts_without_future_information():
    torch.manual_seed(3)
    out = make_model()(torch.randint(0, 64, (1, 8)))
    assert torch.allclose(out["predicted_state"][:, 0], torch.zeros_like(out["predicted_state"][:, 0]))
