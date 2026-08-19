import torch

from experiments.self_state.transformer_probe import run


def test_native_transformer_state_probe():
    result = run(batch=2, seq_len=8, vocab_size=64, state_dim=8)
    assert result["logits_shape"] == [2, 8, 64]
    assert result["state_shape"] == [2, 8, 8]
    assert result["hidden_shape"] == [2, 8, 64]
    assert result["finite"] == 1.0
    assert torch.isfinite(torch.tensor(result["logit_delta_from_zero_init"]))
