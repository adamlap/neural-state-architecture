import pytest

torch = pytest.importorskip("torch")

from nsa.cognition.system_one_hf import FrozenCausalLMLogitBackend
from nsa.cognition.system_one_runtime import DecisionQuestion
from nsa.core.state import CanonicalState


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        del return_tensors, add_special_tokens
        if text == "yes":
            ids = [1, 1, 1]
        elif text == "no":
            ids = [2, 2]
        else:
            ids = [3] * max(1, len(text) // 8)
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, 8, device=input_ids.device)
        targets = input_ids[:, 1:]
        for i in range(seq - 1):
            target = targets[:, i]
            logits[:, i, :] = torch.where(
                target.unsqueeze(-1) == 1,
                torch.tensor(6.0, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )
            logits[:, i, :] = torch.where(
                target.unsqueeze(-1) == 2,
                torch.tensor(1.0, device=input_ids.device),
                logits[:, i, :],
            )
        return type("Output", (), {"logits": logits})()


def test_frozen_open_weight_backend_uses_logits_without_generation():
    backend = FrozenCausalLMLogitBackend(FakeModel(), FakeTokenizer())
    result = backend.decide(
        DecisionQuestion("binary", "choose", ("yes", "no"), min_confidence=0.0),
        CanonicalState(),
    )
    assert result.selected_choice == "yes"
    assert result.backend == "frozen-causal-lm-logit"
    assert result.probabilities["yes"] > result.probabilities["no"]
    assert result.confidence > 0.5
