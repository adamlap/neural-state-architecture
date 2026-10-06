"""Zero-training System 1 adapter for frozen Hugging Face causal LMs.

The adapter extracts decision evidence directly from a frozen model's output
logits. It never calls generate() and never performs training or gradient
updates. Choices can be multi-token; their normalized sequence log-probability
is compared to produce a typed probability distribution.
"""
from __future__ import annotations

from typing import Any, Mapping

from nsa.cognition.system_one_runtime import DecisionQuestion, SystemOneDecision
from nsa.core.state import CanonicalState


class FrozenCausalLMLogitBackend:
    """Use a frozen open-weight causal LM as a generation-free System 1."""

    name = "frozen-causal-lm-logit"

    def __init__(self, model: Any, tokenizer: Any, *, device: str | None = None, freeze: bool = True) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError("FrozenCausalLMLogitBackend requires the 'ml' extra") from exc
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
        self.model.eval()
        self.device = device or str(next(self.model.parameters()).device)

    def decide(
        self,
        question: DecisionQuestion,
        state: CanonicalState,
        *,
        features: Mapping[str, float] | None = None,
    ) -> SystemOneDecision:
        import math
        import time

        started = time.monotonic()
        state_text = repr(state.summary())
        choices = "\\n".join(
            f"- {choice}: {question.descriptions.get(choice, '')}" for choice in question.choices
        )
        prompt = (
            "You are the fast System 1 controller inside Neural State Architecture.\\n"
            f"STATE={state_text}\\nQUESTION={question.question}\\n"
            f"CHOICES:\\n{choices}\\nDECISION:"
        )

        scores = self._batch_choice_log_probabilities(prompt, question.choices)
        if features:
            # Optional deterministic prior can steer routing without overriding
            # model evidence. It is deliberately tiny compared with logit evidence.
            scores = {choice: score + math.log(max(1e-6, float(features.get(choice, 1.0)))) * 0.05
                      for choice, score in scores.items()}
        max_score = max(scores.values())
        weights = {choice: math.exp(score - max_score) for choice, score in scores.items()}
        total = sum(weights.values())
        probabilities = {choice: weight / total for choice, weight in weights.items()}
        selected = max(probabilities, key=probabilities.get)
        uncertainty = self._entropy(tuple(probabilities.values()))
        confidence = probabilities[selected]
        risk = float(question.risk_weights.get(selected, 0.0))
        passed = confidence >= question.min_confidence
        return SystemOneDecision(
            question=question.name,
            selected_choice=selected,
            probabilities=probabilities,
            confidence=confidence,
            uncertainty=uncertainty,
            risk=risk,
            passed_gate=passed,
            backend=self.name,
            latency_ms=(time.monotonic() - started) * 1000.0,
            reason="" if passed else f"confidence {confidence:.2f} < gate {question.min_confidence:.2f}",
        )

    def _batch_choice_log_probabilities(self, prompt: str, choices: tuple[str, ...] | list[str]) -> dict[str, float]:
        torch = self.torch
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
        prompt_len = prompt_ids.shape[1]

        choice_token_list = [
            self.tokenizer(c, return_tensors="pt", add_special_tokens=False)["input_ids"]
            for c in choices
        ]

        seqs = [torch.cat([prompt_ids, c_ids], dim=1) for c_ids in choice_token_list]
        max_len = max(s.shape[1] for s in seqs)

        batch_size = len(choices)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
        batch_input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, s in enumerate(seqs):
            length = s.shape[1]
            batch_input_ids[i, :length] = s[0]
            attention_mask[i, :length] = 1

        with torch.inference_mode():
            output = self.model(input_ids=batch_input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, "logits") else output[0]
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target = batch_input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

        scores = {}
        for i, choice in enumerate(choices):
            c_len = choice_token_list[i].shape[1]
            start = prompt_len - 1
            end = start + c_len
            choice_scores = token_log_probs[i, start:end]
            if choice_scores.numel() == 0:
                scores[choice] = -10.0
            else:
                scores[choice] = float(choice_scores.mean().item())
        return scores

    def _choice_log_probability(self, prompt: str, choice: str) -> float:
        torch = self.torch
        # Tokenize separately so only appended choice tokens are scored.
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        choice_ids = self.tokenizer(choice, return_tensors="pt", add_special_tokens=False)
        input_ids = torch.cat([prompt_ids["input_ids"], choice_ids["input_ids"]], dim=1).to(self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits if hasattr(output, "logits") else output[0]
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

        start = prompt_ids["input_ids"].shape[1] - 1
        end = token_log_probs.shape[1]
        choice_scores = token_log_probs[:, start:end]
        if choice_scores.numel() == 0:
            raise ValueError(f"choice {choice!r} produced no tokens")
        # Mean log probability avoids systematic preference for shorter labels.
        return float(choice_scores.mean().item())

    @staticmethod
    def _entropy(probs: tuple[float, ...]) -> float:
        import math
        if len(probs) <= 1:
            return 0.0
        h = -sum(p * math.log(p) for p in probs if p > 1e-12)
        return max(0.0, min(1.0, h / math.log(len(probs))))


__all__ = ["FrozenCausalLMLogitBackend"]
