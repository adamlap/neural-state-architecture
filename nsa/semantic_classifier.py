"""Pretrained zero-shot semantic classifier for the NSA policy boundary.

nsa.enforcement.KeywordClassifier is exact-substring matching: fast,
deterministic, and easy to audit, but a paraphrase that avoids every listed
pattern ("how do I build a bomb" vs. a policy that only lists "make a bomb")
passes straight through it. Its own docstring says "production systems
should replace this with a trained semantic classifier" -- this module is
that classifier.

It is a `PolicyClassifier` like any other: it turns text into category
names. It never grants authority and it never decides `allow`/`deny`/
`escalate` itself -- `PolicyEngine` still does that, deterministically, from
whatever categories any classifier (or combination of classifiers) reports.
This keeps NSA's central invariant intact: "intelligence is not authority".
A model may propose that text belongs to a category; only the policy engine
turns that into a decision.

Requires the optional `ml` extra (torch + transformers). Importing this
module never requires them; only constructing/using a classifier does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from nsa.policy import NSAPolicy

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
DEFAULT_HYPOTHESIS_TEMPLATE = "This text is about {}."


def _require_transformers():
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "The semantic classifier requires: pip install 'neural-state-architecture[ml]'"
        ) from exc
    return pipeline


@dataclass
class ZeroShotSemanticClassifier:
    """Classify text against policy categories with a pretrained NLI model.

    Unlike KeywordClassifier, this generalises: the model scores every
    category's short label against the *meaning* of the text, so a
    category doesn't need an exhaustive, hand-written pattern list to be
    detected. It is slower (one forward pass per call, scoring every
    label at once) and probabilistic (a threshold, not an exact match), so
    treat it as a backstop, not a replacement, for exact keyword rules --
    see `HybridClassifier`.

    labels: {category: short phrase}, e.g. from NSAPolicy.classifier_labels().
    threshold: minimum score (0-1) for a category to be reported.
    """

    labels: Mapping[str, str]
    model_name: str = DEFAULT_MODEL
    threshold: float = 0.55
    hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE
    device: Optional[str] = None
    max_length: int = 512
    _pipeline: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        # a duplicate label would make two categories indistinguishable to the model
        seen: dict[str, str] = {}
        for category, label in self.labels.items():
            if label in seen:
                raise ValueError(f"duplicate classifier label {label!r} for categories {seen[label]!r} and {category!r}")
            seen[label] = category

    @classmethod
    def from_policy(cls, policy: "NSAPolicy", **kwargs: Any) -> "ZeroShotSemanticClassifier":
        return cls(labels=policy.classifier_labels(), **kwargs)

    def _resolve_device(self) -> int:
        if self.device is not None:
            return 0 if str(self.device).startswith("cuda") else -1
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    def _load_pipeline(self) -> Any:
        if self._pipeline is None:
            pipeline = _require_transformers()
            self._pipeline = pipeline("zero-shot-classification", model=self.model_name, device=self._resolve_device())
        return self._pipeline

    def classify(self, text: str) -> Sequence[str]:
        if not text.strip() or not self.labels:
            return ()
        classifier = self._load_pipeline()
        label_to_category = {label: category for category, label in self.labels.items()}
        candidate_labels = list(label_to_category)
        result = classifier(
            text[: self.max_length * 8],  # a generous character cap; the model itself truncates by tokens
            candidate_labels,
            multi_label=True,
            hypothesis_template=self.hypothesis_template,
        )
        return tuple(
            label_to_category[label]
            for label, score in zip(result["labels"], result["scores"])
            if score >= self.threshold
        )


@dataclass
class HybridClassifier:
    """Union the categories reported by several classifiers.

    Typical use: a fast, exact KeywordClassifier plus a slower
    ZeroShotSemanticClassifier as a paraphrase backstop. Every classifier
    still only *proposes* categories; PolicyEngine remains the sole place a
    decision is made, so combining classifiers can only ever make the
    policy more (never less) likely to notice something -- it cannot
    weaken an existing keyword rule.

    short_circuit: skip the remaining (presumably slower) classifiers once
    an earlier one in the list has reported at least one category. This
    trades completeness (a later classifier might have added a *different*
    category) for latency: a zero-shot pass costs on the order of hundreds
    of milliseconds on CPU, so paying it again after an exact keyword hit
    already settled the decision is often wasted. Off by default, since the
    full picture (every matched category, from every classifier) is more
    useful for audit logging than the latency saving.
    """

    classifiers: Sequence[Any]
    short_circuit: bool = False

    def classify(self, text: str) -> Sequence[str]:
        seen: list[str] = []
        for classifier in self.classifiers:
            matched = classifier.classify(text)
            for category in matched:
                if category not in seen:
                    seen.append(category)
            if self.short_circuit and matched:
                break
        return tuple(seen)


__all__ = ["DEFAULT_MODEL", "DEFAULT_HYPOTHESIS_TEMPLATE", "ZeroShotSemanticClassifier", "HybridClassifier"]
