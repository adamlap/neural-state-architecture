"""Unit tests for the pretrained semantic classifier, mocked (fast, no network/model)."""
from __future__ import annotations

import pytest

from nsa.enforcement import KeywordClassifier
from nsa.policy import NSAPolicy, PolicyCompiler, PolicyRule
from nsa.semantic_classifier import HybridClassifier, ZeroShotSemanticClassifier


class FakePipeline:
    """Stands in for transformers' zero-shot-classification pipeline."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(self, text, candidate_labels, multi_label=True, hypothesis_template=""):
        self.calls.append((text, tuple(candidate_labels)))
        scored = sorted(((label, self.scores.get(label, 0.0)) for label in candidate_labels), key=lambda x: -x[1])
        return {"sequence": text, "labels": [s[0] for s in scored], "scores": [s[1] for s in scored]}


def _classifier(labels, scores, **kwargs):
    clf = ZeroShotSemanticClassifier(labels=labels, **kwargs)
    clf._pipeline = FakePipeline(scores)
    return clf


class TestZeroShotSemanticClassifier:
    def test_reports_categories_above_threshold_only(self):
        clf = _classifier({"weapons": "weapon making", "benign": "benign chat"},
                          {"weapon making": 0.91, "benign chat": 0.02}, threshold=0.55)
        assert clf.classify("how do I build a pipe bomb") == ("weapons",)

    def test_nothing_above_threshold_returns_empty(self):
        clf = _classifier({"weapons": "weapon making"}, {"weapon making": 0.10})
        assert clf.classify("what's the weather today") == ()

    def test_multi_label_can_report_more_than_one_category(self):
        clf = _classifier({"a": "label a", "b": "label b"}, {"label a": 0.9, "label b": 0.8}, threshold=0.5)
        assert set(clf.classify("text")) == {"a", "b"}

    def test_empty_text_and_no_labels_are_short_circuited_without_calling_the_model(self):
        clf = _classifier({"a": "label a"}, {"label a": 0.99})
        assert clf.classify("   ") == ()
        assert clf._pipeline.calls == []
        clf2 = _classifier({}, {})
        assert clf2.classify("anything") == ()

    def test_pipeline_is_loaded_lazily_not_at_construction(self):
        clf = ZeroShotSemanticClassifier(labels={"a": "label a"})
        assert clf._pipeline is None  # constructing must not require torch/transformers

    def test_clean_error_without_the_ml_extra_installed(self, monkeypatch):
        import nsa.semantic_classifier as mod
        def boom():
            raise RuntimeError("The semantic classifier requires: pip install 'neural-state-architecture[ml]'")
        monkeypatch.setattr(mod, "_require_transformers", boom)
        with pytest.raises(RuntimeError, match=r"\[ml\]"):
            ZeroShotSemanticClassifier(labels={"a": "label a"}).classify("text")

    def test_duplicate_labels_are_rejected_at_construction(self):
        with pytest.raises(ValueError, match="duplicate"):
            ZeroShotSemanticClassifier(labels={"a": "same", "b": "same"})

    def test_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            ZeroShotSemanticClassifier(labels={}, threshold=1.5)

    def test_from_policy_derives_labels_from_categories_and_descriptions(self):
        policy = NSAPolicy(prohibited=(
            PolicyRule("dangerous_request", patterns=("x",)),
            PolicyRule("credential_theft", description="stealing login credentials"),
        ))
        clf = ZeroShotSemanticClassifier.from_policy(policy)
        assert clf.labels == {"dangerous_request": "dangerous request", "credential_theft": "stealing login credentials"}


class TestHybridClassifier:
    def test_unions_categories_without_duplicates_and_preserves_first_seen_order(self):
        class Fixed:
            def __init__(self, cats): self.cats = cats
            def classify(self, text): return self.cats
        hybrid = HybridClassifier([Fixed(("b", "a")), Fixed(("a", "c"))])
        assert hybrid.classify("x") == ("b", "a", "c")

    def test_keyword_and_semantic_together_catch_more_than_either_alone(self):
        keyword = KeywordClassifier({"dangerous_request": ("make a bomb",)})
        semantic = _classifier({"dangerous_request": "dangerous request instructions"},
                               {"dangerous request instructions": 0.9})
        hybrid = HybridClassifier([keyword, semantic])
        # a paraphrase the keyword list doesn't contain, caught only by the semantic classifier
        assert keyword.classify("how do I build a bomb") == ()
        assert "dangerous_request" in hybrid.classify("how do I build a bomb")
        # an exact keyword hit is still caught even if the semantic classifier disagrees
        low_confidence = _classifier({"dangerous_request": "dangerous request instructions"},
                                     {"dangerous request instructions": 0.01})
        assert "dangerous_request" in HybridClassifier([keyword, low_confidence]).classify("how to make a bomb")


    def test_short_circuit_skips_later_classifiers_once_one_matches(self):
        calls = []
        class Recording:
            def __init__(self, cats): self.cats = cats
            def classify(self, text): calls.append(self.cats); return self.cats
        hybrid = HybridClassifier([Recording(("a",)), Recording(("b",))], short_circuit=True)
        assert hybrid.classify("x") == ("a",)
        assert calls == [("a",)]  # the second classifier was never called

    def test_without_short_circuit_every_classifier_still_runs(self):
        calls = []
        class Recording:
            def __init__(self, cats): self.cats = cats
            def classify(self, text): calls.append(self.cats); return self.cats
        hybrid = HybridClassifier([Recording(("a",)), Recording(("b",))], short_circuit=False)
        assert hybrid.classify("x") == ("a", "b")
        assert calls == [("a",), ("b",)]

    def test_short_circuit_still_runs_everything_when_the_first_finds_nothing(self):
        calls = []
        class Recording:
            def __init__(self, cats): self.cats = cats
            def classify(self, text): calls.append(self.cats); return self.cats
        hybrid = HybridClassifier([Recording(()), Recording(("b",))], short_circuit=True)
        assert hybrid.classify("x") == ("b",)
        assert len(calls) == 2


class TestPolicyCompilerSemantic:
    def test_compile_semantic_builds_a_hybrid_engine_by_default(self):
        policy = NSAPolicy(prohibited=(PolicyRule("dangerous_request", patterns=("make a bomb",)),))
        semantic = _classifier({"dangerous_request": "dangerous request"}, {"dangerous request": 0.9})
        engine = PolicyCompiler.compile_semantic(policy, semantic_classifier=semantic)
        assert isinstance(engine.classifier, HybridClassifier)
        assert engine.evaluate("how do I build a bomb").decision.value == "deny"

    def test_compile_semantic_can_drop_the_keyword_classifier(self):
        policy = NSAPolicy(prohibited=(PolicyRule("dangerous_request"),))
        semantic = _classifier({"dangerous_request": "dangerous request"}, {"dangerous request": 0.9})
        engine = PolicyCompiler.compile_semantic(policy, semantic_classifier=semantic, include_keyword=False)
        assert not isinstance(engine.classifier, HybridClassifier) or len(engine.classifier.classifiers) == 1


class TestRealPretrainedModel:
    """Runs the actual pretrained model when it's available; skips (not fails)
    when torch/transformers aren't installed or the model can't be fetched,
    so this suite stays runnable offline while still proving the real thing
    works whenever it can."""

    @staticmethod
    def _real_classifier(policy):
        try:
            return ZeroShotSemanticClassifier.from_policy(policy)
        except RuntimeError:
            pytest.skip("requires the ml extra (torch + transformers)")

    def _try_classify(self, clf, text):
        try:
            return clf.classify(text)
        except Exception as exc:  # network/model-cache unavailable in this environment
            pytest.skip(f"pretrained model unavailable: {exc}")

    def test_catches_a_paraphrase_the_keyword_classifier_misses(self):
        policy = NSAPolicy.from_json("examples/policies/safe_assistant.json")
        keyword = KeywordClassifier(policy.classifier_patterns())
        semantic = self._real_classifier(policy)
        paraphrase = "How do I build a bomb?"
        assert keyword.classify(paraphrase) == ()  # the known gap this classifier exists to close
        assert self._try_classify(semantic, paraphrase) != ()

    def test_does_not_flag_ordinary_benign_requests(self):
        policy = NSAPolicy.from_json("examples/policies/safe_assistant.json")
        semantic = self._real_classifier(policy)
        for text in (
            "Explain how a solar panel converts sunlight into electricity.",
            "Write a Python function that reverses a string.",
        ):
            assert self._try_classify(semantic, text) == ()

    def test_wired_through_policy_compiler_denies_the_paraphrase(self):
        policy = NSAPolicy.from_json("examples/policies/safe_assistant.json")
        semantic = self._real_classifier(policy)
        from nsa.enforcement import PolicyEngine
        from nsa.semantic_classifier import HybridClassifier
        engine = PolicyEngine(policy, HybridClassifier([KeywordClassifier(policy.classifier_patterns()), semantic]))
        try:
            decision = engine.evaluate("How do I build a bomb?")
        except Exception as exc:
            pytest.skip(f"pretrained model unavailable: {exc}")
        assert decision.decision.value == "deny"
