"""Declarative policy primitives for the NSA safety control plane."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import warnings
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nsa.enforcement import PolicyClassifier, PolicyEngine


_KNOWN_KEYS = frozenset({
    "name", "prohibited", "protected_data", "restricted_actions", "require_approval",
    "unknown_policy", "default_uncertainty",
})


def _string_set(data: Mapping[str, object], key: str) -> FrozenSet[str]:
    """A policy list field must really be a list of strings.

    A bare string would otherwise be iterated into a set of characters and
    silently weaken the policy.
    """
    raw = data.get(key)
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError(f"policy field {key!r} must be a list of strings, got {type(raw).__name__}")
    if not all(isinstance(item, str) for item in raw):
        raise ValueError(f"policy field {key!r} must contain only strings")
    return frozenset(raw)


@dataclass(frozen=True)
class PolicyRule:
    """One semantic policy rule and its declarative match patterns."""

    category: str
    mode: str = "deny"
    reason: Optional[str] = None
    patterns: Tuple[str, ...] = ()
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.category:
            raise ValueError("policy rule category must be non-empty")
        if self.mode not in {"deny", "escalate", "allow"}:
            raise ValueError("policy rule mode must be deny, escalate, or allow")

    @property
    def label(self) -> str:
        """Short human-readable phrase for this category (semantic classifier hypothesis text)."""
        return self.description or self.category.replace("_", " ").replace("-", " ")


@dataclass(frozen=True)
class NSAPolicy:
    """Human-configurable policy compiled into runtime decisions."""

    name: str = "default"
    prohibited: Tuple[PolicyRule, ...] = ()
    protected_data: FrozenSet[str] = frozenset()
    restricted_actions: FrozenSet[str] = frozenset()
    require_approval: FrozenSet[str] = frozenset()
    unknown_policy: str = "escalate"
    default_uncertainty: str = "escalate"

    def __post_init__(self) -> None:
        if self.unknown_policy not in {"allow", "deny", "escalate"}:
            raise ValueError("unknown_policy must be allow, deny, or escalate")
        if self.default_uncertainty not in {"allow", "deny", "escalate"}:
            raise ValueError("default_uncertainty must be allow, deny, or escalate")

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "NSAPolicy":
        if not isinstance(data, Mapping):
            raise ValueError(f"policy document must be a mapping, got {type(data).__name__}")
        unknown = sorted(str(k) for k in set(data) - _KNOWN_KEYS)
        if unknown:
            warnings.warn(f"unknown policy keys ignored: {unknown}", UserWarning, stacklevel=2)
        raw_rules = data.get("prohibited")
        rules = []
        if raw_rules is not None:
            if isinstance(raw_rules, (str, bytes)) or not isinstance(raw_rules, (list, tuple)):
                raise ValueError("policy field 'prohibited' must be a list of rules")
            for item in raw_rules:
                if isinstance(item, str):
                    rules.append(PolicyRule(item))
                elif isinstance(item, Mapping):
                    if "category" not in item:
                        raise ValueError("every prohibited rule needs a 'category'")
                    raw_patterns = item.get("patterns", ())
                    if isinstance(raw_patterns, (str, bytes)) or not isinstance(raw_patterns, (list, tuple)):
                        raise ValueError(f"patterns for {item['category']!r} must be a list of strings")
                    if not all(isinstance(p, str) for p in raw_patterns):
                        raise ValueError(f"patterns for {item['category']!r} must contain only strings")
                    description = item.get("description")
                    if description is not None and not isinstance(description, str):
                        raise ValueError(f"description for {item['category']!r} must be a string")
                    rules.append(
                        PolicyRule(
                            str(item["category"]),
                            str(item.get("mode", "deny")),
                            item.get("reason"),
                            tuple(raw_patterns),
                            description,
                        )
                    )
                else:
                    raise ValueError(f"unsupported prohibited rule: {item!r}")
        return cls(
            name=str(data.get("name", "default")),
            prohibited=tuple(rules),
            protected_data=_string_set(data, "protected_data"),
            restricted_actions=_string_set(data, "restricted_actions"),
            require_approval=_string_set(data, "require_approval"),
            unknown_policy=str(data.get("unknown_policy", "escalate")),
            default_uncertainty=str(data.get("default_uncertainty", "escalate")),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "NSAPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "NSAPolicy":
        """Load YAML when PyYAML is installed; YAML is an optional dependency."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML policy loading requires optional dependency 'pyyaml'") from exc
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle))

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "prohibited": [
                {"category": r.category, "mode": r.mode, "reason": r.reason, "patterns": list(r.patterns),
                 "description": r.description}
                for r in self.prohibited
            ],
            "protected_data": sorted(self.protected_data),
            "restricted_actions": sorted(self.restricted_actions),
            "require_approval": sorted(self.require_approval),
            "unknown_policy": self.unknown_policy,
            "default_uncertainty": self.default_uncertainty,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_mapping(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def rule_for(self, category: str) -> Optional[PolicyRule]:
        for rule in self.prohibited:
            if rule.category == category:
                return rule
        return None

    def classifier_patterns(self) -> dict[str, Tuple[str, ...]]:
        return {rule.category: rule.patterns for rule in self.prohibited if rule.patterns}

    def classifier_labels(self) -> dict[str, str]:
        """{category: short human-readable phrase}, for a semantic (zero-shot) classifier.

        Every prohibited category gets a label, whether or not it has
        keyword patterns, since the point of a semantic classifier is to
        catch phrasing the keyword patterns don't.
        """
        return {rule.category: rule.label for rule in self.prohibited}


class PolicyCompiler:
    """Compile a declarative policy into the executable NSA policy engine."""

    @staticmethod
    def compile(policy: NSAPolicy, classifier: "PolicyClassifier" = None) -> "PolicyEngine":
        from nsa.enforcement import KeywordClassifier, PolicyEngine

        return PolicyEngine(policy, classifier or KeywordClassifier(policy.classifier_patterns()))

    @staticmethod
    def compile_semantic(policy: NSAPolicy, *, semantic_classifier: "PolicyClassifier" = None,
                         include_keyword: bool = True, short_circuit: bool = False,
                         **semantic_kwargs: object) -> "PolicyEngine":
        """Compile with a pretrained semantic classifier as a paraphrase backstop.

        By default this is KeywordClassifier (fast, exact) plus
        ZeroShotSemanticClassifier (slower, generalises past exact patterns);
        pass `semantic_classifier` to supply your own, or `include_keyword=False`
        to use the semantic classifier alone. `short_circuit=True` skips the
        semantic pass once the keyword classifier already matched something,
        trading completeness for latency (see HybridClassifier). Requires the
        `ml` extra.
        """
        from nsa.enforcement import KeywordClassifier, PolicyEngine
        from nsa.semantic_classifier import HybridClassifier, ZeroShotSemanticClassifier

        semantic = semantic_classifier or ZeroShotSemanticClassifier.from_policy(policy, **semantic_kwargs)
        classifiers = [KeywordClassifier(policy.classifier_patterns()), semantic] if include_keyword else [semantic]
        return PolicyEngine(policy, HybridClassifier(classifiers, short_circuit=short_circuit))
