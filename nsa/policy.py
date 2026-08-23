"""Declarative policy primitives for the NSA safety control plane."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from nsa.enforcement import PolicyClassifier, PolicyEngine


@dataclass(frozen=True)
class PolicyRule:
    """One semantic policy rule."""
    category: str
    mode: str = "deny"
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.category:
            raise ValueError("policy rule category must be non-empty")
        if self.mode not in {"deny", "escalate", "allow"}:
            raise ValueError("policy rule mode must be deny, escalate, or allow")


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
        raw_rules = data.get("prohibited", ())
        rules = []
        if isinstance(raw_rules, Sequence) and not isinstance(raw_rules, (str, bytes)):
            for item in raw_rules:
                if isinstance(item, str):
                    rules.append(PolicyRule(item))
                elif isinstance(item, Mapping):
                    rules.append(PolicyRule(str(item["category"]), str(item.get("mode", "deny")), item.get("reason")))
        return cls(
            name=str(data.get("name", "default")),
            prohibited=tuple(rules),
            protected_data=frozenset(str(x) for x in data.get("protected_data", ()) or ()),
            restricted_actions=frozenset(str(x) for x in data.get("restricted_actions", ()) or ()),
            require_approval=frozenset(str(x) for x in data.get("require_approval", ()) or ()),
            unknown_policy=str(data.get("unknown_policy", "escalate")),
            default_uncertainty=str(data.get("default_uncertainty", "escalate")),
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "NSAPolicy":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "NSAPolicy":
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML policy loading requires optional dependency 'pyyaml'") from exc
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(yaml.safe_load(handle))

    def to_mapping(self) -> dict:
        return {
            "name": self.name,
            "prohibited": [{"category": r.category, "mode": r.mode, "reason": r.reason} for r in self.prohibited],
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


class PolicyCompiler:
    """Compile a declarative policy into the executable NSA policy engine."""
    @staticmethod
    def compile(policy: NSAPolicy, classifier: "PolicyClassifier") -> "PolicyEngine":
        from nsa.enforcement import PolicyEngine
        return PolicyEngine(policy, classifier)
