"""Episodic-to-semantic memory consolidation.

Orchestrates offline/idle consolidation cycles:
1. Replays raw episodic trajectory records.
2. Induces generalized causal rules and empirical outcome priors.
3. Synthesizes semantic state updates and tunes residency predictors.
4. Prunes stale working memory and demotes inactive regions to cold storage.

Enforces the strict governance boundary: consolidation cannot alter hard authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from nsa.cce.trajectory import CognitiveTrajectory, TrajectoryRecord
from nsa.core.state import CanonicalState, SemanticState
from nsa.residency.manager import NeuralResidencyManager
from nsa.residency.types import MemoryTier


@dataclass(frozen=True)
class ConsolidatedRule:
    """An empirical rule induced from replaying episodic trajectories."""
    action_id: str
    sample_count: int
    commit_rate: float
    context_tags: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class ConsolidationReport:
    """Summary of operations performed during a consolidation cycle."""
    replayed_episodes: int
    induced_rules: tuple[ConsolidatedRule, ...]
    pruned_tags_count: int
    demoted_regions_count: int
    duration_ms: float
    timestamp: float = field(default_factory=monotonic)


class MemoryConsolidator:
    """Autonomous consolidation engine transforming episodic logs into semantic schema."""

    def __init__(
        self,
        min_samples_for_rule: int = 2,
        commit_rate_threshold: float = 0.5,
    ) -> None:
        self.min_samples_for_rule = max(1, min_samples_for_rule)
        self.commit_rate_threshold = commit_rate_threshold

    def consolidate(
        self,
        trajectory: CognitiveTrajectory,
        current_state: CanonicalState,
        residency_manager: Optional[NeuralResidencyManager] = None,
        retention_window: int = 50,
    ) -> tuple[CanonicalState, ConsolidationReport]:
        """Execute a full consolidation cycle over a cognitive trajectory."""
        start_time = monotonic()
        records = trajectory.records
        if not records:
            empty_report = ConsolidationReport(
                replayed_episodes=0,
                induced_rules=(),
                pruned_tags_count=0,
                demoted_regions_count=0,
                duration_ms=0.0,
            )
            return current_state, empty_report

        # Stage 1 & 2: Replay & Inductive Pattern Induction
        action_stats: Dict[str, Dict[str, Any]] = {}
        for rec in records[-retention_window:]:
            receipt = rec.receipt
            if receipt is None or not receipt.action_id:
                continue
            act = receipt.action_id
            stats = action_stats.setdefault(act, {"attempts": 0, "commits": 0, "tags": set()})
            stats["attempts"] += 1
            if receipt.committed:
                stats["commits"] += 1
            for event in rec.events:
                stats["tags"].add(event.kind.value)

        # Stage 3: Semantic Schema Synthesis
        induced_rules: List[ConsolidatedRule] = []
        for act, stats in action_stats.items():
            if stats["attempts"] >= self.min_samples_for_rule:
                rate = stats["commits"] / stats["attempts"]
                rule = ConsolidatedRule(
                    action_id=act,
                    sample_count=stats["attempts"],
                    commit_rate=round(rate, 4),
                    context_tags=tuple(sorted(stats["tags"])),
                    summary=f"Action '{act}' has {rate:.1%} commit rate over {stats['attempts']} trials.",
                )
                induced_rules.append(rule)

                # Feed outcome back to residency predictor if available
                if residency_manager is not None and hasattr(residency_manager.predictor, "observe_state_tags"):
                    for tag in stats["tags"]:
                        residency_manager.predictor.observe_state_tags([tag], act)

        # Update SemanticState with consolidated knowledge
        existing_semantic = current_state.semantic.value if isinstance(current_state.semantic.value, dict) else {}
        new_semantic_dict = dict(existing_semantic)
        consolidated_dict = new_semantic_dict.setdefault("consolidated_rules", {})
        for r in induced_rules:
            consolidated_dict[r.action_id] = {
                "commit_rate": r.commit_rate,
                "samples": r.sample_count,
                "tags": list(r.context_tags),
            }

        # Stage 4: Working Memory Compaction & Residency Cache Demotion
        pruned_count = 0
        if "transient_observations" in new_semantic_dict:
            pruned_count = len(new_semantic_dict["transient_observations"])
            del new_semantic_dict["transient_observations"]

        demoted_count = 0
        if residency_manager is not None:
            # Demote resident regions that were not accessed recently
            recent_regions: Set[str] = set()
            for rec in records[-10:]:
                if rec.receipt and rec.receipt.action_id:
                    recent_regions.add(rec.receipt.action_id)
            for rid, tier in list(residency_manager.tiers.items()):
                if tier == MemoryTier.VRAM and rid not in recent_regions and rid != residency_manager.current_region:
                    residency_manager.record_resident(rid, MemoryTier.RAM, reason="consolidation-demoted")
                    demoted_count += 1

        # Advance CanonicalState step and commit consolidated semantic knowledge
        updated_state = current_state.with_semantic(new_semantic_dict)

        duration_ms = (monotonic() - start_time) * 1000
        report = ConsolidationReport(
            replayed_episodes=len(records[-retention_window:]),
            induced_rules=tuple(induced_rules),
            pruned_tags_count=pruned_count,
            demoted_regions_count=demoted_count,
            duration_ms=round(duration_ms, 2),
        )

        return updated_state, report