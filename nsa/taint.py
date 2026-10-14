"""Integrity (taint) tracking and tool-call authorisation against runtime state.

The whitepaper's integrity lattice (Biba: TRUSTED <= UNTRUSTED) says semantic content may
not manufacture authority. This module makes that operational for an agent loop:

* every observation carries an ``IntegrityLabel`` assigned by the *ingress path* (which knows
  where the bytes came from), never by the text itself;
* the integrity of the model's context is the join of the labels of what it was shown;
* a tool call is authorised against that state: an unknown tool or a missing capability is a
  DENY, and a side-effect tool whose *sink* arguments (recipient, path, URL, ...) have no
  trusted origin is REQUIRE_APPROVAL when the context is tainted.

"Trusted origin" is a data-flow check, not a text classifier: the argument value must appear
in a trusted observation or in trusted memory. That keeps user-directed actions working
("email the summary to bob@corp.com", where the recipient came from the user) while blocking
values introduced only by untrusted content.

Known limits: (1) the check is exact-substring on values, so an attacker who can only choose
*among* values already present in trusted context is not stopped; (2) labels on model output
are not tracked, so laundering through the model's own text is not covered; (3) the label
assignment at ingress is the trust boundary.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from nsa.algebra import IntegrityLabel
from nsa.cognition.tools import ToolSpec
from nsa.decision import Decision, SecurityDecision

POLICY_NAME = "integrity-taint"


def join_integrity(labels: Iterable[IntegrityLabel]) -> IntegrityLabel:
    out = IntegrityLabel.TRUSTED
    for label in labels:
        out = out.join(label)
    return out


def _text(payload: Any) -> str:
    return payload if isinstance(payload, str) else str(payload)


def has_trusted_origin(value: Any, trusted_texts: Sequence[str]) -> bool:
    """True if ``value`` occurs in a trusted text as a whole token.

    Boundary-aware so that a short value ("a", "/") does not count as "found" inside an
    unrelated word, and ``evil@x.io`` is not found inside ``devil@x.io.attacker``.
    """
    needle = _text(value).strip().lower()
    if not needle:
        return True  # nothing to attribute
    pattern = re.compile(r"(?<![\w@\-/])" + re.escape(needle) + r"(?![\w@\-/]|\.\w)")
    return any(pattern.search(text.lower()) for text in trusted_texts)


def authorize_tool_call(
    tool: ToolSpec | None,
    args: Mapping[str, Any],
    *,
    granted: Iterable[str],
    context_integrity: IntegrityLabel,
    trusted_texts: Sequence[str],
) -> SecurityDecision:
    if tool is None:
        return SecurityDecision(Decision.DENY, POLICY_NAME, "unknown tool (fail closed)", risk=1.0)
    granted = frozenset(granted)
    if tool.capability not in granted:
        return SecurityDecision(
            Decision.DENY, POLICY_NAME, f"capability {tool.capability!r} not granted",
            required_capabilities=frozenset({tool.capability}), risk=tool.risk,
        )
    if not tool.side_effect or context_integrity is IntegrityLabel.TRUSTED:
        return SecurityDecision(Decision.ALLOW, POLICY_NAME, "no untrusted influence on a privileged sink", risk=tool.risk)
    sinks = tuple(tool.metadata.get("sink_params", tuple(args)))
    orphaned = sorted(p for p in sinks if p in args and not has_trusted_origin(args[p], trusted_texts))
    if orphaned:
        return SecurityDecision(
            Decision.REQUIRE_APPROVAL, POLICY_NAME,
            "context is untrusted and sink argument(s) have no trusted origin: " + ", ".join(orphaned),
            required_capabilities=frozenset({tool.capability}), risk=max(tool.risk, 0.8),
            metadata={"orphaned_params": tuple(orphaned), "context_integrity": context_integrity.name},
        )
    return SecurityDecision(
        Decision.ALLOW, POLICY_NAME, "untrusted context, but every sink argument has a trusted origin",
        risk=tool.risk, metadata={"context_integrity": context_integrity.name},
    )


__all__ = ["authorize_tool_call", "has_trusted_origin", "join_integrity", "POLICY_NAME"]
