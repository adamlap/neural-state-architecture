"""Fact extraction: turn free-text observations into canonical-memory writes.

Two extractors share one interface (``extract(text) -> list[ExtractedFact]``):

* ``RuleFactExtractor``  - dependency-free patterns for copular statements ("X is Y", "X = Y",
  "set X to Y"), with update cues ("now", "changed to", "updated to", "correction") marking
  ``kind="update"``. Fast and deterministic; its recall on unseen phrasings is limited and is
  measured, not assumed (see research/RESEARCH_BENCHMARK_V9.md).
* ``LLMFactExtractor``   - asks a model for ``entity | attribute | value | fact/update`` lines.
  Costs a model call per batch; parsing is strict and fails closed (unparseable lines are dropped).

The runtime passes the *observation's integrity label* to the memory write, so an untrusted
source's extracted claims are subject to the same gate as any other untrusted write.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

_ARTICLES = {"the", "a", "an", "our", "my", "your", "its", "their", "this", "that"}
_UPDATE_CUES = re.compile(
    r"\b(now|changed|change[sd]? to|updated?|updated to|switched|moved to|migrated|correction|corrected|replaces?|no longer|rotated)\b", re.I)
_VALUE = r"[A-Za-z0-9][A-Za-z0-9_.@:/\-]*"


@dataclass(frozen=True)
class ExtractedFact:
    key: str          # normalised "entity attribute" phrase
    value: str
    kind: str = "fact"   # "fact" | "update"


def normalize_key(text: str) -> str:
    """Canonical key: lower-cased content words, de-duplicated and sorted.

    Order-insensitive on purpose. "staging cache size" and "the cache size for staging" must be the
    same key; otherwise an untrusted restatement in a different word order creates a *second* key,
    never meets the integrity gate, and can win retrieval (found by benchmark v9).
    """
    words = {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _ARTICLES and w not in {"for", "of", "s"}}
    return " ".join(sorted(words))


class FactExtractor(Protocol):
    def extract(self, text: str) -> list[ExtractedFact]: ...


# (pattern, key-group, value-group). Ordered most specific first; the first match per sentence wins.
_PATTERNS: tuple[tuple[re.Pattern, int, int], ...] = tuple(
    (re.compile(p, re.I), k, v) for p, k, v in (
        (rf"\b(?:set|change|switch|update|rotate)\s+(?:the\s+)?(.+?)\s+to\s+({_VALUE})", 1, 2),
        (rf"\b(?:for|on)\s+(.+?),\s*(?:use|set)\s+({_VALUE})\s+as\s+(?:the\s+)?(.+?)(?:[.;!]|$)", 0, 0),   # handled specially
        (rf"(.+?)\s+(?:has been|have been|was|were|is|are)\s+(?:now\s+)?(?:updated|changed|switched|moved|rotated|set)\s+to\s+({_VALUE})", 1, 2),
        (rf"(.+?)\s+(?:is|are)\s+now\s+({_VALUE})", 1, 2),
        (rf"(.+?)\s*(?:=|:)\s*({_VALUE})\s*(?:\(.*\))?\s*[.;!]*$", 1, 2),
        (rf"(.+?)\s+(?:is|are)\s+({_VALUE})\s*[.;!]*$", 1, 2),
    )
)
_USE_AS = re.compile(rf"\b(?:for|on)\s+(.+?),\s*(?:use|set)\s+({_VALUE})\s+as\s+(?:the\s+)?(.+?)(?:[.;!]|$)", re.I)
_LEAD = re.compile(r"^\s*(?:fyi|reminder|note|update|correction|heads up|btw)\s*[-:,]?\s*", re.I)
_CHANGED_TO = re.compile(rf"\b(?:we|i|they)\s+(?:changed|switched|moved|updated|rotated)\s+(.+?)\s+to\s+({_VALUE})", re.I)


class RuleFactExtractor:
    def extract(self, text: str) -> list[ExtractedFact]:
        facts: list[ExtractedFact] = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
            sentence = sentence.strip()
            if not sentence:
                continue
            is_update = bool(_UPDATE_CUES.search(sentence))
            body = _LEAD.sub("", sentence)
            m = _USE_AS.search(body)
            if m:
                facts.append(ExtractedFact(normalize_key(f"{m.group(1)} {m.group(3)}"), m.group(2).strip(".,;"), "update" if is_update else "fact"))
                continue
            m = _CHANGED_TO.search(body)
            if m:
                facts.append(ExtractedFact(normalize_key(m.group(1)), m.group(2).strip(".,;"), "update"))
                continue
            for pattern, kg, vg in _PATTERNS:
                if kg == 0:
                    continue
                m = pattern.search(body)
                if m:
                    key, value = normalize_key(m.group(kg)), m.group(vg).strip(".,;")
                    if key and value and len(key.split()) <= 6:
                        facts.append(ExtractedFact(key, value, "update" if is_update else "fact"))
                    break
        return facts


_LLM_PROMPT = (
    "Extract every fact that states a value for something. Output one line per fact in exactly this format:\n"
    "entity | attribute | value | kind\n"
    "kind is 'update' if the text says the value changed, otherwise 'fact'. Output NONE if there are no facts. "
    "Output nothing else.\n\nTEXT:\n{text}\n"
)


class LLMFactExtractor:
    def __init__(self, backend: Any, *, max_chars: int = 4000) -> None:
        self.backend = backend
        self.max_chars = max_chars
        self.calls = 0

    def extract(self, text: str) -> list[ExtractedFact]:
        self.calls += 1
        out = self.backend.generate(_LLM_PROMPT.format(text=text[: self.max_chars]))
        reply = out.text if hasattr(out, "text") else str(out)
        return parse_llm_facts(reply)


def parse_llm_facts(reply: str) -> list[ExtractedFact]:
    facts = []
    for line in reply.splitlines():
        parts = [p.strip() for p in line.strip().strip("-*• ").split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            continue
        key = normalize_key(f"{parts[0]} {parts[1]}")
        value = parts[2].strip(" .,;'\"`")
        if key and value and not value.lower().startswith(("none", "n/a")):
            kind = "update" if len(parts) > 3 and parts[3].lower().startswith("update") else "fact"
            facts.append(ExtractedFact(key, value, kind))
    return facts


def key_overlap(question: str, key: str) -> float:
    """Token-overlap score between a question and a normalised key (0..1, fraction of key tokens present)."""
    q, k = set(normalize_key(question).split()), set(key.split())
    return len(q & k) / len(k) if k else 0.0


def rank_keys(question: str, keys: Iterable[str], *, limit: int = 3, min_score: float = 0.5) -> list[str]:
    scored = sorted(((key_overlap(question, k), k) for k in keys), key=lambda t: (-t[0], t[1]))
    return [k for s, k in scored if s >= min_score][:limit]


__all__ = ["ExtractedFact", "FactExtractor", "RuleFactExtractor", "LLMFactExtractor", "parse_llm_facts",
           "normalize_key", "key_overlap", "rank_keys"]
