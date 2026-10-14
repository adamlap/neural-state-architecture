"""Answer extraction and grading.

``strict`` is the benchmark's original exact-match rule. ``lenient`` additionally
accepts a correct value token that is wrapped in template punctuation
(``VALUE=<V007_000>``, ``VALUE="V007_000"``) or followed by echoed memory metadata
(``VALUE=V007_000_VERSION_2_STATUS_CURRENT``). Reporting both separates "did the
model use the right memory" from "did it follow the output format"; a rendering
change that only affects the latter should not look like a memory regression.
"""
from __future__ import annotations

import re

VALUE_RE = re.compile(r"VALUE\s*=\s*([A-Z0-9_-]+)", re.IGNORECASE)
_ECHO_SUFFIXES = ("_VERSION", "_STATUS", "_CURRENT_VALUE")


def extract(text: str | None) -> str | None:
    match = VALUE_RE.search(text or "")
    return match.group(1).upper() if match else None


def _parts(expected: str) -> list[str]:
    return [p.strip().upper() for p in expected.split("|")]


def strict_correct(text: str | None, expected: str) -> bool:
    if "|" in expected:
        return (text or "").strip().upper() == f"VALUE={expected.upper()}"
    return extract(text) == expected.upper()


def _token_ok(token: str, expected: str) -> bool:
    return token == expected or any(token.startswith(expected + suffix) for suffix in _ECHO_SUFFIXES)


_WRAPPERS = re.compile(r"[<>\"'`\[\]()*]")


def lenient_correct(text: str | None, expected: str) -> bool:
    if strict_correct(text, expected):
        return True
    # drop template punctuation around the answer, e.g. VALUE=<V007_000>
    text = _WRAPPERS.sub("", text or "")
    if strict_correct(text, expected):
        return True
    raw = text.strip().upper()
    if "|" in expected:
        match = re.match(r"VALUE\s*=\s*(.+)$", raw, re.DOTALL)
        if not match:
            return False
        got = [p.strip() for p in match.group(1).split("|")]
        want = _parts(expected)
        return len(got) == len(want) and all(_token_ok(re.sub(r"[^A-Z0-9_-].*$", "", g), w) for g, w in zip(got, want))
    token = extract(text)
    return token is not None and _token_ok(token, expected.upper())
