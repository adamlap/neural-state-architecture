"""Signed ingress: integrity labels that content cannot assert for itself.

``nsa/taint.py`` trusts the ``IntegrityLabel`` attached to an observation. Without more, any
caller can pass ``integrity=TRUSTED``, so the label is only as good as the code that sets it.
An ``IngressAuthority`` holds a secret that lives only in the trusted ingress component (the
code that actually knows where the bytes came from: the operator console, an authenticated
user session, a tool wrapper). It *seals* content into an ``IngressEnvelope`` binding

    (source, integrity label, SHA-256 of the content, nonce, issue time)

under HMAC-SHA256. The runtime accepts a TRUSTED label only if the envelope verifies for exactly
that content. Anything else (tampered content, a label edited after signing, a signature from
another key, a replayed or expired envelope) degrades to UNTRUSTED. It never upgrades.

Limits: this moves the trust boundary to the secret; it does not remove it. A process that can
read the secret can mint trusted envelopes, so the secret must not be reachable from the model,
its tools or the content path. Envelopes carry no confidentiality.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

from nsa.algebra import IntegrityLabel


class IngressError(ValueError):
    """Raised by ``verify(..., strict=True)`` when an envelope does not verify."""


@dataclass(frozen=True)
class IngressEnvelope:
    source: str
    integrity: IntegrityLabel
    content_digest: str
    nonce: str
    issued_at: float
    signature: str


@dataclass(frozen=True)
class IngressResult:
    label: IntegrityLabel
    verified: bool
    reason: str


def content_digest(content: Any) -> str:
    text = content if isinstance(content, str) else repr(content)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _signing_input(source: str, integrity: IntegrityLabel, digest: str, nonce: str, issued_at: float) -> bytes:
    return "\x1f".join((source, integrity.name, digest, nonce, f"{issued_at:.6f}")).encode("utf-8")


class IngressAuthority:
    def __init__(self, secret: bytes, *, max_age_sec: float = 300.0, clock: Callable[[], float] = time.time,
                 replay_window: int = 100_000) -> None:
        if len(secret) < 16:
            raise ValueError("ingress secret must be at least 16 bytes")
        self._secret = bytes(secret)
        self.max_age_sec = max_age_sec
        self._clock = clock
        self._seen: "OrderedDict[str, None]" = OrderedDict()
        self._replay_window = replay_window

    @classmethod
    def ephemeral(cls, **kwargs: Any) -> "IngressAuthority":
        return cls(secrets.token_bytes(32), **kwargs)

    def seal(self, content: Any, *, source: str, integrity: IntegrityLabel = IntegrityLabel.TRUSTED) -> IngressEnvelope:
        nonce, issued = secrets.token_hex(16), self._clock()
        digest = content_digest(content)
        sig = hmac.new(self._secret, _signing_input(source, integrity, digest, nonce, issued), hashlib.sha256).hexdigest()
        return IngressEnvelope(source, integrity, digest, nonce, issued, sig)

    def verify(self, content: Any, envelope: IngressEnvelope | None, *, strict: bool = False) -> IngressResult:
        """Return the label the runtime may use for ``content``; failures yield UNTRUSTED (never an upgrade)."""
        def fail(reason: str) -> IngressResult:
            if strict:
                raise IngressError(reason)
            return IngressResult(IntegrityLabel.UNTRUSTED, False, reason)

        if envelope is None:
            return fail("no envelope")
        expected = hmac.new(
            self._secret,
            _signing_input(envelope.source, envelope.integrity, envelope.content_digest, envelope.nonce, envelope.issued_at),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(envelope.signature, expected):
            return fail("bad signature (forged, or fields edited after signing)")
        if not hmac.compare_digest(envelope.content_digest, content_digest(content)):
            return fail("content does not match the sealed digest")
        age = self._clock() - envelope.issued_at
        if age > self.max_age_sec or age < -5.0:
            return fail("envelope expired or from the future")
        if envelope.nonce in self._seen:
            return fail("replayed envelope")
        self._seen[envelope.nonce] = None
        while len(self._seen) > self._replay_window:
            self._seen.popitem(last=False)
        return IngressResult(envelope.integrity, True, "verified")


__all__ = ["IngressAuthority", "IngressEnvelope", "IngressError", "IngressResult", "content_digest"]
