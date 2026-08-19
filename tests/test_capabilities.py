from datetime import datetime, timedelta, timezone

import pytest

from nsa.capabilities import Capability, CapabilityAuthority


def capability(**overrides):
    values = dict(
        capability_id="cap-1",
        issuer="runtime",
        subject="agent-1",
        action="filesystem.read",
        scope="/safe/data",
        purpose="task",
    )
    values.update(overrides)
    return Capability(**values)


def test_authority_issues_only_its_own_capabilities():
    authority = CapabilityAuthority("runtime")
    issued = authority.issue(capability())
    assert issued.allows("cap-1", "agent-1", "filesystem.read", "/safe/data")

    with pytest.raises(PermissionError):
        authority.issue(capability(issuer="model"))


def test_capability_is_narrowly_scoped():
    authority = CapabilityAuthority("runtime").issue(capability())
    assert not authority.allows("cap-1", "agent-1", "filesystem.write", "/safe/data")
    assert not authority.allows("cap-1", "agent-1", "filesystem.read", "/other")
    assert not authority.allows("cap-1", "agent-2", "filesystem.read", "/safe/data")


def test_expired_capability_is_rejected():
    expired = capability(expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    authority = CapabilityAuthority("runtime").issue(expired)
    assert not authority.allows("cap-1", "agent-1", "filesystem.read", "/safe/data")
