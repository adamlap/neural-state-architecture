"""Tests for CCEConfig and CLI helpers (Phase CCE-8)."""
from __future__ import annotations

import pathlib
from nsa.runtime.cce_config import CCEConfig


def test_cce_config_defaults_and_serialization(tmp_path: pathlib.Path):
    cfg = CCEConfig(dimension=8, port=9000, model="qwen2.5:7b")
    assert cfg.dimension == 8
    assert cfg.port == 9000
    assert cfg.model == "qwen2.5:7b"

    p = tmp_path / "cce_test_config.json"
    cfg.save(p)
    assert p.exists()

    loaded = CCEConfig.load(p)
    assert loaded.dimension == 8
    assert loaded.port == 9000
    assert loaded.model == "qwen2.5:7b"
