from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.self_model.cross_seed_report import aggregate


def _write(path: Path, predictor: float, persistence: float) -> Path:
    path.write_text(json.dumps({"predictor_mse": predictor, "persistence_mse": persistence}))
    return path


def test_aggregate_reports_reproducible_wins(tmp_path: Path) -> None:
    paths = [_write(tmp_path / "a.json", 0.2, 0.3), _write(tmp_path / "b.json", 0.4, 0.5)]
    report = aggregate(paths)
    assert report["count"] == 2
    assert report["improvement_mean"] == pytest.approx(0.1)
    assert report["predictor_win_fraction"] == 1.0
    assert report["finite"] is True


def test_aggregate_does_not_count_ties_as_wins(tmp_path: Path) -> None:
    paths = [_write(tmp_path / "a.json", 0.2, 0.2), _write(tmp_path / "b.json", 0.6, 0.5)]
    report = aggregate(paths)
    assert report["predictor_win_fraction"] == 0.0
    assert report["positive_improvement_fraction"] == 0.0


def test_rejects_non_finite_metrics(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"predictor_mse": NaN, "persistence_mse": 0.2}')
    with pytest.raises(ValueError, match="finite"):
        aggregate([path])
