import argparse
from types import SimpleNamespace

import pytest

from tests.tiny_qwen import make_tiny_qwen, torch_backend_usable

pytestmark = pytest.mark.skipif(not torch_backend_usable(), reason="needs a working torch + transformers + accelerate stack")


def _row(prefetch, decode, prefetched, sha="x", dropped=3, hit=1.0, already_resident=0):
    return {
        "prefetch": prefetch, "decode_sec": decode, "tokens_per_sec": 10 / decode, "load_sec": 1.0,
        "output_sha256": sha, "page_cache_files_dropped": dropped, "bytes_already_resident": already_resident,
        "trace": {"bytes_prefetched": prefetched, "prefetch_hit_rate": hit, "prefetch_coverage": hit},
    }


def test_summary_reports_speedup_and_flags_meaningless_comparisons():
    from experiments.residency.benchmark_matrix import _summarize
    good = _summarize([_row(True, 1.0, 100), _row(False, 2.0, 0), _row(True, 1.2, 100), _row(False, 2.2, 0)])
    assert good["decode_speedup_on_vs_off"] == pytest.approx(2.1 / 1.1)
    assert good["outputs_identical_across_runs"] and good["notes"] == []

    noop = _summarize([_row(True, 1.0, 0), _row(False, 1.0, 0)])
    assert any("touched no bytes" in note for note in noop["notes"])

    # bytes_prefetched==0 alone is not a problem: the background prefetch thread can
    # legitimately lose its race and find the region already resident instead (see
    # test_resident_backend.py). That must not be flagged as "nothing happened".
    raced_but_found = _summarize([_row(True, 1.0, 0, already_resident=100), _row(False, 1.0, 0)])
    assert not any("touched no bytes" in note for note in raced_but_found["notes"])

    diverged = _summarize([_row(True, 1.0, 5, sha="a"), _row(False, 1.0, 0, sha="b")])
    assert not diverged["outputs_identical_across_runs"]
    assert any("differ" in note for note in diverged["notes"])

    warm = _summarize([_row(True, 1.0, 5, dropped=0), _row(False, 1.0, 0, dropped=0)])
    assert any("page cache was not dropped" in note for note in warm["notes"])


def test_drop_page_cache_counts_files(tmp_path):
    from experiments.residency.benchmark_matrix import _drop_page_cache
    (tmp_path / "a.bin").write_bytes(b"x" * 1024)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"y" * 1024)
    assert _drop_page_cache(tmp_path, tmp_path / "missing") == 2


def test_matrix_end_to_end_alternates_order_and_compares_outputs(tmp_path, monkeypatch):
    import experiments.residency.benchmark_matrix as bm
    monkeypatch.setenv("HOME", str(tmp_path))  # offload folder lives under ~/.cache/nsa
    monkeypatch.setattr(bm, "get_local_model", lambda key: SimpleNamespace(model_id="tiny/qwen"))
    root = make_tiny_qwen(tmp_path / "model")
    args = argparse.Namespace(
        model="1.5b", prompt="explain how persistent cognitive state", max_tokens=6, prefetch="both", runs=2,
        device="cpu", hot_layers=1, warm_layers=1, cold_cache=True,
    )
    rows = bm.run_matrix(args, str(root), 1.0, 1.0)
    assert [r["prefetch"] for r in rows] == [True, False, False, True]  # counterbalanced
    summary = bm._summarize(rows)
    assert summary["outputs_identical_across_runs"]
    assert summary["prefetch_off"]["bytes_prefetched_median"] == 0
    assert summary["prefetch_off"]["bytes_already_resident_median"] == 0  # prefetch=off never touches anything
    assert all(r["tokens"] == 6 and r["load_sec"] > 0 for r in rows)
    assert all(r["page_cache_files_dropped"] > 0 for r in rows)
    # On a model this small/fast, Accelerate's own synchronous read of a
    # disk-tier layer during token 1 (required for that layer's forward
    # regardless of prefetching) warms it before the background prefetch
    # thread would even be scheduled; needs_warming() then correctly finds
    # every later region already resident and schedules nothing at all --
    # see test_resident_backend.py's end-to-end test. bytes_prefetched and
    # bytes_already_resident legitimately staying at 0 (and the resulting
    # "touched no bytes" note) is the intended optimum here, not a failure;
    # this is a toy/fast model, not the meaningful case for that heuristic.
