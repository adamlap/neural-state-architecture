"""Tests for the mincore-based page-cache residency probe.

These use the real filesystem and real OS page cache (via posix_fadvise),
not mocks -- the whole point of this module is to report ground truth.
"""
import os

import pytest

from nsa.residency.page_cache import PageCacheProbe, resident_fraction

pytestmark = pytest.mark.skipif(not hasattr(os, "posix_fadvise"), reason="posix_fadvise is POSIX-only")

SIZE = 4 * 1024 * 1024  # 4 MiB: several pages, small enough to run fast


def _evict(path, offset=0, length=SIZE):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


@pytest.fixture
def cold_file(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(os.urandom(SIZE))
    _evict(str(path))
    return str(path)


def test_evicted_file_reads_as_cold(cold_file):
    fraction = resident_fraction(cold_file, 0, SIZE)
    assert fraction is not None
    assert fraction < 0.05


def test_fully_read_file_reads_as_warm(cold_file):
    with open(cold_file, "rb") as handle:
        handle.read()
    assert resident_fraction(cold_file, 0, SIZE) > 0.95


def test_resolution_is_per_page_not_per_file(cold_file):
    """Reading only the back half must not mark the front half warm."""
    with open(cold_file, "rb") as handle:
        handle.seek(SIZE // 2)
        handle.read(SIZE // 2)
    assert resident_fraction(cold_file, 0, SIZE // 4) < 0.05
    assert resident_fraction(cold_file, 3 * SIZE // 4, SIZE // 4) > 0.95


def test_re_evicting_returns_to_cold(cold_file):
    with open(cold_file, "rb") as handle:
        handle.read()
    assert resident_fraction(cold_file, 0, SIZE) > 0.95
    _evict(cold_file)
    assert resident_fraction(cold_file, 0, SIZE) < 0.05


def test_missing_file_returns_none():
    assert resident_fraction("/nonexistent/path/does/not/exist", 0, 4096) is None


def test_non_positive_length_returns_none(cold_file):
    assert resident_fraction(cold_file, 0, 0) is None
    assert resident_fraction(cold_file, 0, -1) is None


def test_offset_beyond_end_of_file_does_not_raise(cold_file):
    # mmap of a range beyond EOF either fails cleanly or maps a truncated
    # region; either way this must return None, never raise.
    resident_fraction(cold_file, SIZE * 10, 4096)


def test_unavailable_libc_is_reported_as_none(monkeypatch):
    import nsa.residency.page_cache as mod
    monkeypatch.setattr(mod, "_get_libc", lambda: None)
    assert resident_fraction("/etc/hostname", 0, 10) is None


class TestPageCacheProbe:
    """PageCacheProbe caches one mmap per file instead of remapping every
    call -- the earlier remap-every-time design measured as more expensive
    than the read it existed to avoid."""

    def test_reused_probe_agrees_with_one_shot_function(self, cold_file):
        probe = PageCacheProbe()
        try:
            assert probe.resident_fraction(cold_file, 0, SIZE) < 0.05
            with open(cold_file, "rb") as handle:
                handle.read()
            assert probe.resident_fraction(cold_file, 0, SIZE) > 0.95
        finally:
            probe.close()

    def test_probe_maps_each_file_only_once(self, cold_file):
        probe = PageCacheProbe()
        try:
            probe.resident_fraction(cold_file, 0, 4096)
            probe.resident_fraction(cold_file, 4096, 4096)
            assert len(probe._files) == 1
        finally:
            probe.close()

    def test_a_file_that_does_not_exist_yet_is_retried_not_cached_as_absent(self, tmp_path):
        path = str(tmp_path / "appears-later.bin")
        probe = PageCacheProbe()
        try:
            assert probe.resident_fraction(path, 0, 4096) is None
            with open(path, "wb") as handle:
                handle.write(os.urandom(4096))
            assert probe.resident_fraction(path, 0, 4096) is not None
        finally:
            probe.close()

    def test_close_is_idempotent_and_releases_mappings(self, cold_file):
        probe = PageCacheProbe()
        probe.resident_fraction(cold_file, 0, 4096)
        probe.close()
        probe.close()  # must not raise
        assert probe._files == {}

    def test_probes_on_different_files_are_independent(self, tmp_path):
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(os.urandom(SIZE))
        b.write_bytes(os.urandom(SIZE))
        _evict(str(a))
        _evict(str(b))
        probe = PageCacheProbe()
        try:
            with open(a, "rb") as handle:
                handle.read()
            assert probe.resident_fraction(str(a), 0, SIZE) > 0.95
            assert probe.resident_fraction(str(b), 0, SIZE) < 0.05
        finally:
            probe.close()
