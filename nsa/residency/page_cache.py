"""Query real OS page-cache residency via mincore(2), Linux/POSIX only.

This exists to answer one question honestly instead of guessing: "is this
byte range already in the page cache, so warming it again would just be
wasted I/O?" A naive prefetcher that re-reads a region on every decoder
step regardless of whether it's already resident can cost more than it
saves once the model is small enough (or RAM is large enough) that the
region stays warm between reads -- see docs/ACTIVE_RESIDENCY.md.

Measured cost, not assumed: opening a fresh mmap and closing it on every
single check (the first version of this module) cost *more* than the disk
read it was trying to avoid, on files small enough to already be cheap to
re-read -- mmap/munmap is a real syscall, and doing it once per tensor per
decoder step adds up. PageCacheProbe instead keeps one mmap per file open
for its own lifetime (closed explicitly, or on GC) and only pays mincore's
cost on repeat checks.
"""
from __future__ import annotations

import ctypes
import mmap
import os
from pathlib import Path
from typing import Optional

_libc: object = None
_unavailable = False


def _get_libc():
    global _libc, _unavailable
    if _unavailable:
        return None
    if _libc is None:
        try:
            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            if not hasattr(libc, "mincore"):
                raise OSError("mincore not exported by libc")
            _libc = libc
        except OSError:
            _unavailable = True
            return None
    return _libc


class _MappedFile:
    """One cached read-only mmap of a file, sized to the file at open time."""

    __slots__ = ("fd", "mapping", "size")

    def __init__(self, path: str) -> None:
        self.fd = os.open(path, os.O_RDONLY)
        try:
            self.size = os.fstat(self.fd).st_size
            # ACCESS_COPY (private, copy-on-write) works read-only and gives
            # ctypes a "writable enough" buffer view without ever writing.
            self.mapping = mmap.mmap(self.fd, max(self.size, 1), access=mmap.ACCESS_COPY) if self.size else None
        except (ValueError, OSError):
            os.close(self.fd)
            raise

    def close(self) -> None:
        if self.mapping is not None:
            self.mapping.close()
        os.close(self.fd)


class PageCacheProbe:
    """Reusable mincore-based residency checker; caches one mmap per file.

    Not thread-safe on its own -- callers that share one instance across
    threads must serialise access (AccelerateDiskPrefetcher already holds a
    lock around its own use of this).
    """

    def __init__(self) -> None:
        self._files: dict[str, Optional[_MappedFile]] = {}

    def _mapped(self, path: str) -> Optional[_MappedFile]:
        cached = self._files.get(path)
        if cached is not None:
            return cached
        try:
            mapped = _MappedFile(path)
        except OSError:
            return None  # not cached: a file that doesn't exist yet may still appear
        self._files[path] = mapped
        return mapped

    def resident_fraction(self, path: str, offset: int, length: int) -> Optional[float]:
        """Fraction (0.0-1.0) of ``[offset, offset+length)`` in ``path`` already
        resident in the OS page cache; ``None`` when this can't be determined
        (non-Linux/no mincore, the file can't be opened/mapped, or an offset
        past EOF) -- callers must treat that as "unknown", not "cold".

        The file is assumed not to grow after first being probed (true for a
        completed Accelerate/safetensors offload write); a file that changes
        size later should use a fresh PageCacheProbe.
        """
        if length <= 0:
            return None
        libc = _get_libc()
        if libc is None:
            return None
        mapped = self._mapped(path)
        if mapped is None or mapped.mapping is None:
            return None
        page_size = mmap.PAGESIZE
        aligned_offset = (offset // page_size) * page_size
        span = min(offset + length, mapped.size) - aligned_offset
        if span <= 0:
            return None
        num_pages = -(-span // page_size)  # ceil division
        try:
            view = (ctypes.c_char * span).from_buffer(mapped.mapping, aligned_offset)
            vector = (ctypes.c_uint8 * num_pages)()
            address = ctypes.addressof(view)
            rc = libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(span), vector)
            del view  # release before any close()/resize touches the mapping
            if rc != 0:
                return None
            return sum(1 for byte in vector if byte & 1) / num_pages
        except (ValueError, OSError):
            return None

    def close(self) -> None:
        for mapped in self._files.values():
            if mapped is not None:
                mapped.close()
        self._files.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def resident_fraction(path: str, offset: int, length: int) -> Optional[float]:
    """One-shot residency check (opens and closes its own mapping).

    Prefer PageCacheProbe for repeated checks against the same files (e.g.
    once per decoder step) -- see the module docstring for why.
    """
    probe = PageCacheProbe()
    try:
        return probe.resident_fraction(path, offset, length)
    finally:
        probe.close()


__all__ = ["PageCacheProbe", "resident_fraction"]
