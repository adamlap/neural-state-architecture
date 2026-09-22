"""Accelerate-compatible NVMe page-cache prefetch adapter.

Accelerate owns actual parameter materialization. This adapter deliberately does
not mutate Accelerate hooks or model parameters. It warms the OS page cache for
the bytes belonging to a predicted region, so the existing Accelerate hook can
consume the same files with less cold-disk latency.

Accelerate's disk-offload folder holds an ``index.json`` whose entries take one
of these shapes, all of which are understood here:

* ``{"dtype": ..., "shape": [...]}``  -> weights live in ``<offload_dir>/<key>.dat``
* ``{"safetensors_file": path, "weight_name": name, ...}`` -> a tensor inside a safetensors file
* ``"<file>.safetensors"`` or ``{"filename": ...}`` (Hugging Face ``weight_map`` style)

Safetensors headers are parsed directly, so only the byte range of each tensor
is read and no torch/safetensors import is needed.
"""
from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Mapping, Optional, Sequence

from nsa.residency.page_cache import PageCacheProbe

_MAX_HEADER_BYTES = 100 * 1024 * 1024


@dataclass(frozen=True)
class Extent:
    """A contiguous byte range inside a file."""
    path: Path
    offset: int
    length: int


def read_safetensors_header(path: Path) -> dict[str, tuple[int, int]]:
    """Return ``{tensor_name: (absolute_start, absolute_end)}`` for a safetensors file."""
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            return {}
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > _MAX_HEADER_BYTES:
            return {}
        header = json.loads(handle.read(header_len).decode("utf-8"))
    base = 8 + header_len
    extents: dict[str, tuple[int, int]] = {}
    for name, meta in header.items():
        if name == "__metadata__" or not isinstance(meta, dict):
            continue
        offsets = meta.get("data_offsets")
        if isinstance(offsets, list) and len(offsets) == 2:
            extents[name] = (base + int(offsets[0]), base + int(offsets[1]))
    return extents


class AccelerateDiskPrefetcher:
    """Warm offloaded weight pages for selected NSA neural regions."""

    def __init__(
        self,
        offload_dir: str | Path,
        parameter_prefixes: Mapping[str, Sequence[str]],
        chunk_bytes: int = 8 * 1024 * 1024,
        skip_if_resident: bool = True,
        resident_threshold: float = 0.98,
    ) -> None:
        self.offload_dir = Path(offload_dir).expanduser()
        self.parameter_prefixes = {region: tuple(prefixes) for region, prefixes in parameter_prefixes.items()}
        self.chunk_bytes = max(4096, chunk_bytes)
        # Re-reading bytes already in the OS page cache is pure overhead: it
        # cannot make Accelerate's own later read any faster, since the page
        # cache is what makes that read fast in the first place. Gate on real
        # residency (mincore) instead of assuming every predicted region is
        # cold; see docs/ACTIVE_RESIDENCY.md for measurements with this on/off.
        self.skip_if_resident = skip_if_resident
        self.resident_threshold = resident_threshold
        self._extents: dict[str, tuple[Extent, ...]] = {}
        self._headers: dict[Path, dict[str, tuple[int, int]]] = {}
        self._region_cache: dict[str, list[Extent]] = {}
        self._lock = Lock()
        self.bytes_prefetched = 0
        self.bytes_already_resident = 0
        self.read_errors = 0
        # One mmap per file, reused: a fresh mmap()/munmap() per check measured
        # as more expensive than the re-read it exists to avoid (see page_cache.py).
        self._page_cache = PageCacheProbe()
        self._load_index()

    @property
    def available(self) -> bool:
        """True only when at least one parameter maps to readable bytes."""
        return bool(self._extents)

    def _header(self, path: Path) -> dict[str, tuple[int, int]]:
        if path not in self._headers:
            try:
                self._headers[path] = read_safetensors_header(path)
            except (OSError, ValueError):
                self._headers[path] = {}
        return self._headers[path]

    def _safetensors_extent(self, path: Path, name: str) -> Optional[Extent]:
        span = self._header(path).get(name)
        return Extent(path, span[0], span[1] - span[0]) if span and span[1] > span[0] else None

    def _dat_extent(self, key: str) -> Optional[Extent]:
        path = self.offload_dir / f"{key}.dat"
        try:
            size = path.stat().st_size
        except OSError:
            return None
        return Extent(path, 0, size) if size > 0 else None

    def _resolve(self, key: str, meta: object) -> Optional[Extent]:
        if isinstance(meta, str):
            return self._safetensors_extent(self.offload_dir / meta, key)
        if not isinstance(meta, dict):
            return None
        if meta.get("safetensors_file"):
            return self._safetensors_extent(Path(str(meta["safetensors_file"])), str(meta.get("weight_name", key)))
        if meta.get("filename"):
            return self._safetensors_extent(self.offload_dir / str(meta["filename"]), key)
        if "dtype" in meta and "shape" in meta:
            return self._dat_extent(key)
        return None

    def _load_index(self) -> None:
        extents: dict[str, Extent] = {}
        index_path = self.offload_dir / "index.json"
        if index_path.is_file():
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = None
            if isinstance(payload, dict):
                weight_map = payload.get("weight_map", payload)
                if isinstance(weight_map, dict):
                    for key, meta in weight_map.items():
                        extent = self._resolve(str(key), meta)
                        if extent is not None:
                            extents[str(key)] = extent
        else:
            # A non-sharded safetensors checkpoint without an index.
            files = sorted(self.offload_dir.glob("*.safetensors")) if self.offload_dir.is_dir() else []
            if len(files) == 1:
                for name in self._header(files[0]):
                    extent = self._safetensors_extent(files[0], name)
                    if extent is not None:
                        extents[name] = extent
        self._extents = {key: (extent,) for key, extent in extents.items()}

    def _region_extents(self, region_id: str) -> list[Extent]:
        cached = self._region_cache.get(region_id)
        if cached is not None:
            return cached
        prefixes = self.parameter_prefixes.get(region_id, ())
        if not prefixes:
            return []  # an unknown/unmapped region must never warm the whole model
        found = [
            extent
            for key, extents in self._extents.items()
            if any(key.startswith(prefix) for prefix in prefixes)
            for extent in extents
        ]
        self._region_cache[region_id] = sorted(found, key=lambda e: (str(e.path), e.offset))
        return self._region_cache[region_id]

    def covers(self, region_id: str) -> bool:
        """Whether any of the region's parameters live in the offload index."""
        return bool(self._region_extents(region_id))

    def needs_warming(self, region_id: str) -> bool:
        """Cheap synchronous check: is any part of this region not yet resident?

        Meant to run on the caller's own thread *before* deciding whether a
        background prefetch task is worth scheduling at all -- scheduling one
        (lock, ThreadPoolExecutor handoff, telemetry) costs more than this
        check does once mmaps are cached (PageCacheProbe), so for an
        already-warm region (the common case once a model has run for a
        while) this avoids that cost entirely rather than paying it only to
        find nothing to read.
        """
        if not self.skip_if_resident:
            return True
        with self._lock:
            extents = self._region_extents(region_id)
        return any(not self._already_resident(extent) for extent in extents)

    def _warm(self, extent: Extent, buffer: bytearray) -> int:
        """Read an extent through the page cache, discarding the data."""
        read_total = 0
        with open(extent.path, "rb", buffering=0) as handle:
            fadvise = getattr(os, "posix_fadvise", None)
            if fadvise is not None:
                try:
                    fadvise(handle.fileno(), extent.offset, extent.length, os.POSIX_FADV_WILLNEED)
                except OSError:
                    pass
            handle.seek(extent.offset)
            view = memoryview(buffer)
            remaining = extent.length
            while remaining > 0:
                got = handle.readinto(view[: min(len(buffer), remaining)])
                if not got:
                    break
                read_total += got
                remaining -= got
        return read_total

    def _already_resident(self, extent: Extent) -> bool:
        if not self.skip_if_resident:
            return False
        fraction = self._page_cache.resident_fraction(str(extent.path), extent.offset, extent.length)
        return fraction is not None and fraction >= self.resident_threshold

    def prefetch(self, region_id: str) -> int:
        """Warm the pages backing ``region_id`` and return the bytes actually read.

        Returns 0 when nothing could be warmed (unknown region, unmapped
        parameters, missing files) *or* when everything was already resident
        in the OS page cache (skip_if_resident, the default): re-reading
        already-cached bytes cannot make Accelerate's later read any faster
        and only adds disk/CPU contention. No model parameter is replaced
        and no NSA state is mutated.
        """
        with self._lock:
            extents = self._region_extents(region_id)
        buffer = bytearray(self.chunk_bytes)
        total = 0
        skipped = 0
        for extent in extents:
            if self._already_resident(extent):
                skipped += extent.length
                continue
            try:
                total += self._warm(extent, buffer)
            except OSError:
                with self._lock:
                    self.read_errors += 1
        with self._lock:
            self.bytes_prefetched += total
            self.bytes_already_resident += skipped
        return total

    def __call__(self, region_id: str, tier: object = None) -> int:
        return self.prefetch(region_id)

    def close(self) -> None:
        """Release cached mmaps. Safe to call more than once."""
        self._page_cache.close()


__all__ = ["AccelerateDiskPrefetcher", "Extent", "read_safetensors_header"]
