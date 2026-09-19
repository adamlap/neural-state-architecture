"""Accelerate-compatible NVMe page-cache prefetch adapter.

Accelerate owns actual parameter materialization. This adapter deliberately does
not mutate Accelerate hooks or model parameters. It warms the OS/NVMe page
cache for the tensors belonging to a predicted region, so the existing
Accelerate hook can consume the same files with less cold-disk latency.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Mapping, Sequence


class AccelerateDiskPrefetcher:
    """Warm offloaded safetensor pages for selected NSA neural regions."""

    def __init__(self, offload_dir: str | Path, parameter_prefixes: Mapping[str, Sequence[str]]) -> None:
        self.offload_dir = Path(offload_dir).expanduser()
        self.parameter_prefixes = {
            region: tuple(prefixes) for region, prefixes in parameter_prefixes.items()
        }
        self._index: dict[str, object] = {}
        self._file_keys: dict[str, tuple[str, ...]] = {}
        self._lock = Lock()
        self._load_index()

    @property
    def available(self) -> bool:
        return bool(self._index)

    def _load_index(self) -> None:
        index_path = self.offload_dir / "index.json"
        if not index_path.is_file():
            # Accelerate can offload a non-sharded checkpoint without an index.
            # Discover a single safetensors file so prefetch still works.
            files = sorted(self.offload_dir.glob("*.safetensors"))
            if len(files) == 1:
                try:
                    from safetensors import safe_open
                    with safe_open(str(files[0]), framework="pt", device="cpu") as handle:
                        self._index = {key: files[0].name for key in handle.keys()}
                        self._file_keys = {files[0].name: tuple(handle.keys())}
                except ImportError:
                    pass
            return
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", payload)
        if not isinstance(weight_map, dict):
            return
        self._index = {str(k): v for k, v in weight_map.items() if isinstance(v, (str, dict))}
        grouped: dict[str, list[str]] = {}
        for key, meta in self._index.items():
            filename = str(meta.get("filename", "")) if isinstance(meta, dict) else str(meta) if isinstance(meta, dict) else str(meta)
            if filename:
                grouped.setdefault(filename, []).append(key)
        self._file_keys = {name: tuple(keys) for name, keys in grouped.items()}

    def _keys_for_region(self, region_id: str) -> dict[str, tuple[str, ...]]:
        prefixes = self.parameter_prefixes.get(region_id, ())
        grouped: dict[str, list[str]] = {}
        for key, meta in self._index.items():
            if prefixes and not any(key.startswith(prefix) for prefix in prefixes):
                continue
            filename = str(meta.get("filename", ""))
            if filename:
                grouped.setdefault(filename, []).append(key)
        return {name: tuple(keys) for name, keys in grouped.items()}

    def prefetch(self, region_id: str) -> int:
        """Read predicted tensors once and immediately release them.

        Returns the number of tensors touched. No model parameter is replaced
        and no NSA state is mutated by this method.
        """
        with self._lock:
            groups = self._keys_for_region(region_id)
        if not groups:
            return 0
        try:
            from safetensors import safe_open
        except ImportError:
            return 0

        touched = 0
        for filename, keys in groups.items():
            path = self.offload_dir / filename
            if not path.is_file():
                continue
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    # Force the mapped pages to be read before releasing the
                    # temporary tensor. Accelerate will subsequently read the
                    # same backing file during its normal hook execution.
                    if tensor.numel():
                        tensor.reshape(-1)[:: max(1, tensor.numel() // 4096)].sum().item()
                    touched += 1
        return touched

    def __call__(self, region_id: str, tier: object) -> None:
        # The target tier is where the prediction wants the region next;
        # the adapter warms the disk-backed source before Accelerate loads it.
        self.prefetch(region_id)


__all__ = ["AccelerateDiskPrefetcher"]
