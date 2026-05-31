"""
cache.py — tiny on-disk JSON cache for the expensive live-mode artifacts.

Completions and judge verdicts cost real money in --live mode, so we key them by
content and persist them. Re-running an eval then reuses prior answers instead of
paying again. In mock mode caching is unnecessary (completions are deterministic
and free) so the runner leaves it disabled.

The cache is a single JSON file; the eval runs sequentially, so no locking is
needed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def make_key(*parts: str) -> str:
    """Stable short key from arbitrary string parts."""
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32]


class DiskCache:
    def __init__(self, path: str | None) -> None:
        self.enabled = path is not None
        self._path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        if self.enabled and self._path is not None and self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> dict | None:
        return self._data.get(key)

    def set(self, key: str, value: dict) -> None:
        if not self.enabled or self._path is None:
            return
        self._data[key] = value
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data), encoding="utf-8")
