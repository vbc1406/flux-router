"""
File: router/prompt_cache.py

Purpose:
Cache-aware routing. Provider-side prompt caching (a warm KV-cache
for a shared prefix — typically the system prompt) commonly cuts cost more
than model routing does, but it is provider-specific and prefix-specific:
routing a request to a nominally "cheaper" model can destroy a warm cache on
the incumbent provider and increase the total bill. This module tracks which
provider currently holds a warm prefix for a given conversation/run so
routing_engine.py can protect it.

Main Classes:
  PromptCacheTracker — records/looks up (key -> provider, prefix hash, TTL)

Config Dependencies (all in config.py):
  CACHE_PREFIX_MIN_TOKENS   — below this prefix size, caching is not modeled
  CACHE_DEFAULT_TTL_SECONDS — fallback TTL when a model doesn't specify one
  CACHE_TRACKER_MAX_ENTRIES — LRU cap on tracked keys

Not to be confused with router/cache.py (ResponseCache), which caches whole
LLM *responses* keyed by full-prompt fingerprint. This module tracks
provider-side *prefix* caching state — it never stores prompt or response
content, only a one-way hash of the prefix and which provider last used it.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import structlog

from .config import CACHE_DEFAULT_TTL_SECONDS, CACHE_TRACKER_MAX_ENTRIES

log = structlog.get_logger(__name__)


def hash_prefix(text: str) -> str:
    """One-way hash of a cache-relevant prefix (e.g. the system prompt).

    Used only to detect "is this the same prefix as last time", never to
    recover or log the prefix content itself.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass
class _CacheEntry:
    provider: str
    prefix_hash: str
    prefix_tokens: int
    expires_at: float


class PromptCacheTracker:
    """
    Thread-safe, process-local, LRU-bounded map of
    (conversation/run key) -> which provider currently holds a warm prefix.

    🔧 EXTENSION POINT: swap the internal dict for a Redis client (keyed the
    same way) to share cache-stickiness state across multiple server
    instances — a single-process assumption is fine for one uvicorn worker
    but wrong for a fleet, where "warm on provider X" is a per-worker fact
    only if this state is local to each worker.
    """

    def __init__(self, max_entries: int = CACHE_TRACKER_MAX_ENTRIES) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get_warm_provider(self, key: str, prefix_hash: str) -> str | None:
        """
        Return the provider currently holding a warm cache for this exact
        prefix, or None if unknown, expired, or the prefix has changed.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at or entry.prefix_hash != prefix_hash:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return entry.provider

    def record(
        self,
        key: str,
        provider: str,
        prefix_hash: str,
        prefix_tokens: int,
        ttl_seconds: float = CACHE_DEFAULT_TTL_SECONDS,
    ) -> None:
        """Record that `provider` now holds the warm cache for this key/prefix."""
        with self._lock:
            if key not in self._store and len(self._store) >= self._max_entries:
                evicted_key, _ = self._store.popitem(last=False)
                log.info("prompt_cache_evicted_lru", limit=self._max_entries, evicted=evicted_key)
            self._store[key] = _CacheEntry(
                provider=provider,
                prefix_hash=prefix_hash,
                prefix_tokens=prefix_tokens,
                expires_at=time.monotonic() + ttl_seconds,
            )
            self._store.move_to_end(key)
