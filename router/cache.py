"""
File: router/cache.py

Purpose:
In-memory LRU response cache with prompt fingerprinting.  Fingerprinting
normalises a request to a stable SHA-256 hash so that semantically identical
prompts (modulo whitespace, filler words, dates) hit the same cache entry.

Main Classes:
  ResponseCache   — LRU cache keyed by prompt fingerprint

Key Functions:
  fingerprint(prompt, system_prompt, history, temperature) — produce a cache key
  is_cache_eligible(task_type, temperature)               — decide if a request can be cached

Config Dependencies (all in config.py):
  CACHE_TTL_SECONDS     — entry TTL
  CACHE_MAX_ENTRIES     — max entries before LRU eviction
  CACHE_ENABLED         — master switch
  CACHEABLE_TASK_TYPES  — task types eligible for caching
  NON_CACHEABLE_TASK_TYPES — task types that must never be cached

🔧 EXTENSION POINT: to add cache normalisation for a new text pattern (e.g., user names),
  add a compiled regex and substitution call in _normalize().  The fingerprint function
  uses _normalize() on all text components.

Things NOT to change without discussion:
  - Temperature is always included in the fingerprint hash — removing it would cause
    creative (high-temp) requests to return cached deterministic responses.
  - Only the last 2 history turns are included in the fingerprint — including the full
    history would cause too many cache misses on long conversations.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict

import structlog

from .config import (
    CACHE_ENABLED,
    CACHE_MAX_ENTRIES,
    CACHE_TTL_SECONDS,
    CACHEABLE_TASK_TYPES,
    NON_CACHEABLE_TASK_TYPES,
)
from .schemas import CachedResponse, ModelOption

log = structlog.get_logger(__name__)

# Regex patterns compiled once at import time for < 1 ms fingerprinting.
_WHITESPACE_RE   = re.compile(r"\s+")
_TIME_VARIANT_RE = re.compile(
    r"\b(today|now|currently|yesterday|tomorrow|tonight|this\s+\w+)\b",
    re.IGNORECASE,
)
_DATE_ISO_RE     = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")
_DATE_HUMAN_RE   = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\s+\d{1,2},?\s+\d{4}\b",
    re.IGNORECASE,
)
_TIMESTAMP_RE    = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE)
_FILLER_RE       = re.compile(
    r"\b(please|can you|could you|i want you to|i need you to|hey|hi there|"
    r"kindly|would you|will you)\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """
    Strip/normalise a text fragment so that trivial rewording does not produce
    different cache keys.  Order matters — whitespace collapse comes last.
    """
    text = text.lower()
    text = _TIME_VARIANT_RE.sub("", text)
    text = _DATE_ISO_RE.sub("", text)
    text = _DATE_HUMAN_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    text = _FILLER_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    # Remove spaces inserted before punctuation by earlier substitutions.
    text = re.sub(r" +([?.!,;:])", r"\1", text)
    return text


def fingerprint(
    prompt: str,
    system_prompt: str | None,
    history: list[dict],
    temperature: float | None,
) -> str:
    """
    Produce a stable SHA-256 hex digest for cache lookup.

    Includes temperature so that creative (high-temp) calls cannot return a
    cached deterministic (low-temp) response — they are fundamentally
    different requests even if the prompt text is identical.

    Only the last two history turns are included; including the full history
    would cause too many cache misses on long conversations.
    """
    parts: list[str] = [_normalize(prompt)]

    if system_prompt:
        parts.append("SYS:" + _normalize(system_prompt))

    for msg in history[-2:]:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}:{_normalize(content)}")
        else:
            # Multimodal content (list of dicts) — use role + type tags only
            parts.append(f"{role}:multimodal")

    if temperature is None:
        parts.append("temp:default")
    else:
        parts.append(f"temp:t{temperature:.2f}")

    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def is_cache_eligible(task_type: str, temperature: float | None) -> bool:
    """
    A request is cache-eligible ONLY when all four conditions hold:
      1. task_type is in CACHEABLE_TASK_TYPES
      2. task_type is NOT in NON_CACHEABLE_TASK_TYPES
      3. temperature is None or ≤ 0.3  (high temp → user wants variety)
      4. Implicitly: vision tasks are already in NON_CACHEABLE_TASK_TYPES

    This is evaluated by the classifier and stored in TaskAnalysis.cache_eligible
    so the routing engine does not need to re-evaluate it.
    """
    if task_type in NON_CACHEABLE_TASK_TYPES:
        return False
    if task_type not in CACHEABLE_TASK_TYPES:
        return False
    if temperature is not None and temperature > 0.3:
        return False
    return True


class _CacheEntry:
    """Internal wrapper that adds a TTL expiry timestamp to a CachedResponse."""

    __slots__ = ("response", "expires_at")

    def __init__(self, response: CachedResponse, ttl: int) -> None:
        self.response   = response
        self.expires_at = time.monotonic() + ttl


class ResponseCache:
    """
    In-memory LRU cache backed by an OrderedDict.

    Key  → prompt fingerprint (SHA-256 hex)
    Value → CachedResponse with the original response text and metadata.

    Thread-safe.  All hit/miss counters are updated atomically.
    """

    def __init__(
        self,
        max_entries: int = CACHE_MAX_ENTRIES,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        enabled: bool    = CACHE_ENABLED,
    ) -> None:
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._max   = max_entries
        self._ttl   = ttl_seconds
        self._enabled = enabled
        self._lock  = threading.Lock()
        self._hits  = 0
        self._misses = 0

    # ── Public API ──────────────────────────────────────────────────────────

    def get(self, fp: str) -> CachedResponse | None:
        """
        Return a cached response for the given fingerprint, or None on miss/expiry.
        Moves the entry to the end (most-recently-used) on a hit.
        """
        if not self._enabled:
            return None
        with self._lock:
            entry = self._store.get(fp)
            if entry is None:
                self._misses += 1
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[fp]
                self._misses += 1
                return None
            # LRU: promote to end
            self._store.move_to_end(fp)
            self._hits += 1
            return entry.response

    def set(
        self,
        fp: str,
        response_text: str,
        model_used: ModelOption,
        original_cost: float,
        quality_score: float | None = None,
        ttl: int | None = None,
    ) -> None:
        """
        Store a response.  Evicts the least-recently-used entry when full.
        TTL defaults to the instance-level setting.
        """
        if not self._enabled:
            return
        effective_ttl = ttl if ttl is not None else self._ttl
        from .schemas import CachedResponse  # local import to avoid circular at module level
        cached = CachedResponse(
            fingerprint=fp,
            response_text=response_text,
            model_used=model_used,
            original_cost=original_cost,
            quality_score=quality_score,
        )
        with self._lock:
            if fp in self._store:
                self._store.move_to_end(fp)
            self._store[fp] = _CacheEntry(cached, effective_ttl)
            if len(self._store) > self._max:
                evicted_key, _ = self._store.popitem(last=False)
                log.debug("cache_lru_eviction", key=evicted_key[:16])

    def invalidate(self, fp: str) -> bool:
        """Remove a specific entry. Returns True if it existed."""
        with self._lock:
            return self._store.pop(fp, None) is not None

    def stats(self) -> dict[str, float | int]:
        """Return current hit rate and counters (useful for the analytics layer)."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hit_rate":      self._hits / total if total > 0 else 0.0,
                "total_hits":    self._hits,
                "total_misses":  self._misses,
                "entries_count": len(self._store),
            }
