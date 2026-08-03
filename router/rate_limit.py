"""
rate_limit.py — inbound request rate limiting for the HTTP proxy.

Every budget control elsewhere in Flux bounds SPEND (BudgetTracker,
DailyBudgetTracker, run budgets, TENANT_DAILY_CAP_USD). None of them bound
REQUEST RATE, and the two are not substitutes: spend caps are opt-in
(TENANT_DAILY_CAP_USD defaults to None and only applies in FLUX_SERVER_TOKENS
mode), and even when enabled they only stop a caller *after* the money is
gone. A runaway agent loop can hold a valid token and issue requests as fast
as the event loop will take them, and every one is forwarded upstream.

This module is the missing ceiling: a per-key token bucket, checked before a
request is routed or dispatched.

  RATE_LIMIT_RPM     — sustained requests per minute per key (0 disables)
  RATE_LIMIT_BURST   — bucket capacity, i.e. how big a spike is absorbed

WHAT THIS IS NOT: a distributed limiter. Buckets are process-local, exactly
like run_budget.InMemoryRunStore — with FLUX_SERVER_WORKERS > 1 each worker
enforces its own share, so the effective global limit is roughly
RATE_LIMIT_RPM x workers. server.py warns about this at startup on the same
terms as the run store. Put a shared limiter at the ingress (nginx, Envoy,
API gateway) if you need an exact global bound.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import structlog

from .config import RATE_LIMIT_BURST, RATE_LIMIT_MAX_TRACKED_KEYS, RATE_LIMIT_RPM

log = structlog.get_logger(__name__)


@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class RateLimiter:
    """Token bucket per key, with a bounded number of tracked keys.

    The key-count cap matters as much as the rate cap. Keys come from
    caller-influenced values (a client IP when auth is off), so an unbounded
    dict is itself a memory-exhaustion vector — the same reasoning behind
    ATTRIBUTION_METRICS_MAX_LABEL_COMBOS in attribution.py. Buckets are held
    in an LRU and the least-recently-used one is evicted at the cap.

    Eviction is deliberately a *reset*, not a bypass: an evicted key that
    returns starts from a full bucket. Under a key-flooding attack that makes
    the limiter approximate, but the alternative — denying unknown keys once
    the table is full — would let an attacker lock out every legitimate
    caller, which is a worse failure than a leaky limit.
    """

    def __init__(
        self,
        rpm: int = RATE_LIMIT_RPM,
        burst: int = RATE_LIMIT_BURST,
        max_keys: int = RATE_LIMIT_MAX_TRACKED_KEYS,
    ) -> None:
        self.rpm = rpm
        self.burst = max(1, burst)
        self.max_keys = max(1, max_keys)
        self._refill_per_sec = rpm / 60.0 if rpm > 0 else 0.0
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    def check(self, key: str) -> float | None:
        """Consume one token for `key`.

        Returns None when the request is allowed, or the number of seconds to
        wait before a token is available when it is not — the caller turns
        that into a Retry-After header.
        """
        if not self.enabled:
            return None

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                if len(self._buckets) >= self.max_keys:
                    evicted_key, _ = self._buckets.popitem(last=False)
                    log.warning(
                        "rate_limit_key_table_full",
                        limit=self.max_keys,
                        evicted=evicted_key,
                        msg="rate-limit key table at capacity; evicting the "
                        "least-recently-used bucket. A flood of distinct keys "
                        "makes enforcement approximate — see RateLimiter.",
                    )
                bucket = _Bucket(tokens=float(self.burst), last_refill=now)
                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)
                elapsed = now - bucket.last_refill
                if elapsed > 0:
                    bucket.tokens = min(
                        float(self.burst), bucket.tokens + elapsed * self._refill_per_sec
                    )
                    bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return None

            # Not enough for a whole token — report when the next one lands.
            deficit = 1.0 - bucket.tokens
            return deficit / self._refill_per_sec

    def reset(self) -> None:
        """Drop all buckets. For tests and for operators reloading config."""
        with self._lock:
            self._buckets.clear()
