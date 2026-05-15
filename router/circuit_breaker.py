"""
circuit_breaker.py — Per-provider circuit breaker (Fix 4).

Prevents cascading failures when a provider is down: after
CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive failures the circuit opens and
that provider's models are excluded from routing for CIRCUIT_BREAKER_RECOVERY_TIMEOUT
seconds.  A half-open probe is allowed once the timeout expires; if it succeeds
the circuit closes, otherwise it reopens.

Thread-safe — safe to share across concurrent route() calls.
"""

from __future__ import annotations

import threading
import time

import structlog

log = structlog.get_logger(__name__)

_CLOSED = "closed"
_OPEN = "open"
_HALFOPEN = "half-open"


class CircuitBreaker:
    """
    Simple three-state circuit breaker keyed by provider name.

    States:
      closed    → provider appears healthy; all requests allowed.
      open      → provider is failing; all requests blocked.
      half-open → recovery timeout has elapsed; one probe request allowed.
                  Success → closed.  Failure → back to open.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        # provider → {"failures": int, "state": str, "opened_at": float|None,
        #              "probe_in_flight": bool}
        self._providers: dict[str, dict] = {}
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def is_available(self, provider: str) -> bool:
        """
        Return True when the provider should be used (circuit closed or half-open).
        Exactly one half-open probe is allowed per recovery cycle.
        """
        with self._lock:
            entry = self._providers.get(provider)
            if entry is None or entry["state"] == _CLOSED:
                return True

            if entry["state"] == _OPEN:
                elapsed = time.monotonic() - entry["opened_at"]
                # Transition to half-open; allow exactly one probe.
                if elapsed >= self.recovery_timeout and not entry["probe_in_flight"]:
                    entry["state"] = _HALFOPEN
                    entry["probe_in_flight"] = True
                    log.info(
                        "circuit_half_open",
                        provider=provider,
                        elapsed_s=round(elapsed, 1),
                    )
                    return True
                return False  # still open, no probe yet

            if entry["state"] == _HALFOPEN:
                # Only one probe in flight at a time.
                return not entry["probe_in_flight"]

            return True

    def record_success(self, provider: str) -> None:
        """A call to this provider succeeded — close the circuit."""
        with self._lock:
            entry = self._providers.get(provider)
            if entry and entry["state"] != _CLOSED:
                log.info("circuit_closed", provider=provider)
            self._providers[provider] = {
                "failures": 0,
                "state": _CLOSED,
                "opened_at": None,
                "probe_in_flight": False,
            }

    def record_failure(self, provider: str) -> None:
        """A call to this provider failed — increment failure count and maybe open."""
        with self._lock:
            entry = self._providers.get(
                provider,
                {
                    "failures": 0,
                    "state": _CLOSED,
                    "opened_at": None,
                    "probe_in_flight": False,
                },
            )

            # Probe failed → reopen immediately.
            if entry["state"] == _HALFOPEN:
                entry["state"] = _OPEN
                entry["opened_at"] = time.monotonic()
                entry["probe_in_flight"] = False
                entry["failures"] += 1
                self._providers[provider] = entry
                log.warning(
                    "circuit_reopened_after_probe_failure",
                    provider=provider,
                    recovery_timeout_s=self.recovery_timeout,
                )
                return

            entry["failures"] += 1
            if entry["failures"] >= self.failure_threshold and entry["state"] == _CLOSED:
                entry["state"] = _OPEN
                entry["opened_at"] = time.monotonic()
                log.warning(
                    "circuit_opened",
                    provider=provider,
                    failures=entry["failures"],
                    recovery_timeout_s=self.recovery_timeout,
                    message=(
                        f"Circuit open for {provider} — "
                        f"{entry['failures']} consecutive failures. "
                        f"Skipping for {self.recovery_timeout}s."
                    ),
                )
            self._providers[provider] = entry

    def get_state(self, provider: str) -> str:
        """Return the current state string for a provider (for testing / monitoring)."""
        with self._lock:
            entry = self._providers.get(provider)
            return entry["state"] if entry else _CLOSED

    def get_failure_count(self, provider: str) -> int:
        """Return the current consecutive failure count for a provider."""
        with self._lock:
            entry = self._providers.get(provider)
            return entry["failures"] if entry else 0
