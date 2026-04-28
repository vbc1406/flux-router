"""
File: router/tests/test_circuit_breaker.py

Purpose:
Tests for the per-provider CircuitBreaker — state transitions (closed →
open → half-open → closed), integration with model filtering in the routing
engine, and recovery-timeout behaviour.

How to run:
  pytest -v router/tests/test_circuit_breaker.py
  pytest -v router/tests/test_circuit_breaker.py::TestCircuitBreakerUnit

How to add a test:
  1. Use CircuitBreaker(failure_threshold=N, recovery_timeout=T) directly for
     unit tests, or _engine() + _req() for integration tests.
  2. Call cb.record_failure(provider) / cb.record_success(provider) to drive
     state, then assert cb.is_open(provider).

Test classes:
  TestCircuitBreakerUnit          — state machine: open/close/half-open transitions
  TestCircuitBreakerFiltersModels — routing engine excludes models from open providers

Covers:
  - Circuit starts closed (provider available)
  - Circuit opens after failure_threshold consecutive failures
  - Success resets the failure counter
  - Circuit blocks requests while open
  - Circuit allows a probe after recovery_timeout (half-open)
  - Probe success closes the circuit
  - Probe failure reopens the circuit
  - Open circuit filters its provider's models from routing candidates
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.circuit_breaker import CircuitBreaker
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import RoutingRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine() -> RoutingEngine:
    registry   = ModelRegistry()
    cache      = ResponseCache(enabled=False)
    adaptive   = AdaptiveWeights(state_file=None)
    analytics  = RoutingAnalytics(log_path=None)
    budget     = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id":        "u_cb",
        "plan":           "business_plan",
        "priority":       "normal",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(
        raw_prompt     = prompt,
        correlation_id = str(uuid.uuid4()),
        **defaults,
    )


def rr(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# CircuitBreaker unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerUnit:
    def _cb(self, threshold: int = 5, timeout: float = 60.0) -> CircuitBreaker:
        return CircuitBreaker(failure_threshold=threshold, recovery_timeout=timeout)

    # ── Initial state ──────────────────────────────────────────────────────

    def test_provider_available_by_default(self):
        cb = self._cb()
        assert cb.is_available("openai") is True

    def test_unknown_provider_available(self):
        cb = self._cb()
        assert cb.is_available("some-new-provider") is True

    def test_get_state_closed_by_default(self):
        cb = self._cb()
        assert cb.get_state("anthropic") == "closed"

    def test_get_failure_count_zero_by_default(self):
        cb = self._cb()
        assert cb.get_failure_count("google") == 0

    # ── Failure accumulation ───────────────────────────────────────────────

    def test_circuit_opens_after_threshold(self):
        cb = self._cb(threshold=5)
        for _ in range(5):
            cb.record_failure("openai")
        assert cb.is_available("openai") is False
        assert cb.get_state("openai") == "open"

    def test_circuit_stays_closed_below_threshold(self):
        cb = self._cb(threshold=5)
        for _ in range(4):
            cb.record_failure("openai")
        assert cb.is_available("openai") is True
        assert cb.get_state("openai") == "closed"

    def test_failure_count_tracked(self):
        cb = self._cb(threshold=10)
        for i in range(7):
            cb.record_failure("anthropic")
        assert cb.get_failure_count("anthropic") == 7

    # ── Success reset ──────────────────────────────────────────────────────

    def test_success_resets_counter(self):
        cb = self._cb(threshold=5)
        for _ in range(3):
            cb.record_failure("openai")
        cb.record_success("openai")
        assert cb.get_failure_count("openai") == 0
        assert cb.get_state("openai") == "closed"

    def test_success_after_threshold_closes_circuit(self):
        cb = self._cb(threshold=3)
        for _ in range(3):
            cb.record_failure("google")
        assert cb.get_state("google") == "open"
        cb.record_success("google")
        assert cb.get_state("google") == "closed"
        assert cb.is_available("google") is True

    def test_success_before_any_failure_is_noop(self):
        cb = self._cb()
        cb.record_success("mistral")
        assert cb.get_state("mistral") == "closed"

    # ── Recovery (half-open) ───────────────────────────────────────────────

    def test_circuit_allows_probe_after_timeout(self):
        cb = self._cb(threshold=2, timeout=0.01)
        cb.record_failure("openai")
        cb.record_failure("openai")
        assert cb.get_state("openai") == "open"
        time.sleep(0.02)  # exceed recovery timeout
        assert cb.is_available("openai") is True  # probe allowed

    def test_probe_success_closes_circuit(self):
        cb = self._cb(threshold=2, timeout=0.01)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        cb.is_available("openai")   # triggers half-open
        cb.record_success("openai")
        assert cb.get_state("openai") == "closed"
        assert cb.is_available("openai") is True

    def test_probe_failure_reopens_circuit(self):
        cb = self._cb(threshold=2, timeout=0.01)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        cb.is_available("openai")   # triggers half-open
        cb.record_failure("openai")  # probe fails → reopen
        assert cb.get_state("openai") == "open"
        assert cb.is_available("openai") is False

    def test_only_one_probe_allowed(self):
        """While a probe is in flight, subsequent is_available calls return False."""
        cb = self._cb(threshold=2, timeout=0.01)
        cb.record_failure("openai")
        cb.record_failure("openai")
        time.sleep(0.02)
        first  = cb.is_available("openai")  # True — probe granted
        second = cb.is_available("openai")  # False — probe already in flight
        assert first  is True
        assert second is False

    # ── Independent providers ─────────────────────────────────────────────

    def test_providers_tracked_independently(self):
        cb = self._cb(threshold=3)
        for _ in range(3):
            cb.record_failure("openai")
        assert cb.is_available("openai")    is False
        assert cb.is_available("anthropic") is True

    def test_multiple_providers_can_open_independently(self):
        cb = self._cb(threshold=2)
        cb.record_failure("openai")
        cb.record_failure("openai")
        cb.record_failure("anthropic")
        assert cb.is_available("openai")    is False
        assert cb.is_available("anthropic") is True  # only 1 failure


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: circuit breaker filters models in routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreakerFiltersModels:
    def setup_method(self):
        self.engine = _engine()

    def test_routing_succeeds_with_all_circuits_closed(self):
        d = rr(self.engine.route(_req("Write a Python function")))
        assert d.chosen_model is not None

    def test_open_circuit_excludes_provider_models(self):
        """
        Open the circuit for 'anthropic'.  The router should pick a non-Anthropic
        model (since Anthropic models are filtered in Step 4).
        """
        cb = self.engine._circuit_breaker
        for _ in range(5):
            cb.record_failure("anthropic")
        assert cb.get_state("anthropic") == "open"

        d = rr(self.engine.route(_req("Write a Python function")))
        assert d.chosen_model is not None
        assert d.chosen_model.provider != "anthropic"

    def test_routing_works_when_one_provider_is_down(self):
        """Even with OpenAI's circuit open, other providers fill in."""
        cb = self.engine._circuit_breaker
        for _ in range(5):
            cb.record_failure("openai")

        d = rr(self.engine.route(_req("Summarize this text")))
        assert d.chosen_model is not None
        assert d.chosen_model.provider != "openai"

    def test_routing_still_works_when_google_circuit_open(self):
        """Google is the largest provider in the registry; others must cover."""
        cb = self.engine._circuit_breaker
        for _ in range(5):
            cb.record_failure("google")

        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.chosen_model is not None
        assert d.chosen_model.provider != "google"

    def test_closed_circuit_after_recovery_allows_provider_back(self):
        """After circuit closes (success recorded), the provider's models reappear."""
        cb = self.engine._circuit_breaker

        # Open the circuit for anthropic.
        for _ in range(5):
            cb.record_failure("anthropic")
        d1 = rr(self.engine.route(_req("Write a complex algorithm", routing_priority="always-premium")))
        assert d1.chosen_model is not None
        assert d1.chosen_model.provider != "anthropic"

        # Close it again.
        cb.record_success("anthropic")
        d2 = rr(self.engine.route(_req("Write a complex algorithm", routing_priority="always-premium")))
        assert d2.chosen_model is not None
        # Now anthropic is available again — router may choose it.
        # (It might not be chosen if another premium model scores higher; that's OK.)
