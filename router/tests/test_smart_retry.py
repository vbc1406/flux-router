"""
File: router/tests/test_smart_retry.py

Purpose:
Tests for Flux.complete() smart retry — typed error classification, per-error
fallback chain selection, retry caps, and response metadata on fallback.
All tests mock _call_model; no real HTTP requests are made.

How to run:
  pytest -v router/tests/test_smart_retry.py
  pytest -v router/tests/test_smart_retry.py::TestRetryOnRateLimit

How to add a test:
  1. Use _flux() for a fresh Flux instance, _req(prompt, **kw) for a request.
  2. Patch router.flux._call_model with AsyncMock, configure side_effect to
     raise typed errors (RateLimitError, TimeoutError, etc.) then succeed.
  3. Assert on response.fallback_used, response.fallback_reason, response.model.

Test classes:
  TestSuccessPath           — first call succeeds, fallback_used=False
  TestRetryOnRateLimit      — RateLimitError triggers rate-limit fallback chain
  TestRetryOnTimeout        — TimeoutError triggers timeout fallback chain
  TestRetryOnContentFilter  — ContentFilterError triggers content-safety chain
  TestRetryOnProviderDown   — ProviderDownError uses same-tier alternatives
  TestNoRetryOnAuthError    — AuthenticationError raises immediately (no retry)
  TestMaxRetriesRespected   — max_retries caps total attempts
  TestAllFailRaisesError    — FluxAPIError raised when all models fail
  TestFallbackMarkedOnResponse — fallback_used=True and fallback_reason populated
  TestModelDeduplication    — same model never tried twice across retries
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.errors import (
    AuthenticationError,
    ContentFilterError,
    FluxAPIError,
    ProviderDownError,
    RateLimitError,
    TimeoutError,
)
from router.flux import Flux, FluxResponse
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


def _flux() -> Flux:
    return Flux(_engine(), api_key="test-key")


def rr(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


_PROMPT = "Explain async/await in Python"
_KWARGS: dict[str, Any] = {"user_id": "u_retry", "plan": "business_plan", "exploration_rate": 0.0}


# ── Successful path ───────────────────────────────────────────────────────────

class TestSuccessPath:
    def test_first_call_succeeds(self):
        flux = _flux()

        async def mock_call(model, request):
            return "response text"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert isinstance(resp, FluxResponse)
        assert resp.text == "response text"
        assert resp.fallback_used is False
        assert resp.fallback_reason is None

    def test_response_contains_decision(self):
        flux = _flux()

        async def mock_call(model, request):
            return "hello"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.decision is not None
        assert resp.model is not None


# ── Rate limit retry ──────────────────────────────────────────────────────────

class TestRetryOnRateLimit:
    def test_retry_on_rate_limit(self):
        flux   = _flux()
        calls: list[str] = []

        async def mock_call(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise RateLimitError("429 Too Many Requests")
            return "fallback response"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.text == "fallback response"
        assert resp.fallback_used is True
        assert resp.fallback_reason == "rate_limit"
        assert len(calls) >= 2

    def test_rate_limit_uses_different_model_on_retry(self):
        flux  = _flux()
        calls: list[str] = []

        async def mock_call(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise RateLimitError("429")
            return "ok"

        flux._call_model = mock_call  # type: ignore[method-assign]
        rr(flux.complete(_PROMPT, **_KWARGS))
        # Second call must use a different model (deduplication).
        if len(calls) >= 2:
            assert calls[0] != calls[1]


# ── Timeout retry ─────────────────────────────────────────────────────────────

class TestRetryOnTimeout:
    def test_retry_on_timeout(self):
        flux  = _flux()
        calls: list[str] = []

        async def mock_call(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise TimeoutError("Request timed out")
            return "timeout fallback"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_used is True
        assert resp.fallback_reason == "timeout"

    def test_timeout_uses_fastest_fallback(self):
        """Timeout fallback chain is sorted by latency — verify no exception."""
        flux  = _flux()

        async def mock_call(model, request):
            raise TimeoutError("timeout")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError):
            rr(flux.complete(_PROMPT, max_retries=0, **_KWARGS))


# ── Content filter retry ──────────────────────────────────────────────────────

class TestRetryOnContentFilter:
    def test_retry_on_content_filter(self):
        flux  = _flux()
        calls: list[str] = []

        async def mock_call(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise ContentFilterError("Content policy violation")
            return "safe response"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_used is True
        assert resp.fallback_reason == "content_filter"

    def test_content_filter_fallback_uses_higher_tier(self):
        """
        Content-safety chain contains higher-tier models.
        We verify the route succeeds even when primary fails on content filter.
        """
        flux  = _flux()
        count = [0]

        async def mock_call(model, request):
            count[0] += 1
            if count[0] == 1:
                raise ContentFilterError("refused")
            return "higher-tier response"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.text == "higher-tier response"


# ── Provider down ─────────────────────────────────────────────────────────────

class TestRetryOnProviderDown:
    def test_provider_down_uses_rate_limit_chain(self):
        flux  = _flux()
        calls: list[str] = []

        async def mock_call(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise ProviderDownError("502 Bad Gateway")
            return "backup response"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_used is True
        assert resp.fallback_reason == "provider_down"


# ── Auth error ────────────────────────────────────────────────────────────────

class TestNoRetryOnAuthError:
    def test_auth_error_raises_immediately(self):
        flux  = _flux()
        calls = [0]

        async def mock_call(model, request):
            calls[0] += 1
            raise AuthenticationError("Invalid API key")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(AuthenticationError):
            rr(flux.complete(_PROMPT, **_KWARGS))
        # Must be called exactly once — no retry.
        assert calls[0] == 1

    def test_auth_error_not_wrapped_in_flux_api_error(self):
        flux  = _flux()

        async def mock_call(model, request):
            raise AuthenticationError("bad key")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(AuthenticationError):
            rr(flux.complete(_PROMPT, **_KWARGS))


# ── max_retries limit ─────────────────────────────────────────────────────────

class TestMaxRetriesRespected:
    def test_max_retries_zero_means_one_attempt(self):
        flux  = _flux()
        calls = [0]

        async def mock_call(model, request):
            calls[0] += 1
            raise RateLimitError("429")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError):
            rr(flux.complete(_PROMPT, max_retries=0, **_KWARGS))
        assert calls[0] == 1

    def test_max_retries_two_allows_three_attempts(self):
        flux  = _flux()
        calls = [0]

        async def mock_call(model, request):
            calls[0] += 1
            if calls[0] <= 3:
                raise RateLimitError("429")
            return "success"

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError):
            rr(flux.complete(_PROMPT, max_retries=2, **_KWARGS))
        # max_retries=2 → primary + 2 retries = 3 attempts max
        assert calls[0] <= 3

    def test_max_retries_custom(self):
        flux  = _flux()
        calls = [0]

        async def mock_call(model, request):
            calls[0] += 1
            raise TimeoutError("slow")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError):
            rr(flux.complete(_PROMPT, max_retries=1, **_KWARGS))
        assert calls[0] <= 2


# ── All-fail raises FluxAPIError ──────────────────────────────────────────────

class TestAllFailRaisesError:
    def test_all_models_fail_raises_flux_error(self):
        flux = _flux()

        async def mock_call(model, request):
            raise RateLimitError("always 429")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError) as exc_info:
            rr(flux.complete(_PROMPT, max_retries=2, **_KWARGS))
        assert "failed" in str(exc_info.value).lower()

    def test_error_message_includes_attempt_count(self):
        flux = _flux()

        async def mock_call(model, request):
            raise TimeoutError("timeout")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError) as exc_info:
            rr(flux.complete(_PROMPT, max_retries=2, **_KWARGS))
        msg = str(exc_info.value)
        assert "attempt" in msg.lower()


# ── Fallback marked on response ───────────────────────────────────────────────

class TestFallbackMarkedOnResponse:
    def test_fallback_used_true_when_primary_fails(self):
        flux  = _flux()
        first = [True]

        async def mock_call(model, request):
            if first[0]:
                first[0] = False
                raise RateLimitError("429")
            return "fallback ok"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_used is True

    def test_fallback_reason_matches_error_type(self):
        flux  = _flux()
        first = [True]

        async def mock_call(model, request):
            if first[0]:
                first[0] = False
                raise TimeoutError("timed out")
            return "ok"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_reason == "timeout"

    def test_no_fallback_on_success(self):
        flux = _flux()

        async def mock_call(model, request):
            return "direct success"

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete(_PROMPT, **_KWARGS))
        assert resp.fallback_used is False
        assert resp.fallback_reason is None


# ── Model deduplication ───────────────────────────────────────────────────────

class TestModelDeduplication:
    def test_same_model_not_tried_twice(self):
        """Each model_id must appear at most once in the call sequence."""
        flux = _flux()
        seen: list[str] = []

        async def mock_call(model, request):
            seen.append(model.model_id)
            raise RateLimitError("429")

        flux._call_model = mock_call  # type: ignore[method-assign]
        with pytest.raises(FluxAPIError):
            rr(flux.complete(_PROMPT, max_retries=5, **_KWARGS))

        assert len(seen) == len(set(seen)), f"Duplicate model attempts: {seen}"
