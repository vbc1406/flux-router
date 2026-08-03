"""
test_rate_limit.py — inbound request rate limiting (router/rate_limit.py).

Covers the bucket mechanics directly, plus the proxy wiring in server.py:

  TestTokenBucket          — burst, refill, per-key isolation, disable switch
  TestKeyTableBound        — the LRU cap that keeps the key table from growing
  TestRateLimitKey         — which identity a request counts against
  TestProxyEnforcement     — 429 + Retry-After on POST /v1/chat/completions
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("fastapi")

from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from router import server  # noqa: E402
from router.flux import make_flux  # noqa: E402
from router.provider_caller import ProviderResult  # noqa: E402
from router.rate_limit import RateLimiter  # noqa: E402


def _body() -> dict:
    """A request body with a unique `user` per call.

    These tests fire far more requests than a typical test does, and every
    successful one bills the process-global BudgetTracker/DailyBudgetTracker
    ledger for whichever user_id it used. Left at the default
    ("flux-server-anonymous", shared with test_server.py) the accumulated
    spend pushes later tests in other files into budget-constrained routing.
    A fresh user per request keeps that ledger growth out of everyone else's
    way — the rate limiter is keyed by tenant/IP, so this doesn't weaken what
    is being tested here.
    """
    return {
        "model": "flux-auto",
        "messages": [{"role": "user", "content": "hi"}],
        "user": f"rate-limit-test-{uuid.uuid4()}",
    }


# One throwaway Flux for this whole module, swapped in for the process-global
# server._flux below. Built once because each instance spins up a SQLite usage
# store and its writer thread.
_ISOLATED_FLUX = make_flux()


@pytest.fixture(autouse=True)
def _isolated_engine(monkeypatch):
    """Point the proxy at a throwaway Flux for the duration of each test.

    These tests push more completions through the proxy than most, and
    server._flux is a process-global whose routing state (adaptive weights,
    prompt-cache warmth, circuit breakers, budget ledgers) accumulates across
    the entire test session. Left on the shared instance, this file changes
    which model test_server.py's later tests get routed to and breaks them —
    a coupling that predates this file and is easy to trip into. Swapping the
    instance keeps that blast radius at zero rather than tuning request counts
    against a threshold nobody documented.
    """
    monkeypatch.setattr(server, "_flux", _ISOLATED_FLUX)


@pytest.fixture(autouse=True)
def _mock_call_model(_isolated_engine, monkeypatch):
    """Same stub as test_server.py — keeps these tests off the network so a
    429 assertion can't be satisfied by an unrelated upstream failure.
    Depends on _isolated_engine so it patches the throwaway instance."""
    mock = AsyncMock(
        return_value=ProviderResult(
            text="mock response text",
            input_tokens=None,
            output_tokens=None,
            usage_source="estimated",
        )
    )
    monkeypatch.setattr(server._flux, "_call_model", mock)
    return mock


class TestTokenBucket:
    def test_allows_up_to_burst_then_denies(self):
        rl = RateLimiter(rpm=60, burst=5)
        assert [rl.check("k") for _ in range(5)] == [None] * 5
        retry_after = rl.check("k")
        assert retry_after is not None
        assert retry_after > 0

    def test_keys_are_isolated(self):
        rl = RateLimiter(rpm=60, burst=2)
        assert rl.check("a") is None
        assert rl.check("a") is None
        assert rl.check("a") is not None
        # "b" has its own bucket and is untouched by "a" exhausting theirs.
        assert rl.check("b") is None

    def test_tokens_refill_over_time(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("router.rate_limit.time.monotonic", lambda: clock["t"])

        rl = RateLimiter(rpm=60, burst=1)  # 1 token/sec
        assert rl.check("k") is None
        assert rl.check("k") is not None  # drained

        clock["t"] += 1.0  # one second → one token
        assert rl.check("k") is None

    def test_refill_is_capped_at_burst(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("router.rate_limit.time.monotonic", lambda: clock["t"])

        rl = RateLimiter(rpm=60, burst=3)
        assert rl.check("k") is None
        clock["t"] += 3600.0  # an hour of idling must not bank 3600 tokens
        assert [rl.check("k") for _ in range(3)] == [None] * 3
        assert rl.check("k") is not None

    def test_retry_after_reports_time_until_next_token(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("router.rate_limit.time.monotonic", lambda: clock["t"])

        rl = RateLimiter(rpm=60, burst=1)  # 1 token/sec
        assert rl.check("k") is None
        retry_after = rl.check("k")
        assert retry_after == pytest.approx(1.0, abs=0.01)

        clock["t"] += 0.75  # three quarters of a token has accrued
        retry_after = rl.check("k")
        assert retry_after == pytest.approx(0.25, abs=0.01)

    def test_rpm_zero_disables_entirely(self):
        rl = RateLimiter(rpm=0, burst=1)
        assert rl.enabled is False
        assert all(rl.check("k") is None for _ in range(1000))

    def test_burst_floor_of_one(self):
        """burst=0 would make a bucket that can never admit anything."""
        rl = RateLimiter(rpm=60, burst=0)
        assert rl.check("k") is None


class TestKeyTableBound:
    def test_key_table_does_not_grow_without_bound(self):
        rl = RateLimiter(rpm=60, burst=1, max_keys=10)
        for i in range(500):
            rl.check(f"key-{i}")
        assert len(rl._buckets) <= 10

    def test_eviction_is_lru_not_fifo(self):
        rl = RateLimiter(rpm=6000, burst=100, max_keys=3)
        rl.check("a")
        rl.check("b")
        rl.check("c")
        rl.check("a")  # touch "a" so "b" becomes least-recently-used
        rl.check("d")  # evicts one key
        assert "a" in rl._buckets
        assert "b" not in rl._buckets

    def test_evicted_key_resets_rather_than_being_denied(self):
        """Documented tradeoff: a key-flood makes the limiter approximate
        instead of letting an attacker lock out legitimate callers."""
        rl = RateLimiter(rpm=60, burst=1, max_keys=1)
        assert rl.check("victim") is None
        assert rl.check("victim") is not None  # drained
        rl.check("flood")  # evicts "victim"
        assert rl.check("victim") is None  # back with a full bucket, not a 429


class TestRateLimitKey:
    def _request(self, host: str | None):
        class _Client:
            def __init__(self, h):
                self.host = h

        class _Req:
            def __init__(self, h):
                self.client = _Client(h) if h is not None else None

        return _Req(host)

    def test_prefers_bound_tenant(self):
        req = self._request("10.0.0.1")
        assert server._rate_limit_key(req, "acme") == "tenant:acme"

    def test_falls_back_to_peer_ip(self):
        req = self._request("10.0.0.1")
        assert server._rate_limit_key(req, None) == "ip:10.0.0.1"

    def test_handles_missing_client(self):
        req = self._request(None)
        assert server._rate_limit_key(req, None) == "anonymous"

    def test_ignores_spoofable_forwarded_for(self):
        """X-Forwarded-For is caller-supplied; keying on it would hand every
        client an unlimited supply of fresh buckets."""
        req = self._request("10.0.0.1")
        req.headers = {"X-Forwarded-For": "1.2.3.4"}
        assert server._rate_limit_key(req, None) == "ip:10.0.0.1"

    def test_bound_tenant_beats_self_declared_header(self, monkeypatch):
        """A caller rotating X-Flux-Tenant-Id must not get a fresh bucket."""
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "SERVER_TOKENS", {"tok": "acme"})
        monkeypatch.setattr(server, "_rate_limiter", RateLimiter(rpm=60, burst=2))
        client = TestClient(server.app)

        headers = {"Authorization": "Bearer tok"}
        for spoofed in ("t1", "t2", "t3"):
            resp = client.post(
                "/v1/chat/completions",
                json=_body(),
                headers={**headers, "X-Flux-Tenant-Id": spoofed},
            )
        # Third request is over a burst of 2 regardless of the header churn.
        assert resp.status_code == 429


class TestProxyEnforcement:
    @pytest.fixture
    def client(self, monkeypatch):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", False)
        monkeypatch.setattr(server, "_rate_limiter", RateLimiter(rpm=60, burst=3))
        return TestClient(server.app)

    def test_returns_429_with_retry_after_once_over_limit(self, client, _mock_call_model):
        for _ in range(3):
            assert client.post("/v1/chat/completions", json=_body()).status_code == 200
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 429
        assert int(resp.headers["Retry-After"]) >= 1

    def test_limit_applies_before_the_body_is_read(self, client, _mock_call_model):
        """A rejected request must not first cost us SERVER_MAX_BODY_BYTES of
        body read — an oversized body should still 429, not 413."""
        for _ in range(3):
            client.post("/v1/chat/completions", json=_body())
        huge = {"model": "flux-auto", "messages": [{"role": "user", "content": "x" * 5_000_000}]}
        resp = client.post("/v1/chat/completions", json=huge)
        assert resp.status_code == 429

    def test_rejected_request_is_never_dispatched(self, client, _mock_call_model):
        for _ in range(3):
            client.post("/v1/chat/completions", json=_body())
        calls_before = _mock_call_model.call_count
        assert client.post("/v1/chat/completions", json=_body()).status_code == 429
        assert _mock_call_model.call_count == calls_before

    def test_health_is_never_rate_limited(self, client):
        """Liveness probes must not be able to lock themselves out."""
        for _ in range(50):
            assert client.get("/health").status_code == 200

    def test_disabled_limiter_lets_everything_through(self, monkeypatch, _mock_call_model):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", False)
        monkeypatch.setattr(server, "_rate_limiter", RateLimiter(rpm=0, burst=1))
        client = TestClient(server.app)
        for _ in range(20):
            assert client.post("/v1/chat/completions", json=_body()).status_code == 200
