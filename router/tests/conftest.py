import pytest


@pytest.fixture(autouse=True)
def _no_fallback_delays(monkeypatch):
    monkeypatch.setattr("router.fallback_chain.FALLBACK_DELAYS", [0.0, 0.0, 0.0])
    monkeypatch.setattr("router.fallback_chain.FALLBACK_JITTER_MAX", 0.0)


@pytest.fixture(autouse=True)
def _reset_model_load_and_circuits():
    """Clear the registry's RPM window and the per-provider circuit breaker
    between tests — both are process-global state on the shared `server._flux`
    singleton, so either one leaking between tests silently changes which
    model a LATER, unrelated test gets routed to.

    RPM window: ModelRegistry tracks per-model requests-per-minute in a
    wall-clock window and drops a model from the candidate list once it
    reaches RATE_LIMIT_SAFETY_MARGIN of its stated cap — e.g. a free-tier
    model capped at 15 RPM retires after its 13th request in any 60 second
    window. A file dispatching 13+ completions to the same model silently
    changed routing for tests in other files, surfacing as fallback chains
    finding no candidates or usage records attributed to an unexpected model.

    Circuit breaker: CircuitBreaker trips a provider open after
    CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive failures and excludes every
    model from that provider until the recovery timeout elapses. Tests that
    simulate persistent provider failures (auth errors, "always 429") trip
    it; without a reset, the tripped provider stays excluded from every
    later test in the process for up to CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    seconds — invisible when the free/cheap tiers have enough other-provider
    headroom to mask it, and a wrong-model / no-fallback-candidates failure
    in some unrelated later test when they don't.
    """
    try:
        from router import server
    except ImportError:  # pragma: no cover - fastapi extra not installed
        yield
        return
    server._flux._engine._registry.reset_load_tracking()
    server._flux._engine._circuit_breaker.reset()
    yield
    server._flux._engine._registry.reset_load_tracking()
    server._flux._engine._circuit_breaker.reset()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Drop inbound rate-limit buckets between tests.

    server._rate_limiter is a module-level singleton and every TestClient
    request reports the same peer host, so a whole test file shares one
    bucket. The suite currently fits inside the default burst, but only just
    (test_server.py alone is ~90 requests against a burst of 100) — without
    this, adding a handful of server tests would start producing 429s that
    look nothing like their actual cause. Real deployments key by tenant or
    by distinct client IPs; this is test isolation, not a workaround for
    production behaviour.
    """
    try:
        from router import server
    except ImportError:  # pragma: no cover - fastapi extra not installed
        yield
        return
    server._rate_limiter.reset()
    yield
    server._rate_limiter.reset()
