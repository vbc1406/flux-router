import pytest


@pytest.fixture(autouse=True)
def _no_fallback_delays(monkeypatch):
    monkeypatch.setattr("router.fallback_chain.FALLBACK_DELAYS", [0.0, 0.0, 0.0])
    monkeypatch.setattr("router.fallback_chain.FALLBACK_JITTER_MAX", 0.0)


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
