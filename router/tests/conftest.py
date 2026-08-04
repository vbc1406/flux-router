import pytest


@pytest.fixture(autouse=True)
def _no_fallback_delays(monkeypatch):
    monkeypatch.setattr("router.fallback_chain.FALLBACK_DELAYS", [0.0, 0.0, 0.0])
    monkeypatch.setattr("router.fallback_chain.FALLBACK_JITTER_MAX", 0.0)


@pytest.fixture(autouse=True)
def _reset_model_load_windows():
    """Clear the registry's sliding-window RPM history between tests.

    ModelRegistry tracks per-model requests-per-minute in a wall-clock window
    and drops a model from the candidate list once it reaches
    RATE_LIMIT_SAFETY_MARGIN of its stated cap. `gemini-2.0-flash-free` — the
    model most proxy tests expect to be routed to — is capped at 15 RPM, so
    the threshold is 15 * 0.85 = 12.75: the 13th proxy request in any 60
    second window retires it and everything after that routes elsewhere.

    Nothing resets that between tests, so a file dispatching 13+ completions
    silently changed which model LATER tests in OTHER files were routed to.
    It surfaced as failures in test_server.py (fallback chains finding no
    candidates, usage records attributed to an unexpected model) that pointed
    nowhere near the file actually responsible.
    """
    try:
        from router import server
    except ImportError:  # pragma: no cover - fastapi extra not installed
        yield
        return
    server._flux._engine._registry.reset_load_tracking()
    yield
    server._flux._engine._registry.reset_load_tracking()


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
