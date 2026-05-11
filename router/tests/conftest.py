import pytest


@pytest.fixture(autouse=True)
def _no_fallback_delays(monkeypatch):
    monkeypatch.setattr("router.fallback_chain.FALLBACK_DELAYS", [0.0, 0.0, 0.0])
    monkeypatch.setattr("router.fallback_chain.FALLBACK_JITTER_MAX", 0.0)
