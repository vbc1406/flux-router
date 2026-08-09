"""Regression tests for fail-closed HTTP binding configuration."""

import pytest

from router.config import validate_server_binding


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.8", "::1", "[::1]", "localhost"])
def test_unauthenticated_loopback_binding_is_allowed(host):
    validate_server_binding(host, False)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.2", "flux.internal", ""])
def test_unauthenticated_remote_binding_is_rejected(host):
    with pytest.raises(RuntimeError, match="Refusing unauthenticated server bind"):
        validate_server_binding(host, False)


def test_authenticated_remote_binding_is_allowed():
    validate_server_binding("0.0.0.0", True)


def test_explicit_dangerous_override_is_allowed():
    validate_server_binding("0.0.0.0", False, allow_unauthenticated_remote=True)
