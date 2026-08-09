"""
errors.py — Typed exception hierarchy for Flux (Fix 3).

All public API errors raised by Flux.complete() derive from FluxAPIError so
callers can catch the base class or be specific about the error type.

Usage:
    from router.errors import (
        FluxAPIError,
        RateLimitError,
        TimeoutError,
        ContentFilterError,
        AuthenticationError,
        ProviderDownError,
    )
"""

from __future__ import annotations


class FluxAPIError(Exception):
    """Base class for all errors raised by the Flux API layer."""


class RateLimitError(FluxAPIError):
    """
    Provider returned HTTP 429.  retry_after (seconds) may be set when the
    provider includes a Retry-After header.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class TimeoutError(FluxAPIError):
    """Provider call exceeded the configured timeout."""


class ContentFilterError(FluxAPIError):
    """Provider refused to complete the request due to its content policy."""


class AuthenticationError(FluxAPIError):
    """
    API key is missing, invalid, or revoked.
    Never retried — the caller must fix the key first.
    """


class ProviderDownError(FluxAPIError):
    """
    Provider returned a 5xx server error or is otherwise unavailable.
    Routed to the rate-limit fallback chain (same-tier alternatives).
    """


class UnsupportedFeatureError(FluxAPIError):
    """
    The request asked for a feature the chosen provider doesn't support
    (currently: response_format routed to an Anthropic model, which has no
    native JSON-mode equivalent — see provider_caller.py). This is a caller
    error, not an upstream failure — never retried, and router/server.py
    maps it to HTTP 400 rather than the 502 used for real provider outages.
    """
