"""
provider_caller.py — HTTP caller for each model provider, used by proxy mode.

Change 9: When request.mode == "proxy", the routing engine uses this module to
make the actual API call after selecting a model.  Only Python stdlib (urllib)
is used — no new external dependencies.

Supported providers:
  anthropic  → POST https://api.anthropic.com/v1/messages
  openai     → POST https://api.openai.com/v1/chat/completions
  google     → POST https://generativelanguage.googleapis.com/v1beta/models/{id}:generateContent
  groq       → POST https://api.groq.com/openai/v1/chat/completions  (OpenAI-compat)
  mistral    → POST https://api.mistral.ai/v1/chat/completions        (OpenAI-compat)

All callers are async thin wrappers around urllib (run in a thread-pool executor
so the event loop is not blocked).  They raise ProviderCallError on failure with
an http_status attribute so FallbackExecutor can classify the error correctly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import urllib.error
import urllib.request
from typing import Any

import structlog

from .config import (
    LOG_PROMPTS,
    MAX_PROVIDER_RESPONSE_BYTES,
    PROVIDER_CALL_TIMEOUT_SECONDS,
)
from .schemas import ModelOption, RoutingRequest

log = structlog.get_logger(__name__)


def _bounded_read(stream, limit: int = MAX_PROVIDER_RESPONSE_BYTES) -> bytes:
    """Read at most `limit` bytes plus one. Raise if exceeded.

    The +1 lets us detect overflow without buffering the entire malicious payload.
    """
    data = stream.read(limit + 1)
    if len(data) > limit:
        raise ProviderCallError(
            f"Provider response exceeded {limit} bytes",
            http_status=None,
        )
    return data


def _response_shape_keys(data: object) -> list[str]:
    """Return the top-level keys of a parsed provider response, for diagnostic use only.

    This is safe to put in exception messages — it reveals structure but not content."""
    if isinstance(data, dict):
        return sorted(data.keys())
    return [type(data).__name__]


# How we build the messages list from a RoutingRequest
def _build_messages(request: RoutingRequest) -> list[dict[str, str]]:
    """Convert RoutingRequest fields into a universal chat messages list."""
    messages: list[dict[str, str]] = []
    for turn in request.message_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": request.raw_prompt})
    return messages


# ── Custom exception ─────────────────────────────────────────────────────────


class ProviderCallError(Exception):
    """Raised when a provider API call fails.  Carries http_status for routing."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.status_code = http_status  # matches FallbackExecutor expectation


# ── Low-level urllib helper ──────────────────────────────────────────────────


def _post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    provider_name: str,
) -> dict[str, Any]:
    """Synchronous POST; raises ProviderCallError on HTTP errors.

    Error messages intentionally exclude the URL and response body to avoid
    leaking prompt fragments, partial keys, emails, or org IDs into logs and
    exception strings.

    Response bodies and parsed provider data are NEVER included in exception
    messages or log statements at INFO/WARN/ERROR level. They are only logged
    at DEBUG level when LOG_PROMPTS is explicitly enabled (default: False).
    Exception messages contain only provider name, HTTP status, and response
    structure metadata (top-level keys).
    """
    # 🔒 SECURITY-CRITICAL: defense-in-depth against B310 / scheme injection
    if not url.startswith("https://"):
        raise ProviderCallError("Invalid URL scheme; only https allowed", http_status=None)
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_CALL_TIMEOUT_SECONDS) as resp:  # nosec B310 — scheme validated above
            return json.loads(_bounded_read(resp).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = ""
        # Swallow body-read failures so the original HTTP error is what we surface.
        # OSError: socket already closed by the time we read.
        # AttributeError: HTTPError without a readable body.
        # ProviderCallError: body exceeded MAX_PROVIDER_RESPONSE_BYTES (diagnostic only).
        with contextlib.suppress(OSError, AttributeError, ProviderCallError):
            body_text = _bounded_read(exc).decode("utf-8", errors="replace")
        if LOG_PROMPTS:
            log.debug(
                "provider_error_body",
                provider=provider_name,
                status=exc.code,
                body=body_text[:200],
            )
        raise ProviderCallError(
            f"HTTP {exc.code} from {provider_name}",
            http_status=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderCallError(f"Network error calling {provider_name}") from exc


# ── Provider-specific callers (synchronous) ──────────────────────────────────


def _call_anthropic_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> str:
    messages = _build_messages(request)
    body: dict[str, Any] = {
        "model": model.model_id,
        "max_tokens": min(request.max_tokens_requested or 1024, model.max_output_tokens),
        "messages": messages,
    }
    if request.system_prompt:
        body["system"] = request.system_prompt
    if request.temperature is not None:
        body["temperature"] = request.temperature

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages", headers, body, provider_name="anthropic"
    )
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(
            f"Unexpected Anthropic response shape (keys={_response_shape_keys(data)})"
        ) from exc


def _uses_max_completion_tokens(provider_name: str, model_id: str) -> bool:
    """Return True if this OpenAI-compat call must use `max_completion_tokens`.

    OpenAI's reasoning models (o-series) and the modern GPT-5 family reject the
    legacy `max_tokens` parameter and require `max_completion_tokens`. Groq and
    Mistral still expect `max_tokens`, so the switch is OpenAI-only.
    """
    if provider_name != "openai":
        return False
    mid = model_id.lower()
    return mid.startswith(("o1", "o3", "o4", "gpt-5"))


def _call_openai_compat_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
    base_url: str,
    provider_name: str,
) -> str:
    """Shared caller for OpenAI, Groq, and Mistral (all use the same message format)."""
    messages = _build_messages(request)
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    body: dict[str, Any] = {
        "model": model.model_id,
        "messages": messages,
    }
    token_limit = min(request.max_tokens_requested or 1024, model.max_output_tokens)
    if _uses_max_completion_tokens(provider_name, model.model_id):
        body["max_completion_tokens"] = token_limit
    else:
        body["max_tokens"] = token_limit
    if request.temperature is not None:
        body["temperature"] = request.temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post_json(f"{base_url}/chat/completions", headers, body, provider_name=provider_name)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(
            f"Unexpected OpenAI-compat response shape (keys={_response_shape_keys(data)})"
        ) from exc


def _call_google_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> str:
    contents: list[dict[str, Any]] = []
    for turn in request.message_history:
        contents.append(
            {
                "role": turn.get("role", "user"),
                "parts": [{"text": turn.get("content", "")}],
            }
        )
    contents.append({"role": "user", "parts": [{"text": request.raw_prompt}]})

    body: dict[str, Any] = {"contents": contents}
    if request.system_prompt:
        body["system_instruction"] = {"parts": [{"text": request.system_prompt}]}
    # Without an explicit cap, Google generates up to the model's own default
    # (which can far exceed what routing/budget estimated this request would
    # cost) — every other provider caller sets an equivalent limit.
    body["generationConfig"] = {
        "maxOutputTokens": min(request.max_tokens_requested or 1024, model.max_output_tokens)
    }

    model_id = model.model_id.replace("-thinking", "")  # strip suffix for API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    # Pass key as a header to avoid it appearing in server access logs.
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    data = _post_json(url, headers, body, provider_name="google")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(
            f"Unexpected Google response shape (keys={_response_shape_keys(data)})"
        ) from exc


# ── Base URL map ─────────────────────────────────────────────────────────────

_OPENAI_COMPAT_BASES: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
}

# Providers whose wire format is already OpenAI SSE (`data: {...}\n\n` chunks with
# a `choices[0].delta` shape). These can be proxied line-for-line by router/server.py
# without reshaping each chunk. Anthropic and Google use different event formats and
# are not included here — router/server.py falls back to a synthesized single-chunk
# stream for those providers rather than translating their event schemas.
STREAMING_NATIVE_PROVIDERS: frozenset[str] = frozenset(_OPENAI_COMPAT_BASES)


def _open_openai_compat_stream(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
    base_url: str,
    provider_name: str,
):
    """Open a streaming POST to an OpenAI-compatible provider. Returns the open response.

    Synchronous — always called via loop.run_in_executor. The caller is
    responsible for closing the returned response object.
    """
    messages = _build_messages(request)
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    body: dict[str, Any] = {"model": model.model_id, "messages": messages, "stream": True}
    token_limit = min(request.max_tokens_requested or 1024, model.max_output_tokens)
    if _uses_max_completion_tokens(provider_name, model.model_id):
        body["max_completion_tokens"] = token_limit
    else:
        body["max_tokens"] = token_limit
    if request.temperature is not None:
        body["temperature"] = request.temperature

    url = f"{base_url}/chat/completions"
    if not url.startswith("https://"):
        raise ProviderCallError("Invalid URL scheme; only https allowed", http_status=None)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=PROVIDER_CALL_TIMEOUT_SECONDS)  # nosec B310
    except urllib.error.HTTPError as exc:
        raise ProviderCallError(
            f"HTTP {exc.code} from {provider_name}", http_status=exc.code
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderCallError(f"Network error calling {provider_name}") from exc


async def stream_openai_compat_lines(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
):
    """
    Async generator yielding raw SSE lines (bytes, including the trailing
    newline) from an OpenAI-compatible provider as they arrive on the wire.

    Each `readline()` call runs in the executor individually rather than
    reading the whole body at once, so lines are yielded incrementally instead
    of being buffered until the response completes.

    Raises ProviderCallError if the provider is not OpenAI-compatible, or on
    HTTP/network failure when opening the connection.
    """
    provider = model.provider.lower()
    base_url = _OPENAI_COMPAT_BASES.get(provider)
    if base_url is None:
        raise ProviderCallError(f"'{provider}' does not support native SSE streaming")

    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None, _open_openai_compat_stream, model, request, api_key, base_url, provider
    )
    total_bytes = 0
    try:
        while True:
            line = await loop.run_in_executor(None, resp.readline)
            if not line:
                break
            total_bytes += len(line)
            if total_bytes > MAX_PROVIDER_RESPONSE_BYTES:
                raise ProviderCallError(
                    f"Streamed response from {provider} exceeded "
                    f"{MAX_PROVIDER_RESPONSE_BYTES} bytes",
                    http_status=None,
                )
            yield line
    finally:
        await loop.run_in_executor(None, resp.close)


# ── Public async entry point ─────────────────────────────────────────────────


async def call_provider(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> str:
    """
    Async wrapper: dispatches to the correct provider caller in a thread-pool
    executor so the routing event loop is not blocked.

    Raises ProviderCallError (with .status_code) on failure.
    """
    provider = model.provider.lower()
    loop = asyncio.get_running_loop()

    # 🔧 EXTENSION POINT: Add a new provider here.
    # Steps: (1) add the provider name to this if/elif chain,
    #        (2) implement _call_<provider>_sync() below following the existing pattern,
    #        (3) add the provider's base URL to _OPENAI_COMPAT_BASES if it's OpenAI-compatible,
    #        (4) add models for that provider to router/models.json.
    if provider == "anthropic":

        def fn() -> str:
            return _call_anthropic_sync(model, request, api_key)
    elif provider in _OPENAI_COMPAT_BASES:
        base = _OPENAI_COMPAT_BASES[provider]

        def fn() -> str:
            return _call_openai_compat_sync(model, request, api_key, base, provider)
    elif provider == "google":

        def fn() -> str:
            return _call_google_sync(model, request, api_key)
    else:
        raise ProviderCallError(
            f"No provider caller implemented for '{provider}'.  "
            f"Supported: anthropic, openai, groq, mistral, google."
        )

    log.debug("proxy_call_start", provider=provider, model=model.model_id)
    return await loop.run_in_executor(None, fn)
