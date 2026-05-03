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
import json
import urllib.error
import urllib.request
from typing import Any

import structlog

from .schemas import ModelOption, RoutingRequest

log = structlog.get_logger(__name__)

# How we build the messages list from a RoutingRequest
def _build_messages(request: RoutingRequest) -> list[dict[str, str]]:
    """Convert RoutingRequest fields into a universal chat messages list."""
    messages: list[dict[str, str]] = []
    for turn in request.message_history:
        role    = turn.get("role", "user")
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
    exception strings. Response bodies are emitted at DEBUG level only.
    """
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
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
        "model":      model.model_id,
        "max_tokens": request.max_tokens_requested or 1024,
        "messages":   messages,
    }
    if request.system_prompt:
        body["system"] = request.system_prompt
    if request.temperature is not None:
        body["temperature"] = request.temperature

    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages", headers, body, provider_name="anthropic"
    )
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(f"Unexpected Anthropic response shape: {data}") from exc


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
        "model":      model.model_id,
        "messages":   messages,
        "max_tokens": request.max_tokens_requested or 1024,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    data = _post_json(
        f"{base_url}/chat/completions", headers, body, provider_name=provider_name
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(f"Unexpected OpenAI-compat response shape: {data}") from exc


def _call_google_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> str:
    contents: list[dict[str, Any]] = []
    for turn in request.message_history:
        contents.append({
            "role":  turn.get("role", "user"),
            "parts": [{"text": turn.get("content", "")}],
        })
    contents.append({"role": "user", "parts": [{"text": request.raw_prompt}]})

    body: dict[str, Any] = {"contents": contents}
    if request.system_prompt:
        body["system_instruction"] = {"parts": [{"text": request.system_prompt}]}

    model_id = model.model_id.replace("-thinking", "")  # strip suffix for API
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent"
    )
    # Pass key as a header to avoid it appearing in server access logs.
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    data = _post_json(url, headers, body, provider_name="google")
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(f"Unexpected Google response shape: {data}") from exc


# ── Base URL map ─────────────────────────────────────────────────────────────

_OPENAI_COMPAT_BASES: dict[str, str] = {
    "openai":  "https://api.openai.com/v1",
    "groq":    "https://api.groq.com/openai/v1",
    "mistral": "https://api.mistral.ai/v1",
}


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
    loop     = asyncio.get_running_loop()

    # 🔧 EXTENSION POINT: Add a new provider here.
    # Steps: (1) add the provider name to this if/elif chain,
    #        (2) implement _call_<provider>_sync() below following the existing pattern,
    #        (3) add the provider's base URL to _OPENAI_COMPAT_BASES if it's OpenAI-compatible,
    #        (4) add models for that provider to router/models.json.
    if provider == "anthropic":
        fn = lambda: _call_anthropic_sync(model, request, api_key)
    elif provider in _OPENAI_COMPAT_BASES:
        base = _OPENAI_COMPAT_BASES[provider]
        fn   = lambda: _call_openai_compat_sync(model, request, api_key, base, provider)
    elif provider == "google":
        fn = lambda: _call_google_sync(model, request, api_key)
    else:
        raise ProviderCallError(
            f"No provider caller implemented for '{provider}'.  "
            f"Supported: anthropic, openai, groq, mistral, google."
        )

    log.debug("proxy_call_start", provider=provider, model=model.model_id)
    return await loop.run_in_executor(None, fn)
