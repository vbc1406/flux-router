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
from dataclasses import dataclass
from typing import Any, AsyncIterator

import structlog

from .config import (
    LOG_PROMPTS,
    MAX_PROVIDER_RESPONSE_BYTES,
    PROVIDER_CALL_TIMEOUT_SECONDS,
)
from .schemas import ModelOption, RoutingRequest

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    """Result of a single (non-streaming) provider call.

    input_tokens/output_tokens are the provider's OWN reported usage figures
    (not our pre-dispatch estimate) — None when the provider didn't report
    usage, or reported something we don't trust (see _safe_usage_int).
    usage_source is "provider" when both are present, "estimated" otherwise,
    so callers never have to re-derive which case they're in.

    tool_calls, when present, is OpenAI-shaped regardless of which provider
    produced the response: [{"id": ..., "type": "function", "function":
    {"name": ..., "arguments": "<json string>"}}, ...] — Anthropic's
    tool_use blocks and Google's functionCall parts are translated into this
    shape at parse time so every caller (Flux SDK, the HTTP proxy) handles
    one format. finish_reason is normalized to OpenAI's vocabulary
    ("stop" | "tool_calls" | "length" | "content_filter").
    """

    text: str
    input_tokens: int | None
    output_tokens: int | None
    usage_source: str  # "provider" | "estimated"
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str = "stop"


def _safe_usage_int(value: object) -> int | None:
    """Coerce a raw provider usage field to a positive int, or None if it's
    missing, the wrong type, or non-positive. A real completion always
    consumes at least one token in each direction it's non-empty on, so a
    zero/negative/non-numeric count is not a trustworthy usage figure —
    treated the same as "provider didn't report this" rather than billed
    as $0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    ivalue = int(value)
    return ivalue if ivalue > 0 else None


def _extract_usage(
    data: object,
    provider: str,
    input_key: str,
    output_key: str,
    container_key: str,
) -> tuple[int | None, int | None]:
    """Best-effort extraction of (input_tokens, output_tokens) from a parsed
    provider response. Never raises — a missing/malformed usage shape must
    not turn a successful completion into an error; it just falls back to
    "estimated" for billing purposes. Logs a debug event with the provider
    name and top-level response shape (never response content) so a
    persistently-missing usage shape is diagnosable without leaking data.
    """
    try:
        if not isinstance(data, dict):
            raise TypeError("response is not an object")
        usage = data.get(container_key)
        if not isinstance(usage, dict):
            raise TypeError(f"'{container_key}' missing or not an object")
        input_tokens = _safe_usage_int(usage.get(input_key))
        output_tokens = _safe_usage_int(usage.get(output_key))
        if input_tokens is None or output_tokens is None:
            raise ValueError(f"'{input_key}'/'{output_key}' missing or invalid")
        return input_tokens, output_tokens
    except (KeyError, TypeError, ValueError) as exc:
        log.debug(
            "provider_usage_missing",
            provider=provider,
            shape=_response_shape_keys(data),
            reason=str(exc),
        )
        return None, None


def compute_actual_cost(model: ModelOption, input_tokens: int, output_tokens: int) -> float:
    """Cost computed from ACTUAL (provider-reported) token counts, as opposed
    to routing_engine._estimate_cost()'s pre-dispatch estimate. Used only for
    post-dispatch spend recording — pre-dispatch budget checks keep using the
    estimate, by definition (actual usage isn't known until after the call).

    # TODO(prompt-cache pricing): this ignores cache-read/cache-write pricing
    # (model.cache_read_cost_per_1m / cache_write_cost_per_1m) since Flux
    # doesn't send cache-control blocks to providers yet. Once it does, a
    # cache-hit response's actual cost must be split accordingly instead of
    # priced entirely at the base input rate.
    """
    return round(
        (input_tokens / 1000.0) * model.cost_per_1k_input
        + (output_tokens / 1000.0) * model.cost_per_1k_output,
        6,
    )


def _bounded_read(stream: Any, limit: int = MAX_PROVIDER_RESPONSE_BYTES) -> bytes:
    """Read at most `limit` bytes plus one. Raise if exceeded.

    The +1 lets us detect overflow without buffering the entire malicious payload.
    `stream` is whatever urllib.request.urlopen()/HTTPError gives us — both
    support .read(n) -> bytes but share no common typed base worth importing
    just for this.
    """
    data: bytes = stream.read(limit + 1)
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
def _build_messages(request: RoutingRequest) -> list[dict[str, Any]]:
    """Convert RoutingRequest fields into a universal (OpenAI-shaped) chat
    messages list. Preserves `tool_calls` (on assistant turns) and
    `tool_call_id`/`name` (on tool-role turns) so multi-turn tool
    conversations survive the round trip — each provider-specific caller
    re-translates this universal shape into its own wire format."""
    messages: list[dict[str, Any]] = []
    for turn in request.message_history:
        role = turn.get("role", "user")
        msg: dict[str, Any] = {"role": role, "content": turn.get("content", "")}
        if turn.get("tool_calls"):
            msg["tool_calls"] = turn["tool_calls"]
        if turn.get("tool_call_id"):
            msg["tool_call_id"] = turn["tool_call_id"]
        if turn.get("name"):
            msg["name"] = turn["name"]
        messages.append(msg)
    # Bugfix: when the last turn in message_history is itself role="tool"
    # (server.py's _messages_to_request_fields() now keeps it there instead
    # of folding it into raw_prompt — see that function's docstring), there
    # is no new prompt to add: raw_prompt is "" and appending it anyway would
    # tack on a spurious, unlabeled empty user turn right after a tool
    # observation the provider still needs a response to.
    if not (messages and messages[-1]["role"] == "tool"):
        messages.append({"role": "user", "content": request.raw_prompt})
    return messages


# ── Tool-calling / response_format translation (Item 1) ──────────────────────
#
# request.tools/tool_choice/response_format are always OpenAI-shaped (that's
# what the HTTP proxy and the Python SDK both accept). Each non-OpenAI-compat
# provider gets a pair of functions here: one to translate the OUTGOING
# request, one to translate the INCOMING response's tool-call/content shape
# back into the OpenAI shape ProviderResult carries. OpenAI/Groq/Mistral need
# no outgoing translation (tools/tool_choice/response_format pass through
# verbatim) since they already speak this format.


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _openai_tool_choice_to_anthropic(
    tool_choice: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"type": "auto"}
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if tool_choice.get("type") == "function" and name:
            return {"type": "tool", "name": name}
    raise ProviderCallError(f"Unsupported tool_choice value for anthropic: {tool_choice!r}")


def _parse_anthropic_content(
    content: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]] | None]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content:
        btype = block.get("type")
        # A real Anthropic response always sets "type", but tolerate a block
        # that just has "text" and no/other type — matches the pre-Item-1
        # behavior of reading content[0]["text"] unconditionally.
        if btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif "text" in block:
            text_parts.append(block.get("text", ""))
    return "".join(text_parts), (tool_calls or None)


def _openai_tools_to_google(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decls = []
    for t in tools:
        fn = t.get("function", t)
        decls.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return [{"functionDeclarations": decls}]


def _openai_tool_choice_to_google(
    tool_choice: dict[str, Any] | str | None,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if tool_choice == "none":
        return {"functionCallingConfig": {"mode": "NONE"}}
    if tool_choice == "required":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function", {})
        name = fn.get("name")
        if tool_choice.get("type") == "function" and name:
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    raise ProviderCallError(f"Unsupported tool_choice value for google: {tool_choice!r}")


def _google_response_format_config(response_format: dict[str, Any] | None) -> dict[str, Any]:
    if not response_format:
        return {}
    rtype = response_format.get("type")
    if rtype == "json_object":
        return {"response_mime_type": "application/json"}
    if rtype == "json_schema":
        cfg: dict[str, Any] = {"response_mime_type": "application/json"}
        schema = (response_format.get("json_schema") or {}).get("schema")
        if schema:
            cfg["response_schema"] = schema
        return cfg
    raise ProviderCallError(f"Unsupported response_format type for google: {rtype!r}")


def _parse_google_parts(
    parts: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]] | None]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for i, part in enumerate(parts):
        if "text" in part:
            text_parts.append(part["text"])
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_calls.append(
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                }
            )
    return "".join(text_parts), (tool_calls or None)


_GOOGLE_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
}


# ── Custom exception ─────────────────────────────────────────────────────────


class ProviderCallError(Exception):
    """Raised when a provider API call fails.  Carries http_status for routing."""

    def __init__(self, message: str, http_status: int | None = None) -> None:
        super().__init__(message)
        self.status_code = http_status  # matches FallbackExecutor expectation


# ── Low-level urllib helper ──────────────────────────────────────────────────

# Bugfix: urllib's default User-Agent ("Python-urllib/3.x") gets a blanket
# 403 from Groq's edge (Cloudflare bot-fight mode, error code 1010) — every
# real Flux->Groq call failed with this, misread as a key/permissions problem
# during live evaluation until isolated by comparing identical requests with
# and without a UA header. Anthropic/OpenAI/Google/Mistral don't enforce this,
# but a real UA is good practice for all providers, not just a Groq patch.
_USER_AGENT = "flux-router/1.0.0"


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
    headers = {**headers, "User-Agent": _USER_AGENT}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=PROVIDER_CALL_TIMEOUT_SECONDS) as resp:  # nosec B310 — scheme validated above
            try:
                decoded = _bounded_read(resp).decode("utf-8")
                parsed = json.loads(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProviderCallError(
                    f"Malformed response from {provider_name}", http_status=None
                ) from exc
            if not isinstance(parsed, dict):
                raise ProviderCallError(
                    f"Invalid response structure from {provider_name}", http_status=None
                )
            return parsed
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
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderCallError(f"Network error calling {provider_name}") from exc


# ── Provider-specific callers (synchronous) ──────────────────────────────────


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the universal OpenAI-shaped message list built by
    _build_messages() (assistant `tool_calls`, role="tool" + `tool_call_id`)
    into Anthropic's tool_use/tool_result content-block shape.

    Bugfix: _call_anthropic_sync() used to forward the OpenAI-shaped list to
    /v1/messages verbatim — Anthropic has no "tool" role and doesn't
    understand a `tool_calls` key on an assistant message, so any step after
    a real tool round trip got a genuine 400 from Anthropic. Confirmed live
    by the agentic-harness (see router/scripts/agentic_e2e_report.md).

    Anthropic represents a tool observation as a role="user" message
    containing a tool_result block, and requires every tool_result answering
    one assistant turn's tool_use block(s) to arrive together in a single
    user turn — so consecutive tool-role turns are merged into one message
    rather than sent as separate back-to-back user turns.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": m.get("content") or "",
            }
            prev = out[-1] if out else None
            if (
                prev is not None
                and prev["role"] == "user"
                and isinstance(prev["content"], list)
                and prev["content"]
                and prev["content"][0].get("type") == "tool_result"
            ):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            text = m.get("content")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                try:
                    tool_input = json.loads(fn.get("arguments") or "{}")
                except (json.JSONDecodeError, TypeError):
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": tool_input,
                    }
                )
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": role, "content": m.get("content", "")})
    return out


def _call_anthropic_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> ProviderResult:
    if request.response_format:
        # http_status=400: a caller error (unsupported provider/feature
        # combination), not an upstream failure — flux.py._call_model maps
        # this specific status to UnsupportedFeatureError so server.py can
        # return 400 instead of the 502 used for real provider outages.
        raise ProviderCallError(
            "Anthropic has no response_format equivalent; route this request "
            "to a different provider or drop response_format.",
            http_status=400,
        )

    messages = _openai_messages_to_anthropic(_build_messages(request))
    body: dict[str, Any] = {
        "model": model.provider_model_id,
        "max_tokens": min(request.max_tokens_requested or 1024, model.max_output_tokens),
        "messages": messages,
    }
    if request.system_prompt:
        body["system"] = request.system_prompt
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools:
        body["tools"] = _openai_tools_to_anthropic(request.tools)
        tool_choice = _openai_tool_choice_to_anthropic(request.tool_choice)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages", headers, body, provider_name="anthropic"
    )
    content = data.get("content")
    if not isinstance(content, list) or not content:
        raise ProviderCallError(
            f"Unexpected Anthropic response shape (keys={_response_shape_keys(data)})"
        )
    text, tool_calls = _parse_anthropic_content(content)
    stop_reason = data.get("stop_reason")
    if tool_calls:
        finish_reason = "tool_calls"
    elif stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = "stop"
    input_tokens, output_tokens = _extract_usage(
        data, "anthropic", "input_tokens", "output_tokens", container_key="usage"
    )
    return ProviderResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source="provider" if input_tokens is not None else "estimated",
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


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
) -> ProviderResult:
    """Shared caller for OpenAI, Groq, and Mistral (all use the same message format)."""
    messages = _build_messages(request)
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    body: dict[str, Any] = {
        "model": model.provider_model_id,
        "messages": messages,
    }
    token_limit = min(request.max_tokens_requested or 1024, model.max_output_tokens)
    if _uses_max_completion_tokens(provider_name, model.model_id):
        body["max_completion_tokens"] = token_limit
    else:
        body["max_tokens"] = token_limit
    if request.temperature is not None:
        body["temperature"] = request.temperature
    # Already OpenAI-shaped — passed through verbatim. An unsupported value
    # for this specific provider (e.g. a tool_choice enum Mistral doesn't
    # recognize) surfaces as a real HTTP error from the provider via
    # _post_json, not a silent drop.
    if request.tools:
        body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        body["response_format"] = request.response_format

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    data = _post_json(f"{base_url}/chat/completions", headers, body, provider_name=provider_name)
    try:
        choice = data["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(
            f"Unexpected OpenAI-compat response shape (keys={_response_shape_keys(data)})"
        ) from exc
    text = message.get("content") or ""
    tool_calls = message.get("tool_calls") or None
    finish_reason = choice.get("finish_reason") or "stop"
    # OpenAI, Groq, and Mistral all report usage as {"usage": {"prompt_tokens":
    # ..., "completion_tokens": ...}} on non-streaming responses.
    input_tokens, output_tokens = _extract_usage(
        data, provider_name, "prompt_tokens", "completion_tokens", container_key="usage"
    )
    return ProviderResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source="provider" if input_tokens is not None else "estimated",
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


def _call_google_sync(
    model: ModelOption,
    request: RoutingRequest,
    api_key: str,
) -> ProviderResult:
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
    generation_config: dict[str, Any] = {
        "maxOutputTokens": min(request.max_tokens_requested or 1024, model.max_output_tokens)
    }
    generation_config.update(_google_response_format_config(request.response_format))
    body["generationConfig"] = generation_config
    if request.tools:
        body["tools"] = _openai_tools_to_google(request.tools)
        tool_config = _openai_tool_choice_to_google(request.tool_choice)
        if tool_config is not None:
            body["toolConfig"] = tool_config

    model_id = (model.provider_model_id or model.model_id).replace(
        "-thinking", ""
    )  # strip suffix for API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    # Pass key as a header to avoid it appearing in server access logs.
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    data = _post_json(url, headers, body, provider_name="google")
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError) as exc:
        raise ProviderCallError(
            f"Unexpected Google response shape (keys={_response_shape_keys(data)})"
        ) from exc
    text, tool_calls = _parse_google_parts(parts)
    raw_finish = data["candidates"][0].get("finishReason")
    finish_reason = (
        "tool_calls" if tool_calls else _GOOGLE_FINISH_REASONS.get(raw_finish, "stop")
    )
    input_tokens, output_tokens = _extract_usage(
        data, "google", "promptTokenCount", "candidatesTokenCount", container_key="usageMetadata"
    )
    return ProviderResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_source="provider" if input_tokens is not None else "estimated",
        tool_calls=tool_calls,
        finish_reason=finish_reason,
    )


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
) -> Any:
    """Open a streaming POST to an OpenAI-compatible provider. Returns the open response.

    Synchronous — always called via loop.run_in_executor. The caller is
    responsible for closing the returned response object.
    """
    messages = _build_messages(request)
    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    body: dict[str, Any] = {
        "model": model.provider_model_id,
        "messages": messages,
        "stream": True,
    }
    token_limit = min(request.max_tokens_requested or 1024, model.max_output_tokens)
    if _uses_max_completion_tokens(provider_name, model.model_id):
        body["max_completion_tokens"] = token_limit
    else:
        body["max_tokens"] = token_limit
    if request.temperature is not None:
        body["temperature"] = request.temperature
    # tool_calls deltas ride through stream_openai_compat_lines verbatim once
    # the upstream body asks for tools — no chunk reshaping needed here.
    if request.tools:
        body["tools"] = request.tools
        if request.tool_choice is not None:
            body["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        body["response_format"] = request.response_format
    # OpenAI-only: asks the stream to emit one extra chunk right before
    # [DONE] carrying a `usage` object (see OpenAI's stream_options docs).
    # Not enabled for Groq/Mistral — their OpenAI-compat surface hasn't been
    # verified to tolerate this field, so they keep estimating usage for
    # streaming responses until that's confirmed.
    # TODO(groq/mistral stream usage): verify and enable per-provider.
    if provider_name == "openai":
        body["stream_options"] = {"include_usage": True}

    url = f"{base_url}/chat/completions"
    if not url.startswith("https://"):
        raise ProviderCallError("Invalid URL scheme; only https allowed", http_status=None)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": _USER_AGENT,
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
) -> AsyncIterator[bytes]:
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
) -> ProviderResult:
    """
    Async wrapper: dispatches to the correct provider caller in a thread-pool
    executor so the routing event loop is not blocked.

    Returns a ProviderResult carrying the response text plus, when the
    provider reported it, actual token usage (usage_source="provider").
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

        def fn() -> ProviderResult:
            return _call_anthropic_sync(model, request, api_key)
    elif provider in _OPENAI_COMPAT_BASES:
        base = _OPENAI_COMPAT_BASES[provider]

        def fn() -> ProviderResult:
            return _call_openai_compat_sync(model, request, api_key, base, provider)
    elif provider == "google":

        def fn() -> ProviderResult:
            return _call_google_sync(model, request, api_key)
    else:
        raise ProviderCallError(
            f"No provider caller implemented for '{provider}'.  "
            f"Supported: anthropic, openai, groq, mistral, google."
        )

    log.debug("proxy_call_start", provider=provider, model=model.model_id)
    return await loop.run_in_executor(None, fn)
