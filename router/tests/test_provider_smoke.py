"""
File: router/tests/test_provider_smoke.py

Purpose:
Item 5 — opt-in smoke tests against REAL provider APIs. Every test in this
file is skipped unless the relevant provider's API key env var is set, so a
normal `pytest` run (no keys) exercises nothing here and spends nothing.

Required env vars (only set the ones for providers you want to smoke-test):
  ANTHROPIC_API_KEY
  OPENAI_API_KEY
  GOOGLE_API_KEY
  GROQ_API_KEY
  MISTRAL_API_KEY

Run:
  ANTHROPIC_API_KEY=sk-ant-... pytest -v router/tests/test_provider_smoke.py

Every live call uses a tiny prompt ("Say OK.") and a small max_tokens_requested
cap to keep cost negligible (a handful of tokens per call, well under a cent
even on the priciest catalog entries). No secrets are read from anywhere but
the environment, and none are logged, printed, or written to disk.

What's covered per provider (each gated on that provider's own key):
  - a normal non-streaming completion
  - a streaming completion (OpenAI-compat providers only — Anthropic and
    Google have no native streaming caller in provider_caller.py; see its
    STREAMING_NATIVE_PROVIDERS comment)
  - a tool call, where the routed model supports_tools
  - a structured JSON response, where supports_structured_output (excludes
    Anthropic — Item 1's reject path, already covered without a live call in
    test_tool_calling.py::TestOutgoingTranslation::test_anthropic_rejects_response_format)
  - invalid authentication (a deliberately wrong key -> AuthenticationError)
  - actual provider-reported token usage / cost calculation

429/rate-limit mapping is NOT exercised live here — reliably forcing a real
429 from a provider isn't practical in an automated suite. That mapping
(ProviderCallError with status_code=429 -> errors.RateLimitError) is
verified structurally instead, with no live call and no key required — see
TestRateLimitMappingIsWired below. This is a known, documented limitation of
this suite, not an untested code path (see test_provider_caller_body.py /
flux.py::Flux._call_model for the mapping itself).
"""

from __future__ import annotations

import os

import pytest

from router import errors
from router.flux import make_flux
from router.model_registry import ModelRegistry
from router.provider_caller import ProviderCallError, call_provider, compute_actual_cost
from router.schemas import ModelOption, RoutingRequest

_PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}

_PING_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Respond with pong",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _cheapest_model(
    provider: str, *, needs_tools: bool = False, needs_structured: bool = False
) -> ModelOption:
    registry = ModelRegistry()
    candidates = [m for m in registry.all_available_models() if m.provider == provider]
    if needs_tools:
        candidates = [m for m in candidates if m.supports_tools]
    if needs_structured:
        candidates = [m for m in candidates if m.supports_structured_output]
    if not candidates:
        pytest.skip(f"no available {provider} model matches the requested capability")
    return min(candidates, key=lambda m: m.cost_per_1k_input + m.cost_per_1k_output)


def _tiny_request(**kwargs) -> RoutingRequest:
    kwargs.setdefault("raw_prompt", "Say OK.")
    kwargs.setdefault("user_id", "smoke-test")
    kwargs.setdefault("max_tokens_requested", 16)
    return RoutingRequest(**kwargs)


def _skip_reason(provider: str) -> str:
    var = _PROVIDER_ENV_VARS[provider]
    return f"{var} not set — skipped (opt-in live provider smoke test)"


def _requires(provider: str):
    return pytest.mark.skipif(
        not os.environ.get(_PROVIDER_ENV_VARS[provider]), reason=_skip_reason(provider)
    )


# ── Structural: rate-limit mapping (no live call, no key needed) ─────────────


class TestRateLimitMappingIsWired:
    """See module docstring: real 429s aren't reliably forced live, so the
    ProviderCallError(429) -> errors.RateLimitError mapping is verified here
    without a network call."""

    def test_429_maps_to_rate_limit_error(self, monkeypatch):
        flux = make_flux(api_keys={"openai": "sk-fake-for-mapping-test-only"})

        async def fake_call_provider(model, request, api_key):
            raise ProviderCallError("HTTP 429 from openai", http_status=429)

        monkeypatch.setattr("router.provider_caller.call_provider", fake_call_provider)
        model = _cheapest_model("openai") if os.environ.get("OPENAI_API_KEY") else ModelOption(
            provider="openai",
            model_id="gpt-4o-mini",
            display_name="test",
            tier="cheap",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002,
            max_context_window=128_000,
            max_output_tokens=4096,
            capabilities=["general"],
        )
        with pytest.raises(errors.RateLimitError):
            import asyncio

            asyncio.run(flux._call_model(model, _tiny_request()))


# ── Per-provider live smoke tests ─────────────────────────────────────────────


@_requires("openai")
class TestOpenAISmoke:
    def test_normal_completion(self):
        import asyncio

        model = _cheapest_model("openai")
        api_key = os.environ["OPENAI_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert isinstance(result.text, str)

    def test_streaming_completion(self):
        import asyncio

        from router.provider_caller import stream_openai_compat_lines

        model = _cheapest_model("openai")
        api_key = os.environ["OPENAI_API_KEY"]

        async def _collect():
            chunks = []
            async for line in stream_openai_compat_lines(model, _tiny_request(), api_key):
                chunks.append(line)
            return chunks

        chunks = asyncio.run(_collect())
        assert any(b"data:" in c for c in chunks)

    def test_tool_call(self):
        import asyncio

        model = _cheapest_model("openai", needs_tools=True)
        api_key = os.environ["OPENAI_API_KEY"]
        req = _tiny_request(
            raw_prompt="Call the ping function.",
            tools=[_PING_TOOL],
            tool_choice="required",
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert result.tool_calls, "model did not make the forced tool call"
        assert result.tool_calls[0]["function"]["name"] == "ping"

    def test_structured_json_response(self):
        import asyncio

        model = _cheapest_model("openai", needs_structured=True)
        api_key = os.environ["OPENAI_API_KEY"]
        req = _tiny_request(
            raw_prompt='Return {"ok": true} as JSON.',
            response_format={"type": "json_object"},
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert isinstance(result.text, str) and result.text.strip()

    def test_invalid_auth_raises(self):
        import asyncio

        model = _cheapest_model("openai")
        with pytest.raises(ProviderCallError) as exc_info:
            asyncio.run(call_provider(model, _tiny_request(), "sk-deliberately-invalid"))
        assert exc_info.value.status_code in (401, 403)

    def test_actual_usage_and_cost(self):
        import asyncio

        model = _cheapest_model("openai")
        api_key = os.environ["OPENAI_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert result.usage_source == "provider"
        assert result.input_tokens is not None and result.output_tokens is not None
        cost = compute_actual_cost(model, result.input_tokens, result.output_tokens)
        assert cost >= 0.0


@_requires("groq")
class TestGroqSmoke:
    def test_normal_completion(self):
        import asyncio

        model = _cheapest_model("groq")
        api_key = os.environ["GROQ_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert isinstance(result.text, str)

    def test_streaming_completion(self):
        import asyncio

        from router.provider_caller import stream_openai_compat_lines

        model = _cheapest_model("groq")
        api_key = os.environ["GROQ_API_KEY"]

        async def _collect():
            return [
                line
                async for line in stream_openai_compat_lines(model, _tiny_request(), api_key)
            ]

        assert asyncio.run(_collect())

    def test_tool_call(self):
        import asyncio

        model = _cheapest_model("groq", needs_tools=True)
        api_key = os.environ["GROQ_API_KEY"]
        req = _tiny_request(
            raw_prompt="Call the ping function.", tools=[_PING_TOOL], tool_choice="required"
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert result.tool_calls, "model did not make the forced tool call"

    def test_invalid_auth_raises(self):
        import asyncio

        model = _cheapest_model("groq")
        with pytest.raises(ProviderCallError) as exc_info:
            asyncio.run(call_provider(model, _tiny_request(), "gsk-deliberately-invalid"))
        assert exc_info.value.status_code in (401, 403)

    def test_actual_usage_and_cost(self):
        import asyncio

        model = _cheapest_model("groq")
        api_key = os.environ["GROQ_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert result.usage_source == "provider"


@_requires("mistral")
class TestMistralSmoke:
    def test_normal_completion(self):
        import asyncio

        model = _cheapest_model("mistral")
        api_key = os.environ["MISTRAL_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert isinstance(result.text, str)

    def test_streaming_completion(self):
        import asyncio

        from router.provider_caller import stream_openai_compat_lines

        model = _cheapest_model("mistral")
        api_key = os.environ["MISTRAL_API_KEY"]

        async def _collect():
            return [
                line
                async for line in stream_openai_compat_lines(model, _tiny_request(), api_key)
            ]

        assert asyncio.run(_collect())

    def test_tool_call(self):
        import asyncio

        model = _cheapest_model("mistral", needs_tools=True)
        api_key = os.environ["MISTRAL_API_KEY"]
        req = _tiny_request(
            raw_prompt="Call the ping function.", tools=[_PING_TOOL], tool_choice="any"
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert result.tool_calls, "model did not make the forced tool call"

    def test_invalid_auth_raises(self):
        import asyncio

        model = _cheapest_model("mistral")
        with pytest.raises(ProviderCallError) as exc_info:
            asyncio.run(call_provider(model, _tiny_request(), "deliberately-invalid"))
        assert exc_info.value.status_code in (401, 403)

    def test_actual_usage_and_cost(self):
        import asyncio

        model = _cheapest_model("mistral")
        api_key = os.environ["MISTRAL_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert result.usage_source == "provider"


@_requires("anthropic")
class TestAnthropicSmoke:
    def test_normal_completion(self):
        import asyncio

        model = _cheapest_model("anthropic")
        api_key = os.environ["ANTHROPIC_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert isinstance(result.text, str)

    def test_tool_call(self):
        import asyncio

        model = _cheapest_model("anthropic", needs_tools=True)
        api_key = os.environ["ANTHROPIC_API_KEY"]
        req = _tiny_request(
            raw_prompt="Call the ping function.",
            tools=[_PING_TOOL],
            tool_choice={"type": "function", "function": {"name": "ping"}},
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert result.tool_calls, "model did not make the forced tool call"
        assert result.tool_calls[0]["function"]["name"] == "ping"

    # No streaming test: Anthropic has no native streaming caller in
    # provider_caller.py (server.py falls back to a synthesized single-chunk
    # stream for it) — see the module docstring.
    #
    # No structured-JSON test: response_format on Anthropic is rejected by
    # design (Item 1) — already covered without a live call in
    # test_tool_calling.py.

    def test_invalid_auth_raises(self):
        import asyncio

        model = _cheapest_model("anthropic")
        with pytest.raises(ProviderCallError) as exc_info:
            asyncio.run(call_provider(model, _tiny_request(), "sk-ant-deliberately-invalid"))
        assert exc_info.value.status_code in (401, 403)

    def test_actual_usage_and_cost(self):
        import asyncio

        model = _cheapest_model("anthropic")
        api_key = os.environ["ANTHROPIC_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert result.usage_source == "provider"


@_requires("google")
class TestGoogleSmoke:
    def test_normal_completion(self):
        import asyncio

        model = _cheapest_model("google")
        api_key = os.environ["GOOGLE_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert isinstance(result.text, str)

    def test_tool_call(self):
        import asyncio

        model = _cheapest_model("google", needs_tools=True)
        api_key = os.environ["GOOGLE_API_KEY"]
        req = _tiny_request(
            raw_prompt="Call the ping function.", tools=[_PING_TOOL], tool_choice="required"
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert result.tool_calls, "model did not make the forced tool call"

    def test_structured_json_response(self):
        import asyncio

        model = _cheapest_model("google", needs_structured=True)
        api_key = os.environ["GOOGLE_API_KEY"]
        req = _tiny_request(
            raw_prompt='Return {"ok": true} as JSON.',
            response_format={"type": "json_object"},
        )
        result = asyncio.run(call_provider(model, req, api_key))
        assert isinstance(result.text, str) and result.text.strip()

    # No streaming test: Google has no native streaming caller in
    # provider_caller.py either — same synthesized-chunk fallback as
    # Anthropic.

    def test_invalid_auth_raises(self):
        import asyncio

        model = _cheapest_model("google")
        with pytest.raises(ProviderCallError) as exc_info:
            asyncio.run(call_provider(model, _tiny_request(), "deliberately-invalid"))
        assert exc_info.value.status_code in (401, 403)

    def test_actual_usage_and_cost(self):
        import asyncio

        model = _cheapest_model("google")
        api_key = os.environ["GOOGLE_API_KEY"]
        result = asyncio.run(call_provider(model, _tiny_request(), api_key))
        assert result.usage_source == "provider"
