"""
File: router/tests/test_tool_calling.py

Purpose:
Tests for Item 1 — end-to-end agent tool calling. Covers:
  - Per-provider outgoing translation of tools/tool_choice/response_format
    (router/provider_caller.py), with realistic mocked provider payloads.
  - Per-provider parsing of tool_calls out of the provider's native response
    shape (Anthropic tool_use blocks, OpenAI-compat choices[].message.tool_calls,
    Google functionCall parts) back into ProviderResult's OpenAI-shaped
    tool_calls field.
  - The Anthropic + response_format reject path (no silent drop).
  - HTTP round-trip through /v1/chat/completions: tool_calls and finish_reason
    surface correctly in the OpenAI-compatible response body.

No real HTTP requests to any provider are made — all provider I/O is mocked
at _post_json (provider_caller unit tests) or _call_model (server HTTP tests),
matching the conventions in test_provider_caller_body.py and test_server.py.
"""

from __future__ import annotations

import pytest

from router import provider_caller
from router.errors import UnsupportedFeatureError
from router.provider_caller import ProviderCallError, ProviderResult
from router.schemas import ModelOption, RoutingRequest

pytest.importorskip("fastapi")

from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


def _model(provider: str, supports_tools: bool = True) -> ModelOption:
    return ModelOption(
        provider=provider,
        model_id=f"{provider}-test-model",
        display_name="test",
        tier="premium",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        max_context_window=128_000,
        max_output_tokens=4096,
        capabilities=["general"],
        supports_tools=supports_tools,
    )


def _request(**kwargs) -> RoutingRequest:
    kwargs.setdefault("raw_prompt", "What's the weather in Paris?")
    kwargs.setdefault("user_id", "u_test")
    return RoutingRequest(**kwargs)


@pytest.fixture
def captured(monkeypatch):
    box: dict = {}

    def fake_post(url, headers, body, provider_name):
        box["url"] = url
        box["body"] = body
        box["provider_name"] = provider_name
        return box["response"]

    monkeypatch.setattr(provider_caller, "_post_json", fake_post)
    return box


# ── Outgoing translation ──────────────────────────────────────────────────────


class TestOutgoingTranslation:
    def test_openai_compat_passes_tools_verbatim(self, captured):
        captured["response"] = {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]
        }
        provider_caller._call_openai_compat_sync(
            _model("openai"),
            _request(tools=[_WEATHER_TOOL], tool_choice="auto"),
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            provider_name="openai",
        )
        body = captured["body"]
        assert body["tools"] == [_WEATHER_TOOL]
        assert body["tool_choice"] == "auto"

    def test_openai_compat_passes_response_format_verbatim(self, captured):
        captured["response"] = {
            "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}]
        }
        rf = {"type": "json_object"}
        provider_caller._call_openai_compat_sync(
            _model("mistral"),
            _request(response_format=rf),
            api_key="k",
            base_url="https://api.mistral.ai/v1",
            provider_name="mistral",
        )
        assert captured["body"]["response_format"] == rf

    def test_anthropic_translates_tools_to_input_schema(self, captured):
        captured["response"] = {"content": [{"type": "text", "text": "ok"}]}
        provider_caller._call_anthropic_sync(
            _model("anthropic"), _request(tools=[_WEATHER_TOOL]), api_key="k"
        )
        body = captured["body"]
        assert body["tools"] == [
            {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "input_schema": _WEATHER_TOOL["function"]["parameters"],
            }
        ]

    def test_anthropic_translates_tool_choice_specific_function(self, captured):
        captured["response"] = {"content": [{"type": "text", "text": "ok"}]}
        provider_caller._call_anthropic_sync(
            _model("anthropic"),
            _request(
                tools=[_WEATHER_TOOL],
                tool_choice={"type": "function", "function": {"name": "get_weather"}},
            ),
            api_key="k",
        )
        assert captured["body"]["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_anthropic_rejects_response_format(self):
        with pytest.raises(ProviderCallError) as exc_info:
            provider_caller._call_anthropic_sync(
                _model("anthropic"),
                _request(response_format={"type": "json_object"}),
                api_key="k",
            )
        assert exc_info.value.status_code == 400
        assert "response_format" in str(exc_info.value)

    def test_google_translates_tools_to_function_declarations(self, captured):
        captured["response"] = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
        }
        provider_caller._call_google_sync(
            _model("google"), _request(tools=[_WEATHER_TOOL]), api_key="k"
        )
        body = captured["body"]
        assert body["tools"] == [
            {
                "functionDeclarations": [
                    {
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": _WEATHER_TOOL["function"]["parameters"],
                    }
                ]
            }
        ]

    def test_google_translates_response_format_json_object(self, captured):
        captured["response"] = {
            "candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]
        }
        provider_caller._call_google_sync(
            _model("google"), _request(response_format={"type": "json_object"}), api_key="k"
        )
        assert captured["body"]["generationConfig"]["response_mime_type"] == "application/json"

    def test_google_tool_choice_required_maps_to_any_mode(self, captured):
        captured["response"] = {
            "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]
        }
        provider_caller._call_google_sync(
            _model("google"),
            _request(tools=[_WEATHER_TOOL], tool_choice="required"),
            api_key="k",
        )
        assert captured["body"]["toolConfig"] == {"functionCallingConfig": {"mode": "ANY"}}


# ── Incoming response parsing ─────────────────────────────────────────────────


class TestIncomingParsing:
    def test_openai_compat_tool_calls_response(self, captured):
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ]
        captured["response"] = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": None, "tool_calls": tool_calls},
                    "finish_reason": "tool_calls",
                }
            ]
        }
        result = provider_caller._call_openai_compat_sync(
            _model("openai"),
            _request(tools=[_WEATHER_TOOL]),
            api_key="k",
            base_url="https://api.openai.com/v1",
            provider_name="openai",
        )
        assert result.tool_calls == tool_calls
        assert result.finish_reason == "tool_calls"
        assert result.text == ""  # content was None, not dropped/crashed

    def test_anthropic_tool_use_block_parsed(self, captured):
        captured["response"] = {
            "content": [
                {"type": "text", "text": "Let me check that."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
        }
        result = provider_caller._call_anthropic_sync(
            _model("anthropic"), _request(tools=[_WEATHER_TOOL]), api_key="k"
        )
        assert result.text == "Let me check that."
        assert result.tool_calls == [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ]
        assert result.finish_reason == "tool_calls"

    def test_google_function_call_part_parsed(self, captured):
        captured["response"] = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "Paris"},
                                }
                            }
                        ]
                    },
                    "finishReason": "STOP",
                }
            ]
        }
        result = provider_caller._call_google_sync(
            _model("google"), _request(tools=[_WEATHER_TOOL]), api_key="k"
        )
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.tool_calls[0]["function"]["arguments"] == '{"location": "Paris"}'


# ── HTTP round-trip through /v1/chat/completions ──────────────────────────────


def _pr_with_tool_call() -> ProviderResult:
    return ProviderResult(
        text="",
        input_tokens=None,
        output_tokens=None,
        usage_source="estimated",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
            }
        ],
        finish_reason="tool_calls",
    )


@pytest.fixture
def client():
    return TestClient(server.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8000")


def _tool_body(model: str = "flux-auto") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
        "tools": [_WEATHER_TOOL],
        "stream": False,
    }


class TestHttpToolCallRoundTrip:
    def test_tool_call_surfaces_in_response(self, client, monkeypatch):
        mock = AsyncMock(return_value=_pr_with_tool_call())
        monkeypatch.setattr(server._flux, "_call_model", mock)
        resp = client.post("/v1/chat/completions", json=_tool_body())
        assert resp.status_code == 200
        data = resp.json()
        choice = data["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_no_tool_call_keeps_finish_reason_stop(self, client, monkeypatch):
        mock = AsyncMock(
            return_value=ProviderResult(
                text="hi", input_tokens=None, output_tokens=None, usage_source="estimated"
            )
        )
        monkeypatch.setattr(server._flux, "_call_model", mock)
        resp = client.post("/v1/chat/completions", json=_tool_body())
        assert resp.status_code == 200
        choice = resp.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert "tool_calls" not in choice["message"]

    def test_unsupported_feature_combo_returns_400(self, client, monkeypatch):
        mock = AsyncMock(side_effect=UnsupportedFeatureError("Anthropic has no response_format"))
        monkeypatch.setattr(server._flux, "_call_model", mock)
        body = _tool_body("claude-haiku-4-5-20251001")
        body["response_format"] = {"type": "json_object"}
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 400

    def test_tool_choice_and_history_round_trip(self, client, monkeypatch):
        captured = {}
        real_route = server._flux.route

        async def spy_route(request, verbose=False):
            captured["tools"] = request.tools
            captured["tool_choice"] = request.tool_choice
            return await real_route(request, verbose=verbose)

        monkeypatch.setattr(server._flux, "route", spy_route)
        mock = AsyncMock(
            return_value=ProviderResult(
                text="ok", input_tokens=None, output_tokens=None, usage_source="estimated"
            )
        )
        monkeypatch.setattr(server._flux, "_call_model", mock)
        body = _tool_body()
        body["tool_choice"] = "auto"
        body["messages"] = [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "sunny"},
            {"role": "user", "content": "and tomorrow?"},
        ]
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert captured["tools"] == [_WEATHER_TOOL]
        assert captured["tool_choice"] == "auto"
