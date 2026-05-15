"""
Verify OpenAI-compatible request body generation for token-limit field selection.

OpenAI's reasoning models (o1/o3/o4) and the GPT-5 family require
`max_completion_tokens` and reject `max_tokens`. Groq and Mistral continue
to expect `max_tokens`.
"""

from __future__ import annotations

import pytest

from router import provider_caller
from router.schemas import ModelOption, RoutingRequest


def _model(provider: str, model_id: str) -> ModelOption:
    return ModelOption(
        provider=provider,
        model_id=model_id,
        display_name=model_id,
        tier="premium",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.002,
        max_context_window=128_000,
        max_output_tokens=4096,
        capabilities=["general"],
    )


def _request() -> RoutingRequest:
    return RoutingRequest(
        raw_prompt="hello",
        user_id="u_test",
        plan="pro_plan",
        max_tokens_requested=256,
    )


@pytest.fixture
def captured(monkeypatch):
    box: dict = {}

    def fake_post(url, headers, body, provider_name):
        box["url"] = url
        box["body"] = body
        box["provider_name"] = provider_name
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(provider_caller, "_post_json", fake_post)
    return box


@pytest.mark.parametrize(
    "model_id",
    ["o3", "o4-mini", "o1-mini", "gpt-5", "gpt-5-mini", "gpt-5.5"],
)
def test_openai_reasoning_models_use_max_completion_tokens(captured, model_id):
    provider_caller._call_openai_compat_sync(
        _model("openai", model_id),
        _request(),
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        provider_name="openai",
    )
    body = captured["body"]
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body


@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
def test_openai_legacy_models_use_max_tokens(captured, model_id):
    provider_caller._call_openai_compat_sync(
        _model("openai", model_id),
        _request(),
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        provider_name="openai",
    )
    body = captured["body"]
    assert body["max_tokens"] == 256
    assert "max_completion_tokens" not in body


def test_groq_uses_max_tokens(captured):
    provider_caller._call_openai_compat_sync(
        _model("groq", "llama-3.3-70b"),
        _request(),
        api_key="gsk-test",
        base_url="https://api.groq.com/openai/v1",
        provider_name="groq",
    )
    body = captured["body"]
    assert body["max_tokens"] == 256
    assert "max_completion_tokens" not in body


def test_mistral_uses_max_tokens(captured):
    provider_caller._call_openai_compat_sync(
        _model("mistral", "mistral-large-latest"),
        _request(),
        api_key="mst-test",
        base_url="https://api.mistral.ai/v1",
        provider_name="mistral",
    )
    body = captured["body"]
    assert body["max_tokens"] == 256
    assert "max_completion_tokens" not in body
