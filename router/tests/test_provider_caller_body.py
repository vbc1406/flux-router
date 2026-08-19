"""
Verify OpenAI-compatible request body generation for token-limit field selection.

OpenAI's reasoning models (o1/o3/o4) and the GPT-5 family require
`max_completion_tokens` and reject `max_tokens`. Groq and Mistral continue
to expect `max_tokens`.
"""

from __future__ import annotations

import asyncio
import io

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


def test_post_json_sends_a_user_agent(monkeypatch):
    """Regression test: _post_json used to send no User-Agent header, which
    left every outbound call using urllib's default ("Python-urllib/3.x").
    Groq's edge (Cloudflare bot-fight mode) blocks that default with a flat
    403 ("error code: 1010") regardless of key validity — reproduced live
    and confirmed fixed by adding a real User-Agent. Assert it's present and
    not the urllib default on every call, not just Groq's, since this is a
    general robustness fix."""
    captured_request: dict = {}

    def fake_urlopen(req, timeout=None):
        captured_request["headers"] = dict(req.header_items())
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(provider_caller.urllib.request, "urlopen", fake_urlopen)
    provider_caller._post_json(
        "https://provider.test/v1/chat/completions",
        {"Authorization": "Bearer test"},
        {"model": "test"},
        "test-provider",
    )
    ua = captured_request["headers"].get("User-agent")
    assert ua is not None
    assert "python-urllib" not in ua.lower()


def test_streaming_call_sends_a_user_agent(monkeypatch):
    """Same regression as above, for the separate header dict built in
    stream_openai_compat_lines — it doesn't go through _post_json, so it
    needed (and now has) its own User-Agent."""
    captured_request: dict = {}

    def fake_urlopen(req, timeout=None):
        captured_request["headers"] = dict(req.header_items())
        return io.BytesIO(b"data: [DONE]\n")

    monkeypatch.setattr(provider_caller.urllib.request, "urlopen", fake_urlopen)
    model = _model("groq", "gpt-oss-20b")

    async def _consume():
        gen = provider_caller.stream_openai_compat_lines(model, _request(), "test-key")
        try:
            await gen.__anext__()
        finally:
            await gen.aclose()

    asyncio.run(_consume())
    ua = captured_request["headers"].get("User-agent")
    assert ua is not None
    assert "python-urllib" not in ua.lower()


@pytest.mark.parametrize("payload", [b"\xff\xfe", b"not-json", b"[]"])
def test_post_json_normalizes_malformed_success_responses(monkeypatch, payload):
    monkeypatch.setattr(
        provider_caller.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(payload),
    )

    with pytest.raises(provider_caller.ProviderCallError, match="response"):
        provider_caller._post_json(
            "https://provider.test/v1/chat/completions",
            {"Authorization": "Bearer test"},
            {"model": "test"},
            "test-provider",
        )


# ── Item 2: provider_model_id — the literal string sent upstream ─────────────


def test_provider_model_id_defaults_to_model_id():
    m = _model("openai", "gpt-5-mini")
    assert m.provider_model_id == "gpt-5-mini"


def test_provider_model_id_override_used_when_set():
    m = ModelOption(
        provider="groq",
        model_id="gpt-oss-120b",
        provider_model_id="openai/gpt-oss-120b",
        display_name="GPT-OSS 120B",
        tier="mid",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        max_context_window=131_072,
        max_output_tokens=32_768,
        capabilities=[],
    )
    assert m.model_id == "gpt-oss-120b"
    assert m.provider_model_id == "openai/gpt-oss-120b"


@pytest.mark.parametrize(
    "model_id,expected_upstream",
    [
        ("gpt-oss-120b", "openai/gpt-oss-120b"),
        ("gpt-oss-20b", "openai/gpt-oss-20b"),
        ("qwen-3.6-27b", "qwen/qwen3.6-27b"),
    ],
)
def test_groq_catalog_entries_send_namespaced_upstream_id(captured, model_id, expected_upstream):
    """The three Groq entries with a real models.json provider_model_id
    override must send THAT string as "model" upstream, not Flux's internal
    model_id — this is the exact bug Item 2 fixes."""
    from router.model_registry import ModelRegistry

    registry = ModelRegistry()
    model = registry.get_model(model_id)
    assert model is not None, f"{model_id} missing from the loaded catalog"
    assert model.provider_model_id == expected_upstream

    provider_caller._call_openai_compat_sync(
        model,
        _request(),
        api_key="gsk-test",
        base_url="https://api.groq.com/openai/v1",
        provider_name="groq",
    )
    assert captured["body"]["model"] == expected_upstream


def test_every_catalog_entry_sends_its_provider_model_id_upstream(monkeypatch):
    """Loop over the whole loaded registry: whatever body/URL a provider's
    caller builds, the upstream model string must equal provider_model_id
    (== model_id when no override is set in models.json)."""
    from router.model_registry import ModelRegistry

    box: dict = {}

    def fake_post_anthropic(url, headers, body, provider_name):
        box["model"] = body["model"]
        return {"content": [{"text": "ok"}]}

    def fake_post_google(url, headers, body, provider_name):
        box["url"] = url
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}

    def fake_post_compat(url, headers, body, provider_name):
        box["model"] = body["model"]
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}

    registry = ModelRegistry()
    for model in registry.all_available_models():
        expected = model.provider_model_id or model.model_id
        if model.provider == "anthropic":
            monkeypatch.setattr(provider_caller, "_post_json", fake_post_anthropic)
            provider_caller._call_anthropic_sync(model, _request(), api_key="k")
            assert box["model"] == expected
        elif model.provider == "google":
            monkeypatch.setattr(provider_caller, "_post_json", fake_post_google)
            provider_caller._call_google_sync(model, _request(), api_key="k")
            assert f"/models/{expected.replace('-thinking', '')}:" in box["url"]
        else:
            monkeypatch.setattr(provider_caller, "_post_json", fake_post_compat)
            base = provider_caller._OPENAI_COMPAT_BASES[model.provider]
            provider_caller._call_openai_compat_sync(
                model, _request(), api_key="k", base_url=base, provider_name=model.provider
            )
            assert box["model"] == expected
