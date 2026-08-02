"""
File: router/tests/test_provider_usage.py

Purpose:
Tests for actual (provider-reported) usage extraction and cost computation —
provider_caller.py's ProviderResult, _safe_usage_int, _extract_usage, and
compute_actual_cost. Mocks _post_json (same pattern as
test_provider_caller_body.py) so no real HTTP calls are made.

How to run:
  pytest -v router/tests/test_provider_usage.py
"""

from __future__ import annotations

import pytest

from router import provider_caller
from router.provider_caller import ProviderResult, compute_actual_cost
from router.schemas import ModelOption, RoutingRequest


def _model(provider: str = "openai", **overrides) -> ModelOption:
    defaults = {
        "provider": provider,
        "model_id": "test-model",
        "display_name": "Test Model",
        "tier": "premium",
        "cost_per_1k_input": 0.01,
        "cost_per_1k_output": 0.03,
        "max_context_window": 128_000,
        "max_output_tokens": 4096,
        "capabilities": ["general"],
    }
    defaults.update(overrides)
    return ModelOption(**defaults)


def _request() -> RoutingRequest:
    return RoutingRequest(raw_prompt="hello", user_id="u_test", plan="pro_plan")


@pytest.fixture
def fake_post(monkeypatch):
    """Patch _post_json to return a canned response dict, regardless of args."""
    box: dict = {}

    def _install(response: dict):
        def fake(url, headers, body, provider_name):
            box["provider_name"] = provider_name
            return response

        monkeypatch.setattr(provider_caller, "_post_json", fake)
        return box

    return _install


# ── _safe_usage_int ──────────────────────────────────────────────────────────


class TestSafeUsageInt:
    def test_positive_int_passes_through(self):
        assert provider_caller._safe_usage_int(42) == 42

    def test_positive_float_coerced_to_int(self):
        assert provider_caller._safe_usage_int(42.9) == 42

    def test_zero_treated_as_missing(self):
        assert provider_caller._safe_usage_int(0) is None

    def test_negative_treated_as_missing(self):
        assert provider_caller._safe_usage_int(-5) is None

    def test_none_treated_as_missing(self):
        assert provider_caller._safe_usage_int(None) is None

    def test_string_treated_as_missing(self):
        assert provider_caller._safe_usage_int("100") is None

    def test_bool_treated_as_missing(self):
        # bool is a subclass of int in Python — must not silently pass as 1/0 tokens.
        assert provider_caller._safe_usage_int(True) is None
        assert provider_caller._safe_usage_int(False) is None

    def test_list_treated_as_missing(self):
        assert provider_caller._safe_usage_int([100]) is None


# ── _extract_usage ────────────────────────────────────────────────────────────


class TestExtractUsage:
    def test_valid_usage_extracted(self):
        data = {"usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (10, 20)

    def test_missing_container_key_returns_none_none(self):
        data = {"choices": []}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (None, None)

    def test_container_not_a_dict_returns_none_none(self):
        data = {"usage": "not-a-dict"}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (None, None)

    def test_response_not_a_dict_returns_none_none(self):
        result = provider_caller._extract_usage(
            ["not", "a", "dict"],
            "openai",
            "prompt_tokens",
            "completion_tokens",
            container_key="usage",
        )
        assert result == (None, None)

    def test_one_field_missing_returns_none_none(self):
        """Partial usage (only one of the two fields) is not trustworthy —
        must not silently bill from half the data."""
        data = {"usage": {"prompt_tokens": 10}}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (None, None)

    def test_zero_value_field_returns_none_none(self):
        data = {"usage": {"prompt_tokens": 0, "completion_tokens": 5}}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (None, None)

    def test_negative_value_field_returns_none_none(self):
        data = {"usage": {"prompt_tokens": -1, "completion_tokens": 5}}
        result = provider_caller._extract_usage(
            data, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
        )
        assert result == (None, None)

    def test_never_raises_on_malformed_shape(self):
        """A missing/malformed usage shape must never turn a successful
        completion into an error."""
        for bad in [None, 42, "string", [], {"usage": None}, {"usage": []}]:
            result = provider_caller._extract_usage(
                bad, "openai", "prompt_tokens", "completion_tokens", container_key="usage"
            )
            assert result == (None, None)


# ── compute_actual_cost ───────────────────────────────────────────────────────


class TestComputeActualCost:
    def test_basic_computation(self):
        model = _model(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
        cost = compute_actual_cost(model, input_tokens=1000, output_tokens=1000)
        assert cost == pytest.approx(0.04)

    def test_rounds_to_six_decimals(self):
        model = _model(cost_per_1k_input=0.0001, cost_per_1k_output=0.0002)
        cost = compute_actual_cost(model, input_tokens=1, output_tokens=1)
        assert cost == round(cost, 6)

    def test_zero_tokens_zero_cost(self):
        model = _model()
        assert compute_actual_cost(model, input_tokens=0, output_tokens=0) == 0.0


# ── Per-provider ProviderResult extraction (via _post_json mock) ─────────────


class TestOpenAICompatUsageExtraction:
    def test_usage_present_yields_provider_source(self, fake_post):
        fake_post(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8},
            }
        )
        result = provider_caller._call_openai_compat_sync(
            _model("openai"),
            _request(),
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            provider_name="openai",
        )
        assert isinstance(result, ProviderResult)
        assert result.text == "hi"
        assert result.input_tokens == 12
        assert result.output_tokens == 8
        assert result.usage_source == "provider"

    def test_missing_usage_falls_back_to_estimated(self, fake_post):
        fake_post({"choices": [{"message": {"content": "hi"}}]})
        result = provider_caller._call_openai_compat_sync(
            _model("groq"),
            _request(),
            api_key="gsk-test",
            base_url="https://api.groq.com/openai/v1",
            provider_name="groq",
        )
        assert result.text == "hi"
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.usage_source == "estimated"

    def test_zero_usage_falls_back_to_estimated(self, fake_post):
        fake_post(
            {
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        )
        result = provider_caller._call_openai_compat_sync(
            _model("openai"),
            _request(),
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            provider_name="openai",
        )
        assert result.usage_source == "estimated"


class TestAnthropicUsageExtraction:
    def test_usage_present_yields_provider_source(self, fake_post):
        fake_post(
            {
                "content": [{"text": "hi"}],
                "usage": {"input_tokens": 15, "output_tokens": 9},
            }
        )
        result = provider_caller._call_anthropic_sync(
            _model("anthropic"), _request(), api_key="sk-ant-test"
        )
        assert result.text == "hi"
        assert result.input_tokens == 15
        assert result.output_tokens == 9
        assert result.usage_source == "provider"

    def test_missing_usage_falls_back_to_estimated(self, fake_post):
        fake_post({"content": [{"text": "hi"}]})
        result = provider_caller._call_anthropic_sync(
            _model("anthropic"), _request(), api_key="sk-ant-test"
        )
        assert result.input_tokens is None
        assert result.usage_source == "estimated"


class TestGoogleUsageExtraction:
    def test_usage_present_yields_provider_source(self, fake_post):
        fake_post(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 5},
            }
        )
        result = provider_caller._call_google_sync(
            _model("google"), _request(), api_key="goog-test"
        )
        assert result.text == "hi"
        assert result.input_tokens == 20
        assert result.output_tokens == 5
        assert result.usage_source == "provider"

    def test_missing_usage_falls_back_to_estimated(self, fake_post):
        fake_post({"candidates": [{"content": {"parts": [{"text": "hi"}]}}]})
        result = provider_caller._call_google_sync(
            _model("google"), _request(), api_key="goog-test"
        )
        assert result.input_tokens is None
        assert result.usage_source == "estimated"


# ── stream_options.include_usage: OpenAI only ─────────────────────────────────


class TestStreamOptionsIncludeUsage:
    def _capture_stream_body(self, monkeypatch, model, provider_name):
        captured: dict = {}

        class FakeResponse:
            def close(self):
                pass

        def fake_urlopen(req, timeout=None):
            import json as _json

            captured["body"] = _json.loads(req.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr(provider_caller.urllib.request, "urlopen", fake_urlopen)
        provider_caller._open_openai_compat_stream(
            model,
            _request(),
            api_key="test-key",
            base_url=f"https://api.{provider_name}.example/v1",
            provider_name=provider_name,
        )
        return captured["body"]

    def test_openai_requests_include_usage(self, monkeypatch):
        body = self._capture_stream_body(monkeypatch, _model("openai"), "openai")
        assert body.get("stream_options") == {"include_usage": True}

    def test_groq_does_not_request_include_usage(self, monkeypatch):
        body = self._capture_stream_body(monkeypatch, _model("groq"), "groq")
        assert "stream_options" not in body

    def test_mistral_does_not_request_include_usage(self, monkeypatch):
        body = self._capture_stream_body(monkeypatch, _model("mistral"), "mistral")
        assert "stream_options" not in body
