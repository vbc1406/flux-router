"""
File: router/tests/test_server.py

Purpose:
Tests for router/server.py — the OpenAI-compatible HTTP proxy (Task 1).
All tests mock Flux._call_model / provider_caller streaming; no real HTTP
requests to model providers are made.

How to run:
  pytest -v router/tests/test_server.py

Test coverage:
  - Routing directive handling (flux-auto/flux-cheap/flux-quality)
  - Passthrough of literal model names (bypasses routing/classification)
  - Unknown literal model name is rejected
  - x-flux-* response headers are populated
  - Streaming chunk integrity (SSE frames, [DONE] terminator)
  - Auth rejection when FLUX_SERVER_TOKEN is set
  - Oversized body rejection
  - /health and /v1/models
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402
from router import config  # noqa: E402


@pytest.fixture(autouse=True)
def _mock_call_model(monkeypatch):
    mock = AsyncMock(return_value="mock response text")
    monkeypatch.setattr(server._flux, "_call_model", mock)
    return mock


@pytest.fixture
def client():
    return TestClient(server.app)


def _body(model: str = "flux-auto", stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "stream": stream,
    }


class TestRoutingDirectives:
    def test_flux_auto_routes_normally(self, client, _mock_call_model):
        resp = client.post("/v1/chat/completions", json=_body("flux-auto"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "mock response text"
        assert "x-flux-model" in resp.headers
        assert resp.headers["x-flux-task-type"] != ""

    def test_flux_cheap_forces_cost_optimized_priority(self, client, monkeypatch):
        captured = {}
        real_route = server._flux.route

        async def spy_route(request, verbose=False):
            captured["priority"] = request.routing_priority
            return await real_route(request, verbose=verbose)

        monkeypatch.setattr(server._flux, "route", spy_route)
        resp = client.post("/v1/chat/completions", json=_body("flux-cheap"))
        assert resp.status_code == 200
        assert captured["priority"] == "cost-optimized"

    def test_flux_quality_forces_quality_first_priority(self, client, monkeypatch):
        captured = {}
        real_route = server._flux.route

        async def spy_route(request, verbose=False):
            captured["priority"] = request.routing_priority
            return await real_route(request, verbose=verbose)

        monkeypatch.setattr(server._flux, "route", spy_route)
        resp = client.post("/v1/chat/completions", json=_body("flux-quality"))
        assert resp.status_code == 200
        assert captured["priority"] == "quality-first"


class TestLiteralModelPassthrough:
    def test_literal_model_name_bypasses_routing(self, client, _mock_call_model):
        resp = client.post("/v1/chat/completions", json=_body("claude-haiku-4-5-20251001"))
        assert resp.status_code == 200
        assert resp.json()["model"] == "claude-haiku-4-5-20251001"

    def test_unknown_literal_model_rejected(self, client):
        resp = client.post("/v1/chat/completions", json=_body("not-a-real-model"))
        assert resp.status_code == 400


class TestStreaming:
    def test_streaming_native_provider_forwards_chunks_incrementally(self, client, monkeypatch):
        async def fake_stream(model, request, api_key):
            for chunk in [
                b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
                b"data: [DONE]\n\n",
            ]:
                yield chunk

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        resp = client.post("/v1/chat/completions", json=_body("gpt-5-mini", stream=True))
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        body = resp.text
        frames = [f for f in body.split("\n\n") if f.strip()]
        assert frames[-1] == "data: [DONE]"
        assert any('"content":"Hel"' in f for f in frames)

    def test_streaming_non_native_provider_synthesizes_single_chunk(self, client, _mock_call_model):
        resp = client.post("/v1/chat/completions", json=_body("gemini-2.0-flash-lite", stream=True))
        assert resp.status_code == 200
        body = resp.text
        assert "mock response text" in body
        assert body.strip().endswith("data: [DONE]")


class TestAuth:
    def test_no_token_configured_allows_request(self, client):
        assert server.SERVER_REQUIRE_AUTH is False
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200

    def test_rejects_when_token_required_and_missing(self, client, monkeypatch):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "secret-token")
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 401

    def test_accepts_when_token_required_and_correct(self, client, monkeypatch):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "secret-token")
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    def test_rejects_when_token_required_and_wrong(self, client, monkeypatch):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "secret-token")
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401


class TestBodySize:
    def test_oversized_body_rejected(self, client, monkeypatch):
        monkeypatch.setattr(config, "SERVER_MAX_BODY_BYTES", 100)
        monkeypatch.setattr(server, "SERVER_MAX_BODY_BYTES", 100)
        big_body = _body()
        big_body["messages"][0]["content"] = "x" * 10_000
        resp = client.post("/v1/chat/completions", json=big_body)
        assert resp.status_code == 413


class TestMisc:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert any(m["id"] == "gpt-5-mini" for m in data["data"])

    def test_missing_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={"model": "flux-auto"})
        assert resp.status_code == 400
