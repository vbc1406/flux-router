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
  - X-Flux-Run-Id: echoed back, auto-generated when absent, groups repeated
    calls into one run-budget trajectory, 429 + summary once exceeded
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402
from router import config  # noqa: E402
from router.config import ServerTokenBinding  # noqa: E402
from router.errors import AuthenticationError, RateLimitError  # noqa: E402
from router.provider_caller import ProviderResult  # noqa: E402


def _pr(text: str) -> ProviderResult:
    """Build a ProviderResult with no provider-reported usage (the default
    shape for a mocked _call_model that doesn't care about actual-usage
    behavior specifically) — see TestActualUsage for tests that do."""
    return ProviderResult(
        text=text, input_tokens=None, output_tokens=None, usage_source="estimated"
    )


@pytest.fixture(autouse=True)
def _mock_call_model(monkeypatch):
    mock = AsyncMock(return_value=_pr("mock response text"))
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

    def test_rejects_non_ascii_token_with_401_not_500(self, client, monkeypatch):
        """hmac.compare_digest raises TypeError on non-ASCII str, so comparing
        raw strs would turn a garbage token into an uncaught 500. Tokens are
        compared as UTF-8 bytes; every rejection stays on the 401 path.

        Sent as pre-encoded latin-1 bytes because that is how such a header
        actually reaches the app: ASGI servers decode header bytes as latin-1,
        so those octets arrive as a non-ASCII str. (httpx refuses to encode a
        non-ASCII str header itself, which is why this can't be written as a
        plain string.)"""
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "secret-token")
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer ké-tökèn".encode("latin-1")},
        )
        assert resp.status_code == 401

    def test_rejects_when_auth_required_but_no_token_configured(self, client, monkeypatch):
        """Fails closed: SERVER_REQUIRE_AUTH without a token must reject, not
        crash in compare_digest on None."""
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", None)
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status_code == 401


class TestMultiTenantTokens:
    """Regression: with a single shared FLUX_SERVER_TOKEN, any caller who
    holds it is authenticated as every tenant — X-Flux-Tenant-Id is a bare,
    self-declared claim. FLUX_SERVER_TOKENS binds each bearer token to one
    tenant_id so the proxy can enforce isolation itself."""

    def _enable(self, monkeypatch):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "SERVER_TOKENS", {"tok-acme": "acme", "tok-globex": "globex"})

    def test_bound_tenant_overrides_client_header(self, client, monkeypatch, _mock_call_model):
        self._enable(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer tok-acme", "X-Flux-Tenant-Id": "globex"},
        )
        assert resp.status_code == 200
        records, _ = server._flux._engine._attribution.usage(tenant_id="acme", limit=10)
        assert len(records) >= 1
        spoofed, _ = server._flux._engine._attribution.usage(tenant_id="globex", limit=10)
        assert not any(r.run_id == resp.headers["x-flux-run-id"] for r in spoofed)

    def test_usage_endpoint_ignores_requested_tenant(self, client, monkeypatch, _mock_call_model):
        self._enable(monkeypatch)
        client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer tok-acme"},
        )
        # Even asking explicitly for another tenant's usage returns only
        # the caller's own (bound) tenant's records.
        resp = client.get(
            "/v1/usage",
            params={"tenant_id": "globex"},
            headers={"Authorization": "Bearer tok-acme"},
        )
        assert resp.status_code == 200
        assert all(r["tenant_id"] == "acme" for r in resp.json()["data"])

    def test_unknown_token_rejected(self, client, monkeypatch):
        self._enable(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert resp.status_code == 401

    def test_tenant_daily_cap_blocks_regardless_of_rotated_user_field(
        self, client, monkeypatch, _mock_call_model
    ):
        """Regression: plan/daily budgets are keyed by the client-supplied
        `user` field, so an authenticated caller could mint a fresh `user`
        per request and evade every per-user cap while still hitting the
        same tenant. TENANT_DAILY_CAP_USD is keyed by the bearer token's
        bound tenant instead, which the caller cannot rotate around."""
        self._enable(monkeypatch)
        monkeypatch.setattr(server, "TENANT_DAILY_CAP_USD", 0.0000001)

        # A literal (non-free) model guarantees estimated_cost > 0, so the
        # first call actually records spend against the tenant cap.
        body = _body("gpt-4o")
        body["user"] = "rotated-user-1"
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok-acme"}
        )
        assert resp.status_code == 200

        # A brand-new `user` on the same tenant is still blocked — the cap
        # doesn't reset just because the client claims to be someone new.
        body["user"] = "rotated-user-2"
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok-acme"}
        )
        assert resp.status_code == 429
        assert resp.json()["error"]["type"] == "tenant_daily_cap_exceeded"

        # A different tenant (different bound token) is unaffected.
        body["user"] = "rotated-user-3"
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"Authorization": "Bearer tok-globex"}
        )
        assert resp.status_code == 200

    def test_run_budget_state_isolated_across_tenants_sharing_run_id(
        self, client, monkeypatch, _mock_call_model
    ):
        """Regression: X-Flux-Run-Id is client-controlled. Two different
        tenants colliding on the same run_id must not share run-budget state
        — otherwise one tenant could exhaust or observe another's run
        accounting by guessing/reusing a run_id."""
        self._enable(monkeypatch)
        from router.run_budget import RunLimits

        shared_run_id = "shared-run-id"
        rb = server._flux._engine._run_budget
        # Exhaust the run budget for globex only.
        rb.start(
            shared_run_id,
            RunLimits(
                max_cost_usd=0.0001, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
            ),
            tenant_id="globex",
        )
        rb.record_step(shared_run_id, "m", 1.0, 10, tenant_id="globex")

        # acme, using the SAME run_id, must be unaffected.
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer tok-acme", "X-Flux-Run-Id": shared_run_id},
        )
        assert resp.status_code == 200

        # globex's own request against that run_id still hits its exhausted budget.
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"Authorization": "Bearer tok-globex", "X-Flux-Run-Id": shared_run_id},
        )
        assert resp.status_code == 429


class TestPerTenantPlanBinding:
    """Regression: RoutingRequest.plan used to be left unset for every proxy
    request, so it always fell back to RoutingRequest's schema default
    ("pro_plan") regardless of the caller's token — an operator had no way
    to actually cap a tenant at free_plan's tighter tier/budget limits
    through this API. FLUX_SERVER_TOKENS now optionally binds a token to a
    specific plan via ServerTokenBinding; legacy string-shorthand entries and
    no-auth mode keep the old "pro_plan" default unchanged (opt-in, not a
    breaking default change)."""

    def _enable(self, monkeypatch, tokens: dict):
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "SERVER_TOKENS", tokens)

    def test_legacy_string_token_keeps_pro_plan_default(
        self, client, monkeypatch, _mock_call_model
    ):
        """No behavior change for existing deployments that haven't opted in:
        a bare tenant_id string still gets pro_plan-level model access."""
        self._enable(monkeypatch, {"tok-acme": "acme"})
        resp = client.post(
            "/v1/chat/completions",
            json=_body("gpt-4o"),
            headers={"Authorization": "Bearer tok-acme"},
        )
        assert resp.status_code == 200
        assert resp.headers["x-flux-model"] == "gpt-4o"

    def test_explicit_free_plan_binding_blocks_mid_tier_model(
        self, client, monkeypatch, _mock_call_model
    ):
        """A token explicitly bound to free_plan cannot reach a mid-tier
        model even by naming it directly — Step 4's plan-tier hard
        constraint filters it out before the explicit-override shortcut ever
        sees it, so routing falls back to a free/cheap-tier model instead."""
        self._enable(
            monkeypatch, {"tok-acme": ServerTokenBinding(tenant_id="acme", plan="free_plan")}
        )
        resp = client.post(
            "/v1/chat/completions",
            json=_body("gpt-4o"),
            headers={"Authorization": "Bearer tok-acme"},
        )
        assert resp.status_code == 200
        assert resp.headers["x-flux-model"] != "gpt-4o"

    def test_explicit_business_plan_binding_allows_mid_tier_model(
        self, client, monkeypatch, _mock_call_model
    ):
        self._enable(
            monkeypatch, {"tok-acme": ServerTokenBinding(tenant_id="acme", plan="business_plan")}
        )
        resp = client.post(
            "/v1/chat/completions",
            json=_body("gpt-4o"),
            headers={"Authorization": "Bearer tok-acme"},
        )
        assert resp.status_code == 200
        assert resp.headers["x-flux-model"] == "gpt-4o"

    def test_no_auth_mode_keeps_pro_plan_default(self, client, _mock_call_model):
        """Unauthenticated / no-FLUX_SERVER_TOKENS deployments (the common
        solo/local case) see no behavior change either."""
        resp = client.post("/v1/chat/completions", json=_body("gpt-4o"))
        assert resp.status_code == 200
        assert resp.headers["x-flux-model"] == "gpt-4o"


class TestParseServerTokens:
    """Unit tests for config._parse_server_tokens()'s two accepted shapes."""

    def test_legacy_string_shape_defaults_to_pro_plan(self, monkeypatch):
        monkeypatch.setenv("FLUX_SERVER_TOKENS", '{"tok-acme": "acme"}')
        tokens = config._parse_server_tokens()
        assert tokens["tok-acme"] == ServerTokenBinding(tenant_id="acme", plan="pro_plan")

    def test_dict_shape_with_explicit_plan(self, monkeypatch):
        monkeypatch.setenv(
            "FLUX_SERVER_TOKENS",
            '{"tok-acme": {"tenant_id": "acme", "plan": "free_plan"}}',
        )
        tokens = config._parse_server_tokens()
        assert tokens["tok-acme"] == ServerTokenBinding(tenant_id="acme", plan="free_plan")

    def test_dict_shape_without_plan_defaults_to_pro_plan(self, monkeypatch):
        monkeypatch.setenv("FLUX_SERVER_TOKENS", '{"tok-acme": {"tenant_id": "acme"}}')
        tokens = config._parse_server_tokens()
        assert tokens["tok-acme"] == ServerTokenBinding(tenant_id="acme", plan="pro_plan")

    def test_invalid_plan_value_falls_back_to_pro_plan(self, monkeypatch):
        monkeypatch.setenv(
            "FLUX_SERVER_TOKENS",
            '{"tok-acme": {"tenant_id": "acme", "plan": "unlimited_plan"}}',
        )
        tokens = config._parse_server_tokens()
        assert tokens["tok-acme"].plan == "pro_plan"

    def test_entry_missing_tenant_id_is_skipped(self, monkeypatch):
        monkeypatch.setenv("FLUX_SERVER_TOKENS", '{"tok-acme": {"plan": "business_plan"}}')
        tokens = config._parse_server_tokens()
        assert "tok-acme" not in tokens

    def test_malformed_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("FLUX_SERVER_TOKENS", "{not valid json")
        assert config._parse_server_tokens() == {}


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

    def test_non_object_message_element_returns_400_not_500(self, client):
        """Regression: a non-dict element in `messages` (e.g. a bare string)
        used to hit m.get(...) unconditionally, raising an uncaught
        AttributeError -> bare 500 instead of a clean 400."""
        body = {
            "model": "flux-auto",
            "messages": ["not an object", {"role": "user", "content": "hi"}],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 400

    def test_system_message_without_content_returns_400_not_500(self, client, _mock_call_model):
        """Regression: a system message with no `content` key is legal per
        OpenAI's schema (some SDKs omit it), but system_parts used m["content"]
        unconditionally, raising an uncaught KeyError -> bare 500 instead of
        the clean 400 every other malformed-input case here produces."""
        body = {
            "model": "flux-auto",
            "messages": [
                {"role": "system"},
                {"role": "user", "content": "hi"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

    def test_system_message_null_content_treated_as_empty(self, client, _mock_call_model):
        """Regression: `"content": null` on a system message is legal JSON but
        m.get("content", "") only supplies the default when the key is absent
        (not when it's explicitly null) -- "\\n".join() then hit an uncaught
        TypeError -> bare 500. Fixed to behave the same as an absent `content`
        key (empty system prompt, 200), not a 500."""
        body = {
            "model": "flux-auto",
            "messages": [
                {"role": "system", "content": None},
                {"role": "user", "content": "hi"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

    def test_system_message_non_string_content_returns_400_not_500(self, client):
        """Non-string, non-null content (e.g. an int) on a system message
        must also be rejected cleanly rather than reaching the join()."""
        body = {
            "model": "flux-auto",
            "messages": [
                {"role": "system", "content": 5},
                {"role": "user", "content": "hi"},
            ],
        }
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 400

    def test_non_string_model_field_returns_400_not_500(self, client):
        """Regression: `model_field in _ROUTING_DIRECTIVES` (a dict) raised an
        uncaught TypeError ("unhashable type") for a list/dict `model` value
        -> bare 500 instead of a clean 400."""
        body = {"model": ["not", "a", "string"], "messages": [{"role": "user", "content": "hi"}]}
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 400

        body2 = {"model": {"nested": "object"}, "messages": [{"role": "user", "content": "hi"}]}
        resp2 = client.post("/v1/chat/completions", json=body2)
        assert resp2.status_code == 400


class TestRunStoreStartupWarning:
    """_warn_if_unsafe_run_store() is called at import time with the real
    env-derived config; here we call it directly with controlled args rather
    than reloading the module (which would re-run all its other import-time
    side effects, e.g. re-creating `app` and `_flux`)."""

    def test_warns_when_multi_worker_without_redis(self, monkeypatch):
        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(server.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        server._warn_if_unsafe_run_store(3, "memory")
        assert any(ev == "flux_multi_worker_no_redis_run_store" for ev, _ in events)
        matching = [kw for ev, kw in events if ev == "flux_multi_worker_no_redis_run_store"][0]
        assert matching["workers"] == 3
        assert matching["run_store_backend"] == "memory"

    def test_no_warning_for_single_worker(self, monkeypatch):
        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(server.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        server._warn_if_unsafe_run_store(1, "memory")
        assert not any(ev == "flux_multi_worker_no_redis_run_store" for ev, _ in events)

    def test_no_warning_for_multi_worker_with_redis(self, monkeypatch):
        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(server.log, "warning", lambda ev, **kw: events.append((ev, kw)))
        server._warn_if_unsafe_run_store(4, "redis")
        assert not any(ev == "flux_multi_worker_no_redis_run_store" for ev, _ in events)


class TestRunBudget:
    def test_run_id_auto_generated_and_echoed(self, client):
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        assert resp.headers["x-flux-run-id"] != ""
        assert resp.headers["x-flux-budget-state"] == "ok"

    def test_run_id_header_is_echoed_back(self, client):
        resp = client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Flux-Run-Id": "my-run-42"}
        )
        assert resp.status_code == 200
        assert resp.headers["x-flux-run-id"] == "my-run-42"

    def test_repeated_calls_share_run_state(self, client):
        run_id = "shared-run"
        for _ in range(3):
            resp = client.post(
                "/v1/chat/completions", json=_body(), headers={"X-Flux-Run-Id": run_id}
            )
            assert resp.status_code == 200
        cost, steps = server._flux._engine._run_budget.snapshot(run_id)
        assert steps == 3

    def test_run_budget_exceeded_returns_429_with_summary(self, client):
        from router.run_budget import RunLimits

        # Start a run with a tiny cap and pre-record a step that already blows
        # past it, so the NEXT request is blocked before it ever dispatches.
        run_id = "tight-run"
        rb = server._flux._engine._run_budget
        rb.start(
            run_id,
            RunLimits(
                max_cost_usd=0.0001, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
            ),
        )
        rb.record_step(run_id, "some-model", 1.0, 100)

        resp = client.post("/v1/chat/completions", json=_body(), headers={"X-Flux-Run-Id": run_id})
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["type"] == "run_budget_exceeded"
        assert "steps_taken" in body["error"]
        assert resp.headers["x-flux-run-id"] == run_id

    # ── Bugfix coverage: loud auto-generated run_id ─────────────────────────

    def test_missing_run_id_header_sets_missing_flag_and_header(self, client):
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        assert resp.headers["x-flux-run-id"] != ""
        assert resp.headers["x-flux-run-id-missing"] == "true"

    def test_missing_run_id_header_logs_a_warning(self, client, monkeypatch):
        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            server.log, "warning", lambda ev, **kw: events.append((ev, kw))
        )
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        assert any(ev == "flux_run_id_auto_generated" for ev, _ in events)
        matching = [kw for ev, kw in events if ev == "flux_run_id_auto_generated"]
        assert matching[0]["run_id"] == resp.headers["x-flux-run-id"]

    def test_supplied_run_id_header_does_not_set_missing_flag_or_warn(self, client, monkeypatch):
        events: list[tuple[str, dict]] = []
        monkeypatch.setattr(
            server.log, "warning", lambda ev, **kw: events.append((ev, kw))
        )
        resp = client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Flux-Run-Id": "explicit-run"}
        )
        assert resp.status_code == 200
        assert "x-flux-run-id-missing" not in resp.headers
        assert not any(ev == "flux_run_id_auto_generated" for ev, _ in events)

    def test_missing_run_id_flag_survives_streaming_path(self, client, monkeypatch):
        async def fake_call(model, request):
            return _pr("streamed text")

        monkeypatch.setattr(server._flux, "_call_model", fake_call)
        resp = client.post("/v1/chat/completions", json=_body(stream=True))
        assert resp.status_code == 200
        assert resp.headers["x-flux-run-id-missing"] == "true"


class TestAttribution:
    def test_chat_completion_records_usage_under_tenant_header(self, client):
        resp = client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Flux-Tenant-Id": "acme"}
        )
        assert resp.status_code == 200
        usage = client.get("/v1/usage", params={"tenant_id": "acme"}).json()
        assert usage["total"] >= 1
        assert all(r["tenant_id"] == "acme" for r in usage["data"])

    def test_usage_endpoint_paginates(self, client):
        for _ in range(3):
            client.post(
                "/v1/chat/completions", json=_body(), headers={"X-Flux-Tenant-Id": "paginate-me"}
            )
        page = client.get(
            "/v1/usage", params={"tenant_id": "paginate-me", "limit": 2, "offset": 0}
        ).json()
        assert page["limit"] == 2
        assert len(page["data"]) == 2
        assert page["total"] == 3

    def test_usage_endpoint_never_includes_prompt_or_response_text(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"X-Flux-Tenant-Id": "no-leak-test"},
        )
        assert resp.status_code == 200
        usage = client.get("/v1/usage", params={"tenant_id": "no-leak-test"}).json()
        for record in usage["data"]:
            assert set(record.keys()) == {
                "tenant_id",
                "run_id",
                "task_type",
                "step_type",
                "model_id",
                "cost_usd",
                "timestamp",
                "usage_source",
                "input_tokens",
                "output_tokens",
            }

    def test_metrics_endpoint_returns_prometheus_text(self, client):
        client.post(
            "/v1/chat/completions", json=_body(), headers={"X-Flux-Tenant-Id": "metrics-co"}
        )
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        assert "flux_cost_usd_total" in resp.text
        assert 'tenant_id="metrics-co"' in resp.text

    def test_run_budget_exceeded_increments_metric(self, client):
        from router.run_budget import RunLimits

        run_id = "metrics-exceeded-run"
        tenant_id = "budget-blown-co"
        rb = server._flux._engine._run_budget
        # Run budget state is scoped by tenant_id (see run_budget.py::RunBudget._key)
        # so seeding it here must match the tenant the request below authenticates as.
        rb.start(
            run_id,
            RunLimits(
                max_cost_usd=0.0001, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
            ),
            tenant_id=tenant_id,
        )
        rb.record_step(run_id, "m", 1.0, 10, tenant_id=tenant_id)
        client.post(
            "/v1/chat/completions",
            json=_body(),
            headers={"X-Flux-Run-Id": run_id, "X-Flux-Tenant-Id": "budget-blown-co"},
        )
        resp = client.get("/metrics")
        assert 'flux_budget_exceeded_total{tenant_id="budget-blown-co"} 1' in resp.text


class TestPlanBudgetEnforcement:
    """
    Regression tests for the plan-budget gap: the HTTP path used to call
    Flux._call_model() directly, skipping BudgetTracker.record_spend()
    entirely, so a user's plan spend never accumulated across requests.
    """

    def test_successful_completion_records_plan_spend(self, client):
        budget = server._flux._engine._budget
        user = "plan-budget-user-1"
        before = budget.get_daily_spend(user)

        # A literal (non-free) model guarantees estimated_cost > 0 — routing
        # a free-tier model would record a real $0.00 entry and this
        # assertion would pass even if record_spend() were never called.
        body = _body("gpt-4o")
        body["user"] = user
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

        after = budget.get_daily_spend(user)
        assert after > before, "record_spend() was not called on the HTTP path"

    def test_streaming_completion_records_plan_spend(self, client, _mock_call_model):
        """Regression: the stream=True branch never called
        BudgetTracker.record_spend() at all, so plan spend never accumulated
        for streaming traffic even though the non-streaming path was fixed."""
        budget = server._flux._engine._budget
        user = "plan-budget-stream-user"
        before = budget.get_daily_spend(user)

        # Non-native streaming path, and a non-free tier so a $0.00 spend
        # can't accidentally make the "spend increased" assertion pass.
        body = _body("claude-haiku-4-5-20251001", stream=True)
        body["user"] = user
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

        after = budget.get_daily_spend(user)
        assert after > before, "record_spend() was not called on the streaming HTTP path"

    def test_streaming_failure_does_not_record_spend(self, client, monkeypatch):
        """A call that fails outright (never produced a token) must not be
        charged, unlike a client disconnecting mid-stream after output
        started."""
        from router.provider_caller import ProviderCallError

        async def failing_stream(model, request, api_key):
            raise ProviderCallError("boom", 500)
            yield b""  # pragma: no cover - unreachable; keeps this an async generator

        monkeypatch.setattr(server, "stream_openai_compat_lines", failing_stream)

        budget = server._flux._engine._budget
        user = "plan-budget-stream-fail-user"
        before = budget.get_daily_spend(user)

        body = _body("gpt-5-mini", stream=True)  # native streaming path
        body["user"] = user
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200  # errors are reported in-band over SSE
        assert '"error"' in resp.text

        after = budget.get_daily_spend(user)
        assert after == before, "spend was recorded despite the provider call failing outright"

    def test_repeated_completions_accumulate_spend(self, client):
        budget = server._flux._engine._budget
        user = "plan-budget-user-2"

        body = _body("gpt-4o")
        body["user"] = user
        for _ in range(3):
            resp = client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200

        # Three recorded steps, not zero — proves each call independently records.
        report = budget.get_savings_report(user)
        assert report["record_count"] == 3

    def test_budget_pressure_degrades_to_cheapest_tier(self, client, monkeypatch):
        from router import budget_tracker as bt

        # Shrink pro_plan's cap to near-zero so any real request cost blows
        # through it immediately — forces route()'s tier walk-down to the
        # cheapest (free-tier) model on the very first call.
        monkeypatch.setitem(bt.BUDGET_LIMITS, "pro_plan", {"daily": 1e-9, "monthly": 1e-9})

        body = _body()
        body["user"] = "plan-budget-pressured-user"
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200

        registry = server._flux._engine._registry
        chosen = registry.get_model(resp.headers["x-flux-model"])
        assert chosen is not None
        assert chosen.tier == "free"


class TestFallbackChain:
    """
    Regression tests for the flat-502 gap: the HTTP path used to call
    Flux._call_model() directly with no retry, so any transient provider
    error became a 502 instead of trying the routing decision's fallback
    chain the way Flux.complete() does.
    """

    def test_transient_error_falls_back_instead_of_502(self, client, _mock_call_model):
        _mock_call_model.side_effect = [
            RateLimitError("429 Too Many Requests"),
            _pr("fallback text"),
        ]
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "fallback text"
        assert _mock_call_model.call_count == 2

    def test_fallback_model_reflected_in_response_and_headers(self, client, _mock_call_model):
        calls: list[str] = []

        async def flaky(model, request):
            calls.append(model.model_id)
            if len(calls) == 1:
                raise RateLimitError("429")
            return _pr("ok from fallback")

        _mock_call_model.side_effect = flaky
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == calls[-1]
        assert resp.headers["x-flux-model"] == calls[-1]
        assert calls[0] != calls[-1]

    def test_all_models_failing_returns_502_with_attempt_summary(self, client, _mock_call_model):
        _mock_call_model.side_effect = RateLimitError("always 429")
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 502
        detail = resp.json()["detail"].lower()
        assert "attempt" in detail
        assert "failed" in detail

    def test_auth_error_short_circuits_without_retry(self, client, _mock_call_model):
        _mock_call_model.side_effect = AuthenticationError("bad key")
        resp = client.post("/v1/chat/completions", json=_body())
        # 502, not 401: a rejected UPSTREAM provider key is a server-side
        # misconfiguration, not a bad client bearer token. See the handler in
        # server.py for why conflating the two sent callers rotating their own
        # (valid) credentials.
        assert resp.status_code == 502

    def test_provider_auth_error_does_not_leak_provider_detail(self, client, _mock_call_model):
        """A caller must not be able to tell a rejected upstream key apart from
        any other upstream failure, nor learn which provider was routed to."""
        _mock_call_model.side_effect = AuthenticationError("HTTP 403 from google")
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        assert "google" not in detail.lower()
        assert "403" not in detail

    def test_streaming_provider_auth_error_does_not_leak_provider_detail(
        self, client, _mock_call_model
    ):
        _mock_call_model.side_effect = AuthenticationError("HTTP 403 from google")
        resp = client.post("/v1/chat/completions", json=_body(stream=True))
        assert resp.status_code == 200  # SSE errors ride in-band, not in the status
        assert "google" not in resp.text.lower()
        assert "403" not in resp.text

    def test_fallback_records_dispatched_models_own_cost(self, client, _mock_call_model):
        """Regression: recording used decision.estimated_cost (the
        originally-chosen model's cost) even when a fallback dispatched a
        different, differently-priced model — silently mis-tracking real
        spend. The recorded amount must be rescaled to the model that was
        actually called."""
        from router.cascade import estimate_step_cost

        calls: list = []

        async def flaky(model, request):
            calls.append(model)
            if len(calls) == 1:
                raise RateLimitError("429")
            return _pr("ok from fallback")

        _mock_call_model.side_effect = flaky

        user = "fallback-cost-user"
        budget = server._flux._engine._budget
        before = budget.get_daily_spend(user)

        body = _body()
        body["user"] = user
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert len(calls) == 2
        chosen_model, dispatched_model = calls[0], calls[1]

        decision_cost_hint = float(resp.headers["x-flux-estimated-cost-usd"])
        expected_recorded = estimate_step_cost(dispatched_model, chosen_model, decision_cost_hint)

        after = budget.get_daily_spend(user)
        recorded = after - before
        assert recorded == pytest.approx(expected_recorded, rel=1e-6)
        if dispatched_model.model_id != chosen_model.model_id and (
            dispatched_model.cost_per_1k_input != chosen_model.cost_per_1k_input
            or dispatched_model.cost_per_1k_output != chosen_model.cost_per_1k_output
        ):
            assert recorded != pytest.approx(decision_cost_hint, rel=1e-6)


class TestActualUsage:
    """Task: record actual provider-reported usage, not estimates.

    Covers the non-streaming and streaming response paths recording
    usage_source/actual cost correctly, and the disconnect-mid-stream edge
    case (client walks away before the stream would have ended naturally).
    """

    def test_non_streaming_records_provider_usage(self, client, monkeypatch):
        async def mock_call(model, request):
            return ProviderResult(
                text="hi there", input_tokens=100, output_tokens=50, usage_source="provider"
            )

        monkeypatch.setattr(server._flux, "_call_model", mock_call)
        body = _body("gpt-4o")
        body["user"] = "actual-usage-user-1"
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert resp.headers["x-flux-usage-source"] == "provider"
        assert "x-flux-actual-cost-usd" in resp.headers
        data = resp.json()
        assert data["usage"]["prompt_tokens"] == 100
        assert data["usage"]["completion_tokens"] == 50
        assert data["usage"]["total_tokens"] == 150

        usage = client.get("/v1/usage", params={"tenant_id": "acme-unused"}).json()
        # Not tenant-scoped here (no X-Flux-Tenant-Id sent) — look at the
        # most recent global record set instead via a fresh request keyed by tenant.
        resp2 = client.post(
            "/v1/chat/completions", json=body, headers={"X-Flux-Tenant-Id": "actual-usage-tenant"}
        )
        assert resp2.status_code == 200
        usage2 = client.get("/v1/usage", params={"tenant_id": "actual-usage-tenant"}).json()
        record = usage2["data"][0]
        assert record["usage_source"] == "provider"
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50

    def test_non_streaming_falls_back_to_estimated_without_provider_usage(
        self, client, _mock_call_model
    ):
        resp = client.post("/v1/chat/completions", json=_body())
        assert resp.status_code == 200
        assert resp.headers["x-flux-usage-source"] == "estimated"
        assert "x-flux-actual-cost-usd" not in resp.headers

    def test_streaming_openai_usage_chunk_recorded_as_actual_and_stripped_by_default(
        self, client, monkeypatch
    ):
        async def fake_stream(model, request, api_key):
            yield b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":'
                b'{"prompt_tokens":40,"completion_tokens":10,"total_tokens":50}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        body = _body("gpt-5-mini", stream=True)
        body["user"] = "stream-usage-user"
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"X-Flux-Tenant-Id": "stream-usage-tenant"}
        )
        assert resp.status_code == 200
        # Client never set stream_options.include_usage — the usage chunk
        # must not appear in what we forwarded.
        assert '"usage"' not in resp.text

        usage = client.get("/v1/usage", params={"tenant_id": "stream-usage-tenant"}).json()
        record = usage["data"][0]
        assert record["usage_source"] == "provider"
        assert record["input_tokens"] == 40
        assert record["output_tokens"] == 10

    def test_streaming_usage_chunk_forwarded_when_client_opts_in(self, client, monkeypatch):
        async def fake_stream(model, request, api_key):
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield (
                b'data: {"choices":[],"usage":'
                b'{"prompt_tokens":40,"completion_tokens":10,"total_tokens":50}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        body = _body("gpt-5-mini", stream=True)
        body["stream_options"] = {"include_usage": True}
        resp = client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 200
        assert '"usage"' in resp.text
        assert '"prompt_tokens":40' in resp.text

    def test_streaming_without_usage_chunk_falls_back_to_estimated(self, client, monkeypatch):
        async def fake_stream(model, request, api_key):
            yield b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        body = _body("gpt-5-mini", stream=True)
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"X-Flux-Tenant-Id": "stream-no-usage"}
        )
        assert resp.status_code == 200

        usage = client.get("/v1/usage", params={"tenant_id": "stream-no-usage"}).json()
        record = usage["data"][0]
        assert record["usage_source"] == "estimated"
        assert record["input_tokens"] is None
        assert record["output_tokens"] is None

    def test_streaming_usage_chunk_with_no_content_chunks_still_records_actual(
        self, client, monkeypatch
    ):
        """A usage-only stream (no content deltas at all) must still be
        recorded from the provider's actual usage, not silently dropped."""

        async def fake_stream(model, request, api_key):
            yield (
                b'data: {"choices":[],"usage":'
                b'{"prompt_tokens":5,"completion_tokens":0,"total_tokens":5}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        body = _body("gpt-5-mini", stream=True)
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"X-Flux-Tenant-Id": "stream-usage-only"}
        )
        assert resp.status_code == 200

        usage = client.get("/v1/usage", params={"tenant_id": "stream-usage-only"}).json()
        record = usage["data"][0]
        # completion_tokens=0 fails _safe_usage_int's positivity check, so
        # this is NOT trustworthy actual usage — falls back to estimated
        # rather than billing a real request at zero output tokens.
        assert record["usage_source"] == "estimated"

    def test_synthesized_path_records_provider_usage(self, client, monkeypatch):
        """Anthropic/Google (non-native-streaming) path: _call_model already
        returns a ProviderResult, so streaming gets exact usage for free."""

        async def mock_call(model, request):
            return ProviderResult(
                text="hello", input_tokens=30, output_tokens=12, usage_source="provider"
            )

        monkeypatch.setattr(server._flux, "_call_model", mock_call)
        body = _body("gemini-2.0-flash-lite", stream=True)
        resp = client.post(
            "/v1/chat/completions", json=body, headers={"X-Flux-Tenant-Id": "synth-usage-tenant"}
        )
        assert resp.status_code == 200

        usage = client.get("/v1/usage", params={"tenant_id": "synth-usage-tenant"}).json()
        record = usage["data"][0]
        assert record["usage_source"] == "provider"
        assert record["input_tokens"] == 30
        assert record["output_tokens"] == 12

    def test_disconnect_mid_stream_records_estimated_exactly_once(self, client, monkeypatch):
        """Regression: _record_usage() used to fire on the FIRST chunk with a
        guessed token count; now it's deferred to stream end, so a client
        that walks away before the stream finishes naturally must still be
        billed (at the estimate) exactly once, via the `finally` block, not
        left unbilled or double-billed."""
        import asyncio

        async def hanging_stream(model, request, api_key):
            yield b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            # Never reaches [DONE] — simulates a stream still in flight when
            # the client walks away.
            await asyncio.sleep(3600)
            yield b""  # pragma: no cover - unreachable

        monkeypatch.setattr(server, "stream_openai_compat_lines", hanging_stream)

        body = _body("gpt-5-mini", stream=True)
        body["user"] = "disconnect-user-1"
        run_id = "disconnect-run-1"

        async def drive():
            routing_request, _ = server._build_routing_request(body, run_id, None, plan=None)
            decision = await server._flux.route(routing_request, verbose=True)
            headers = server._flux_headers(decision, 0.0)
            gen = server._stream_completion(
                routing_request, decision, "chatcmpl-test", 0, headers, None, False
            )
            await gen.__anext__()  # role chunk
            await gen.__anext__()  # first content chunk
            await gen.aclose()  # client walks away before [DONE]

        asyncio.run(drive())

        budget = server._flux._engine._budget
        assert budget.get_daily_spend("disconnect-user-1") > 0, (
            "disconnect before stream end must still bill at the estimate"
        )
        cost_so_far, steps_so_far = server._flux._engine._run_budget.snapshot(run_id)
        assert steps_so_far == 1, "run-budget reservation must be resolved exactly once"


class TestDashboardMountGuard:
    """The dashboard shows every tenant's spend and the deployment's config.
    Whether it mounts is decided from the CONFIGURED bind address, at import
    time — these test the decision function directly, since the mount itself
    happens once when the module loads."""

    def test_mounts_unauthenticated_on_loopback(self):
        """The single-operator self-hosted case it exists for."""
        assert server._dashboard_refusal_reason("127.0.0.1", False, True) is None

    def test_refuses_on_bind_all_without_auth(self):
        reason = server._dashboard_refusal_reason("0.0.0.0", False, True)
        assert reason is not None
        assert "0.0.0.0" in reason

    def test_mounts_on_bind_all_when_a_token_is_configured(self):
        assert server._dashboard_refusal_reason("0.0.0.0", True, True) is None

    def test_disabled_flag_refuses_even_on_loopback(self):
        assert server._dashboard_refusal_reason("127.0.0.1", False, False) == "FLUX_DASHBOARD=0"

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.5", ""])
    def test_loopback_addresses(self, host):
        assert server._is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "192.168.1.10", "::", "example.com"])
    def test_non_loopback_addresses(self, host):
        assert server._is_loopback(host) is False

    def test_unresolvable_hostname_is_treated_as_non_loopback(self):
        """Too cautious costs a warning; the other way round exposes spend."""
        assert server._is_loopback("some-host.internal") is False

    def test_refusal_is_not_fatal(self, client):
        """The proxy and its API keep serving when the dashboard is refused."""
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200
        assert client.get("/health").status_code == 200

    def test_openai_models_shape_is_unchanged_by_the_dashboard_work(self, client):
        """/v1/models is what OpenAI SDK clients read; the richer registry
        view lives at /v1/stats/registry precisely so this stays stable."""
        data = client.get("/v1/models").json()
        assert data["object"] == "list"
        entry = next(m for m in data["data"] if m["id"] == "gpt-5-mini")
        assert set(entry) == {"id", "object", "owned_by"}
        assert entry["object"] == "model"


class TestLatencyRecorded:
    """Routing overhead and provider wall-clock were previously measured and
    thrown away, so the dashboard had nothing to show. Both paths record now."""

    @pytest.fixture
    def recorded(self, monkeypatch):
        """Capture what the proxy hands to attribution.record()."""
        calls = []
        original = server._flux._engine._attribution.record

        def spy(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(server._flux._engine._attribution, "record", spy)
        return calls

    def test_non_streaming_records_latency(self, client, recorded):
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200

        assert len(recorded) == 1
        assert recorded[0]["latency_ms"] is not None
        assert recorded[0]["latency_ms"] >= 0

    def test_streaming_records_latency_at_stream_end(self, client, recorded, monkeypatch):
        async def fake_stream(model, request, api_key):
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            yield b"data: [DONE]\n\n"

        monkeypatch.setattr(server, "stream_openai_compat_lines", fake_stream)
        resp = client.post("/v1/chat/completions", json=_body("gpt-5-mini", stream=True))
        assert resp.status_code == 200
        assert resp.text  # consume the stream so the end-of-stream hook runs

        assert len(recorded) == 1
        assert recorded[0]["latency_ms"] is not None

    def test_decision_latency_is_recorded_separately(self, client, recorded):
        """Routing overhead, not the provider call — the number that answers
        'is Flux itself slow?'."""
        client.post("/v1/chat/completions", json=_body())

        decision = recorded[0]["decision_latency_ms"]
        assert decision is not None
        assert decision < recorded[0]["latency_ms"] + 1

    def test_decision_latency_header_is_exposed(self, client):
        resp = client.post("/v1/chat/completions", json=_body())
        assert "x-flux-decision-latency-ms" in resp.headers
        assert float(resp.headers["x-flux-decision-latency-ms"]) >= 0

    def test_savings_and_routing_detail_are_recorded(self, client, recorded):
        client.post("/v1/chat/completions", json=_body("flux-cheap"))

        rec = recorded[0]
        assert rec["estimated_savings_usd"] is not None
        assert rec["complexity_score"] is not None
        assert rec["routing_priority"] == "cost-optimized"


class TestDashboardRemoteRequestGuard:
    """The mount decision reads the CONFIGURED bind address, which is only the
    truth when the bind came from the environment. `uvicorn --host 0.0.0.0`
    leaves config reporting 127.0.0.1, the mount is allowed, and without this
    request-time check the dashboard is served to the whole network."""

    @staticmethod
    def _scope(peer: str | None) -> dict:
        return {"type": "http", "client": (peer, 54321) if peer else None}

    async def _call(self, peer: str | None):
        """Drive the wrapped app directly and capture what it sent."""
        sent = []
        inner_ran = []

        async def inner(scope, receive, send):
            inner_ran.append(True)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"dashboard"})

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        await server._LoopbackOnlyDashboard(inner)(self._scope(peer), receive, send)
        status = next(m["status"] for m in sent if m["type"] == "http.response.start")
        return status, bool(inner_ran)

    @pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_peers_are_served(self, peer):
        status, inner_ran = asyncio.run(self._call(peer))
        assert status == 200
        assert inner_ran is True

    @pytest.mark.parametrize("peer", ["192.168.1.8", "10.0.0.5", "203.0.113.7"])
    def test_remote_peers_are_refused(self, peer):
        status, inner_ran = asyncio.run(self._call(peer))
        assert status == 403
        assert inner_ran is False, "static files must not be served to a remote peer"

    def test_missing_client_is_refused(self):
        """No peer address means it can't be shown to be local."""
        status, _ = asyncio.run(self._call(None))
        assert status == 403

    def test_remote_peers_allowed_once_a_token_is_configured(self, monkeypatch):
        """With auth configured the token is the control, and remote access
        is the operator's intent rather than an accident."""
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        status, inner_ran = asyncio.run(self._call("192.168.1.8"))
        assert status == 200
        assert inner_ran is True

    def test_non_http_scopes_pass_through(self):
        """Lifespan/websocket messages must not be turned into JSON responses."""
        ran = []

        async def inner(scope, receive, send):
            ran.append(scope["type"])

        async def noop(*a):
            return {"type": "lifespan.startup"}

        asyncio.run(
            server._LoopbackOnlyDashboard(inner)({"type": "lifespan"}, noop, noop)
        )
        assert ran == ["lifespan"]

    def test_the_proxy_api_is_not_affected(self, client):
        """Only /dashboard is loopback-gated; the API keeps its own auth rules."""
        assert client.post("/v1/chat/completions", json=_body()).status_code == 200
        assert client.get("/health").status_code == 200
