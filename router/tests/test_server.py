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

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402
from router import config  # noqa: E402
from router.errors import AuthenticationError, RateLimitError  # noqa: E402


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
            RunLimits(max_cost_usd=0.0001, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9),
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
        body = {"model": "flux-auto", "messages": ["not an object", {"role": "user", "content": "hi"}]}
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
            return "streamed text"

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
        _mock_call_model.side_effect = [RateLimitError("429 Too Many Requests"), "fallback text"]
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
            return "ok from fallback"

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
        assert resp.status_code == 401

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
            return "ok from fallback"

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
