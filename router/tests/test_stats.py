"""
File: router/tests/test_stats.py

Purpose:
Tests for the /v1/stats/* aggregate endpoints (router/server.py) and the
SQL aggregates behind them (SqliteUsageStore.summary/timeseries/by_model/
by_task_type). These are what the local operator dashboard reads.

Every test seeds a real temp-file SqliteUsageStore rather than ":memory:",
so the aggregates are exercised against the same on-disk shape a self-hosted
`flux serve` uses.

How to run:
  pytest -v router/tests/test_stats.py
"""

from __future__ import annotations

import json
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import router.server as server  # noqa: E402
from router import config  # noqa: E402
from router.attribution import CostAttribution, SqliteUsageStore, UsageRecord  # noqa: E402
from router.config import ServerTokenBinding  # noqa: E402

HOUR = 3600.0


def _rec(
    *,
    tenant: str | None = "acme",
    model: str = "gpt-4o-mini",
    task: str = "code_generation",
    cost: float = 0.01,
    age_seconds: float = 60.0,
    timestamp: float | None = None,
    run_id: str | None = "run-1",
    latency: float | None = 100.0,
    savings: float | None = 0.09,
    usage_source: str = "estimated",
    cache_hit: bool = False,
    fallback_used: bool = False,
) -> UsageRecord:
    """A usage row `age_seconds` in the past, with dashboard-relevant fields set.

    `timestamp` overrides `age_seconds` with an absolute epoch value, for
    tests that must control where a row falls relative to an epoch-aligned
    bucket boundary rather than relative to now.
    """
    return UsageRecord(
        tenant_id=tenant,
        run_id=run_id,
        task_type=task,
        step_type="completion",
        model_id=model,
        cost_usd=cost,
        timestamp=timestamp if timestamp is not None else time.time() - age_seconds,
        usage_source=usage_source,
        latency_ms=latency,
        estimated_savings_usd=savings,
        cache_hit=cache_hit,
        fallback_used=fallback_used,
    )


@pytest.fixture
def store(tmp_path):
    """A SqliteUsageStore backed by a real file, as `flux serve` uses."""
    return SqliteUsageStore(str(tmp_path / "flux.db"))


@pytest.fixture
def stats_client(monkeypatch, store):
    """TestClient whose server reads stats from `store`.

    Swaps the attribution facade rather than the app so the endpoints, auth,
    and tenant scoping all run for real.
    """
    monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
    # Present a loopback peer AND a loopback Host: the spend endpoints refuse
    # a caller that fails either when no auth is configured. TestClient's
    # defaults are the literal strings "testclient" and "testserver", neither
    # of which is a local address.
    return TestClient(server.app, client=("127.0.0.1", 50000), base_url="http://127.0.0.1:8000")


class TestSummaryAggregate:
    def test_headline_numbers(self, store):
        store.record(_rec(cost=0.01, savings=0.09, latency=100.0))
        store.record(_rec(cost=0.03, savings=0.07, latency=300.0, run_id="run-2"))

        s = store.summary()
        assert s["requests"] == 2
        assert s["runs"] == 2
        assert s["distinct_models"] == 1
        assert s["total_cost_usd"] == pytest.approx(0.04)
        assert s["estimated_savings_usd"] == pytest.approx(0.16)
        # baseline is cost + savings; savings_pct is the share of it saved.
        assert s["baseline_cost_usd"] == pytest.approx(0.20)
        assert s["savings_pct"] == pytest.approx(80.0)
        assert s["avg_latency_ms"] == pytest.approx(200.0)

    def test_actual_cost_tracks_provider_reported_rows_only(self, store):
        store.record(_rec(cost=0.02, usage_source="provider"))
        store.record(_rec(cost=0.02, usage_source="estimated"))

        s = store.summary()
        assert s["actual_cost_usd"] == pytest.approx(0.02)
        assert s["actual_cost_pct"] == pytest.approx(50.0)

    def test_cache_and_fallback_rates(self, store):
        store.record(_rec(cache_hit=True))
        store.record(_rec(fallback_used=True))
        store.record(_rec())

        s = store.summary()
        assert s["cache_hits"] == 1
        assert s["fallbacks"] == 1
        assert s["cache_hit_rate"] == pytest.approx(100 / 3)
        assert s["fallback_rate"] == pytest.approx(100 / 3)

    def test_rows_without_latency_are_excluded_not_zeroed(self, store):
        """A missing measurement is not a fast one — see _percentile()."""
        store.record(_rec(latency=200.0))
        store.record(_rec(latency=None))

        s = store.summary()
        assert s["avg_latency_ms"] == pytest.approx(200.0)
        assert s["p50_latency_ms"] == pytest.approx(200.0)

    def test_empty_db_returns_zeros_not_errors(self, store):
        s = store.summary()
        assert s["requests"] == 0
        assert s["total_cost_usd"] == 0
        assert s["savings_pct"] == 0.0
        assert s["cache_hit_rate"] == 0.0
        # No rows means no measurement to report, which is None, not zero.
        assert s["avg_latency_ms"] is None
        assert s["p50_latency_ms"] is None


class TestWindowBoundaries:
    def test_since_excludes_older_rows(self, store):
        store.record(_rec(cost=0.01, age_seconds=60))  # inside 1h
        store.record(_rec(cost=0.05, age_seconds=2 * HOUR))  # outside 1h

        assert store.summary(since=time.time() - HOUR)["requests"] == 1
        assert store.summary(since=time.time() - HOUR)["total_cost_usd"] == pytest.approx(0.01)
        assert store.summary()["requests"] == 2

    def test_row_exactly_on_the_boundary_is_included(self, store):
        """`since` is a >= comparison, so the boundary row counts."""
        cutoff = time.time() - HOUR
        store.record(
            UsageRecord(
                tenant_id="acme",
                run_id="run-1",
                task_type="code_generation",
                step_type="completion",
                model_id="gpt-4o-mini",
                cost_usd=0.01,
                timestamp=cutoff,
            )
        )
        assert store.summary(since=cutoff)["requests"] == 1

    def test_timeseries_buckets_are_epoch_aligned(self, store):
        store.record(_rec(age_seconds=30))
        store.record(_rec(age_seconds=90))

        buckets = store.timeseries(bucket_seconds=60)
        assert len(buckets) == 2
        assert all(b["bucket_start"] % 60 == 0 for b in buckets)
        # Oldest first, and empty buckets are absent rather than zero-filled.
        assert buckets[0]["bucket_start"] < buckets[1]["bucket_start"]

    def test_timeseries_groups_rows_in_the_same_bucket(self, store):
        # Anchored to the middle of the current 86400s bucket, not to "now":
        # with now-relative ages these two rows straddle the boundary and
        # land in two buckets whenever the test executes within ~20s after
        # UTC midnight, which is correct behavior but fails this assertion.
        # (Found by running the suite under a faked clock at 00:00:01.)
        day = 86400
        mid_bucket = (time.time() // day) * day + day / 2
        store.record(_rec(cost=0.01, timestamp=mid_bucket))
        store.record(_rec(cost=0.02, timestamp=mid_bucket + 10))

        buckets = store.timeseries(bucket_seconds=day)
        assert len(buckets) == 1
        assert buckets[0]["requests"] == 2
        assert buckets[0]["cost_usd"] == pytest.approx(0.03)


class TestBreakdowns:
    def test_by_model_is_most_expensive_first(self, store):
        store.record(_rec(model="cheap", cost=0.01))
        store.record(_rec(model="pricey", cost=0.50))

        rows = store.by_model()
        assert [r["model_id"] for r in rows] == ["pricey", "cheap"]
        assert sum(r["share_pct"] for r in rows) == pytest.approx(100.0)

    def test_by_task_type_splits_traffic(self, store):
        store.record(_rec(task="code_generation", cost=0.02))
        store.record(_rec(task="summarization", cost=0.01))
        store.record(_rec(task="summarization", cost=0.01))

        rows = {r["task_type"]: r for r in store.by_task_type()}
        assert rows["summarization"]["requests"] == 2
        assert rows["code_generation"]["cost_usd"] == pytest.approx(0.02)

    def test_breakdowns_on_empty_db_are_empty_lists(self, store):
        assert store.by_model() == []
        assert store.by_task_type() == []
        assert store.timeseries() == []


class TestStatsEndpoints:
    def test_summary_endpoint_reports_seeded_rows(self, stats_client, store):
        store.record(_rec(cost=0.01, savings=0.09))

        resp = stats_client.get("/v1/stats/summary?window=24h")
        assert resp.status_code == 200
        data = resp.json()
        assert data["window"] == "24h"
        assert data["requests"] == 1
        assert data["total_cost_usd"] == pytest.approx(0.01)

    @pytest.mark.parametrize("path", ["summary", "timeseries", "models", "tasks"])
    def test_every_endpoint_survives_an_empty_db(self, stats_client, path):
        resp = stats_client.get(f"/v1/stats/{path}?window=24h")
        assert resp.status_code == 200

    @pytest.mark.parametrize("window", ["1h", "24h", "7d", "30d", "all"])
    def test_named_windows_are_accepted(self, stats_client, window):
        assert stats_client.get(f"/v1/stats/summary?window={window}").status_code == 200

    def test_unknown_window_is_rejected(self, stats_client):
        resp = stats_client.get("/v1/stats/summary?window=13f")
        assert resp.status_code == 400
        assert "13f" in resp.json()["detail"]

    def test_window_filters_at_the_endpoint(self, stats_client, store):
        store.record(_rec(cost=0.01, age_seconds=60))
        store.record(_rec(cost=0.05, age_seconds=3 * HOUR))

        assert stats_client.get("/v1/stats/summary?window=1h").json()["requests"] == 1
        assert stats_client.get("/v1/stats/summary?window=24h").json()["requests"] == 2

    def test_timeseries_bucket_width_is_clamped_not_rejected(self, stats_client):
        """An absurd bucket is a caller mistake, not an attack."""
        assert stats_client.get("/v1/stats/timeseries?bucket_seconds=1").json()[
            "bucket_seconds"
        ] == 60
        assert stats_client.get("/v1/stats/timeseries?bucket_seconds=999999").json()[
            "bucket_seconds"
        ] == 86400

    def test_registry_endpoint_lists_models_with_pricing(self, stats_client):
        data = stats_client.get("/v1/stats/registry").json()["data"]
        assert len(data) > 0
        assert {"model_id", "provider", "cost_per_1k_input", "tier"} <= set(data[0])

    def test_stats_unavailable_store_reports_501(self, monkeypatch, stats_client):
        """A custom store implementing only the UsageStore protocol has no
        aggregates — the endpoints say so rather than raising."""
        monkeypatch.setattr(
            server._flux._engine._attribution, "_store", object()  # no .summary
        )
        resp = stats_client.get("/v1/stats/summary")
        assert resp.status_code == 501
        assert "does not support aggregate stats" in resp.json()["detail"]


class TestTenantScoping:
    """In FLUX_SERVER_TOKENS mode a caller sees only their bound tenant's data."""

    @pytest.fixture
    def bound_tokens(self, monkeypatch):
        tokens = {
            "tok-acme": ServerTokenBinding(tenant_id="acme", plan="pro_plan"),
            "tok-globex": ServerTokenBinding(tenant_id="globex", plan="pro_plan"),
        }
        monkeypatch.setattr(server, "SERVER_TOKENS", tokens)
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        return tokens

    def test_store_level_tenant_filter(self, store):
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="globex", cost=0.05))

        assert store.summary(tenant_id="acme")["total_cost_usd"] == pytest.approx(0.01)
        assert store.summary(tenant_id="globex")["total_cost_usd"] == pytest.approx(0.05)
        assert store.summary()["total_cost_usd"] == pytest.approx(0.06)

    def test_endpoint_scopes_to_the_bearer_tokens_tenant(
        self, stats_client, store, bound_tokens
    ):
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="globex", cost=0.05))

        resp = stats_client.get(
            "/v1/stats/summary?window=24h", headers={"Authorization": "Bearer tok-acme"}
        )
        assert resp.status_code == 200
        assert resp.json()["total_cost_usd"] == pytest.approx(0.01)
        assert resp.json()["requests"] == 1

    def test_breakdowns_are_scoped_too(self, stats_client, store, bound_tokens):
        store.record(_rec(tenant="acme", model="acme-model"))
        store.record(_rec(tenant="globex", model="globex-model"))

        data = stats_client.get(
            "/v1/stats/models?window=24h", headers={"Authorization": "Bearer tok-globex"}
        ).json()["data"]
        assert [r["model_id"] for r in data] == ["globex-model"]

    def test_stats_require_auth_when_tokens_are_configured(self, stats_client, bound_tokens):
        assert stats_client.get("/v1/stats/summary").status_code == 401


class TestSharedTokenReadsAreUnscoped:
    """The legacy single shared token pins the WRITE side to one synthetic
    tenant ("shared-token") so budget buckets can't be rotated — but its
    holder is the single operator, documented as authenticated as every
    tenant. Reads must span all tenants: scoping them to the synthetic
    tenant blanked the dashboard, /v1/usage, and /metrics for every row
    carrying a real tenant_id (caught by the docker restart-persistence CI
    step, which seeds tenant "ci" and reads back with the shared token)."""

    @pytest.fixture
    def shared_token(self, monkeypatch):
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "tok-shared")
        return {"Authorization": "Bearer tok-shared"}

    def test_summary_spans_every_tenant(self, stats_client, store, shared_token):
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="ci", cost=0.05, run_id="run-2"))

        resp = stats_client.get("/v1/stats/summary?window=all", headers=shared_token)
        assert resp.status_code == 200
        assert resp.json()["requests"] == 2
        assert resp.json()["total_cost_usd"] == pytest.approx(0.06)

    def test_usage_lists_every_tenant_and_honors_the_filter_param(
        self, stats_client, store, shared_token
    ):
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="ci", cost=0.05, run_id="run-2"))

        body = stats_client.get("/v1/usage", headers=shared_token).json()
        assert body["total"] == 2

        # The operator can still narrow to one tenant on the query string —
        # only FLUX_SERVER_TOKENS mode overrides ?tenant_id=.
        body = stats_client.get("/v1/usage?tenant_id=ci", headers=shared_token).json()
        assert body["total"] == 1
        assert body["data"][0]["tenant_id"] == "ci"

    def test_metrics_render_every_tenant(self, stats_client, store, shared_token, monkeypatch):
        attribution = CostAttribution(store=store)
        monkeypatch.setattr(server._flux._engine, "_attribution", attribution)
        attribution.record(
            tenant_id="acme", run_id="r1", task_type="t", step_type="s",
            model_id="m1", cost_usd=0.01,
        )

        body = stats_client.get("/metrics", headers=shared_token).text
        assert 'tenant_id="acme"' in body

    def test_multi_tenant_mode_still_scopes(self, stats_client, store, monkeypatch):
        # Guard against the fix over-reaching: a FLUX_SERVER_TOKENS binding —
        # even one whose tenant is literally named "shared-token" — stays
        # scoped to its own tenant.
        monkeypatch.setattr(
            server,
            "SERVER_TOKENS",
            {"tok-st": ServerTokenBinding(tenant_id="shared-token", plan="pro_plan")},
        )
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="shared-token", cost=0.05, run_id="run-2"))

        resp = stats_client.get(
            "/v1/stats/summary?window=all", headers={"Authorization": "Bearer tok-st"}
        )
        assert resp.json()["requests"] == 1
        assert resp.json()["total_cost_usd"] == pytest.approx(0.05)


class TestConfigEndpointLeaksNoSecrets:
    """GET /v1/stats/config renders on the dashboard — it must never echo a
    secret. See the SECURITY note on server.stats_config()."""

    def test_no_configured_secret_value_appears_in_the_body(self, monkeypatch, stats_client):
        secrets = {
            "token": "tok-super-secret-value",
            "openai_key": "sk-openai-secret-value",
            "anthropic_key": "sk-ant-secret-value",
            # A Redis URL can embed credentials, so the backend must be reported
            # by name only — never as the connection string.
            "redis_url": "redis://user:redis-password@localhost:6379/0",
        }
        monkeypatch.setattr(
            server, "SERVER_TOKENS", {secrets["token"]: ServerTokenBinding(tenant_id="acme")}
        )
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(
            server._flux,
            "_provider_keys",
            {"openai": secrets["openai_key"], "anthropic": secrets["anthropic_key"]},
        )
        monkeypatch.setattr(server, "RUN_STORE_BACKEND", "redis")
        monkeypatch.setattr(config, "REDIS_URL", secrets["redis_url"])

        resp = stats_client.get(
            "/v1/stats/config", headers={"Authorization": f"Bearer {secrets['token']}"}
        )
        assert resp.status_code == 200
        body = resp.text
        for name, value in secrets.items():
            assert value not in body, f"{name} leaked into /v1/stats/config"

    def test_providers_are_reported_as_configured_yes_no(self, monkeypatch, stats_client):
        monkeypatch.setattr(server._flux, "_provider_keys", {"openai": "sk-secret"})
        monkeypatch.setattr(server._flux, "_api_key", None)

        providers = stats_client.get("/v1/stats/config").json()["providers"]
        assert providers["openai"] is True
        assert providers["anthropic"] is False

    def test_auth_mode_is_a_name_not_a_token(self, monkeypatch, stats_client):
        assert stats_client.get("/v1/stats/config").json()["server"]["auth_mode"] == "none"

        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "tok-shared")
        assert (
            stats_client.get(
                "/v1/stats/config", headers={"Authorization": "Bearer tok-shared"}
            ).json()["server"]["auth_mode"]
            == "shared-token"
        )


class TestSpendDataIsLoopbackOnlyWithoutAuth:
    """Gating /dashboard alone was theatre: the page is only a renderer for
    these endpoints, and /v1/stats/summary served the identical numbers to
    anyone on the network. Every endpoint that returns spend or deployment
    configuration is gated on the peer address when no auth is configured."""

    SPEND_PATHS = [
        "/v1/stats/summary",
        "/v1/stats/timeseries",
        "/v1/stats/models",
        "/v1/stats/tasks",
        "/v1/stats/registry",
        "/v1/stats/config",
        "/v1/usage",
        "/metrics",
    ]

    @staticmethod
    def _client_from(peer: str | None, host: str = "127.0.0.1:8000"):
        return TestClient(
            server.app,
            client=(peer, 50000) if peer else None,
            base_url=f"http://{host}",
        )

    @pytest.mark.parametrize("path", SPEND_PATHS)
    def test_remote_peer_is_refused(self, path, monkeypatch, store):
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        resp = self._client_from("192.168.1.8").get(path)
        assert resp.status_code == 403, f"{path} leaked to a remote peer"
        assert "loopback" in resp.json()["detail"]

    @pytest.mark.parametrize("path", SPEND_PATHS)
    def test_loopback_peer_is_served(self, path, monkeypatch, store):
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        assert self._client_from("127.0.0.1").get(path).status_code == 200

    @pytest.mark.parametrize("path", SPEND_PATHS)
    def test_remote_peer_is_allowed_once_a_token_is_configured(
        self, path, monkeypatch, store
    ):
        """With auth configured the token is the control and remote reads are
        the operator's intent — a Prometheus scrape from another host, say."""
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "tok-remote")

        client = self._client_from("192.168.1.8")
        assert client.get(path).status_code == 401
        assert client.get(
            path, headers={"Authorization": "Bearer tok-remote"}
        ).status_code == 200

    def test_the_proxy_api_stays_reachable_from_anywhere(self, monkeypatch):
        """Only spend/config data is gated. The proxy itself is what the
        deployment is for, and it has its own auth rules."""
        client = self._client_from("192.168.1.8")
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 200

    @pytest.mark.parametrize("path", SPEND_PATHS)
    def test_a_rebound_hostname_is_refused_despite_a_loopback_peer(
        self, path, monkeypatch, store
    ):
        """The peer check alone passes for a DNS-rebinding page driving the
        operator's own browser, and the responses are same-origin to that
        page, so there is no CORS step to save us. The Host header names the
        attacker and is what actually closes this."""
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        resp = self._client_from("127.0.0.1", host="evil.example:8000").get(path)
        assert resp.status_code == 403, f"{path} leaked to a rebound hostname"

    @pytest.mark.parametrize("path", SPEND_PATHS)
    def test_an_operator_named_host_is_served(self, path, monkeypatch, store):
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        monkeypatch.setattr(server, "SERVER_ALLOWED_HOSTS", frozenset({"flux.internal"}))
        assert (
            self._client_from("127.0.0.1", host="flux.internal").get(path).status_code == 200
        )

    def test_missing_peer_is_refused(self, monkeypatch, store):
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        assert self._client_from(None).get("/v1/stats/summary").status_code == 403


class TestTenantBreakdown:
    """by_tenant() + GET /v1/stats/tenants, added for the dashboard's tenant panel."""

    def test_groups_spend_by_tenant_most_expensive_first(self, store):
        store.record(_rec(tenant="acme", cost=0.01, savings=0.09))
        store.record(_rec(tenant="globex", cost=0.50, savings=1.20))
        store.record(_rec(tenant="globex", cost=0.05, savings=0.30))

        rows = store.by_tenant()
        assert [r["tenant_id"] for r in rows] == ["globex", "acme"]
        assert rows[0]["requests"] == 2
        assert rows[0]["cost_usd"] == pytest.approx(0.55)
        assert rows[0]["estimated_savings_usd"] == pytest.approx(1.50)
        assert sum(r["share_pct"] for r in rows) == pytest.approx(100.0)

    def test_untagged_traffic_is_labelled_not_dropped(self, store):
        """Rows must always sum to the headline total, so a null tenant still
        gets a row rather than silently vanishing from the breakdown."""
        store.record(_rec(tenant=None, cost=0.02))
        store.record(_rec(tenant="acme", cost=0.03))

        rows = store.by_tenant()
        assert {r["tenant_id"] for r in rows} == {None, "acme"}
        assert sum(r["cost_usd"] for r in rows) == pytest.approx(
            store.summary()["total_cost_usd"]
        )

    def test_counts_distinct_runs_and_models(self, store):
        store.record(_rec(tenant="acme", run_id="r1", model="gpt-4o-mini"))
        store.record(_rec(tenant="acme", run_id="r1", model="gpt-4o-mini"))
        store.record(_rec(tenant="acme", run_id="r2", model="claude-haiku"))

        row = store.by_tenant()[0]
        assert row["runs"] == 2
        assert row["distinct_models"] == 2

    def test_empty_db_is_an_empty_list(self, store):
        assert store.by_tenant() == []

    def test_endpoint_returns_the_breakdown(self, stats_client, store):
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="globex", cost=0.05))

        resp = stats_client.get("/v1/stats/tenants?window=24h")
        assert resp.status_code == 200
        assert resp.json()["window"] == "24h"
        assert {r["tenant_id"] for r in resp.json()["data"]} == {"acme", "globex"}

    def test_endpoint_survives_an_empty_db(self, stats_client):
        assert stats_client.get("/v1/stats/tenants?window=24h").status_code == 200

    def test_window_filters_the_breakdown(self, stats_client, store):
        store.record(_rec(tenant="acme", age_seconds=60))
        store.record(_rec(tenant="globex", age_seconds=3 * HOUR))

        assert len(stats_client.get("/v1/stats/tenants?window=1h").json()["data"]) == 1
        assert len(stats_client.get("/v1/stats/tenants?window=24h").json()["data"]) == 2

    def test_cannot_enumerate_other_tenants(self, stats_client, store, monkeypatch):
        """The whole point of the scoping: a bound token must see ONE row, its
        own — otherwise this endpoint becomes a directory of everyone's spend."""
        monkeypatch.setattr(
            server,
            "SERVER_TOKENS",
            {
                "tok-acme": ServerTokenBinding(tenant_id="acme", plan="pro_plan"),
                "tok-globex": ServerTokenBinding(tenant_id="globex", plan="pro_plan"),
            },
        )
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        store.record(_rec(tenant="acme", cost=0.01))
        store.record(_rec(tenant="globex", cost=0.50))

        data = stats_client.get(
            "/v1/stats/tenants?window=24h", headers={"Authorization": "Bearer tok-acme"}
        ).json()["data"]
        assert [r["tenant_id"] for r in data] == ["acme"]
        assert data[0]["cost_usd"] == pytest.approx(0.01)

    def test_unauthenticated_tenants_endpoint_is_loopback_only(self, store):
        """Same refusal as every other spend endpoint."""
        remote = TestClient(
            server.app, client=("203.0.113.9", 50000), base_url="http://127.0.0.1:8000"
        )
        assert remote.get("/v1/stats/tenants").status_code == 403


class TestUsageEndpointProjection:
    """The console's activity feed reads latency off /v1/usage; the column is
    persisted but was missing from the endpoint's projection."""

    def test_usage_rows_expose_routing_telemetry(self, stats_client, store):
        store.record(_rec(latency=250.0, cache_hit=True, fallback_used=True))

        row = stats_client.get("/v1/usage?limit=1").json()["data"][0]
        assert row["latency_ms"] == pytest.approx(250.0)
        assert row["cache_hit"] is True
        assert row["fallback_used"] is True

    def test_missing_telemetry_reads_back_as_null(self, stats_client, store):
        """Rows written before those columns existed must not break the feed."""
        store.record(_rec(latency=None))

        row = stats_client.get("/v1/usage?limit=1").json()["data"][0]
        assert row["latency_ms"] is None


class TestConfigWithholdsOperatorFieldsFromTenantTokens:
    """A FLUX_SERVER_TOKENS bearer token belongs to one customer, not to the
    operator. /v1/stats/config still answers it, but without the deployment
    shape: bind address, worker count, data dir, usage db, how many other
    tenants exist, the store backends, and other plans' budgets."""

    OPERATOR_ONLY = (
        "host",
        "port",
        "workers",
        "tenant_count",
        "run_store_backend",
        "budget_store_backend",
        "data_dir",
        "usage_db",
        "usage_db_persistent",
    )

    @pytest.fixture
    def bound(self, monkeypatch):
        tokens = {
            "tok-acme": ServerTokenBinding(tenant_id="acme", plan="free_plan"),
            "tok-globex": ServerTokenBinding(tenant_id="globex", plan="pro_plan"),
        }
        monkeypatch.setattr(server, "SERVER_TOKENS", tokens)
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        return tokens

    def test_bound_tenant_gets_no_operator_fields(self, stats_client, bound):
        body = stats_client.get(
            "/v1/stats/config", headers={"Authorization": "Bearer tok-acme"}
        ).json()

        for field in self.OPERATOR_ONLY:
            assert field not in body["server"], f"{field} leaked to a tenant token"
        # What a caller legitimately needs is still there.
        assert body["server"]["auth_mode"] == "bound-tokens"
        assert body["server"]["max_body_bytes"] > 0
        assert "providers" in body and "run_limits" in body and "rate_limit" in body

    def test_bound_tenant_sees_only_its_own_plan(self, stats_client, bound):
        acme = stats_client.get(
            "/v1/stats/config", headers={"Authorization": "Bearer tok-acme"}
        ).json()
        globex = stats_client.get(
            "/v1/stats/config", headers={"Authorization": "Bearer tok-globex"}
        ).json()

        assert list(acme["budgets"]["plans"]) == ["free_plan"]
        assert list(globex["budgets"]["plans"]) == ["pro_plan"]

    def test_tenant_count_cannot_be_used_to_enumerate_other_tenants(self, stats_client, bound):
        body = stats_client.get(
            "/v1/stats/config", headers={"Authorization": "Bearer tok-acme"}
        ).json()

        assert "tenant_count" not in body["server"]
        assert "globex" not in json.dumps(body)

    def test_loopback_operator_still_sees_everything(self, stats_client, monkeypatch):
        """No auth configured + loopback peer = the operator's own box."""
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", False)

        body = stats_client.get("/v1/stats/config").json()

        for field in self.OPERATOR_ONLY:
            assert field in body["server"], f"{field} missing from the operator view"
        assert body["budgets"]["plans"] == server.BUDGET_LIMITS

    def test_shared_token_still_sees_everything(self, stats_client, monkeypatch):
        """Shared-token mode is documented as a single operator, and is
        deliberately not a read scope -- same reasoning as
        TestSharedTokenReadsAreUnscoped."""
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "shared-secret")

        body = stats_client.get(
            "/v1/stats/config", headers={"Authorization": "Bearer shared-secret"}
        ).json()

        for field in self.OPERATOR_ONLY:
            assert field in body["server"], f"{field} missing for the shared-token operator"
        assert body["budgets"]["plans"] == server.BUDGET_LIMITS


class TestRegistryWithholdsLiveLoadFromTenantTokens:
    """Strix vuln-0001 (CWE-863, CVSS 4.3): current_load_rpm is a
    process-global sliding-window counter -- ModelRegistry.update_load() is
    called for EVERY tenant's routed request, so returning it verbatim let
    one FLUX_SERVER_TOKENS-bound tenant observe load another tenant's
    traffic generated. Every other field on this endpoint is static catalog
    data (same content as models.json) and stays unscoped -- same
    reasoning as TestConfigWithholdsOperatorFieldsFromTenantTokens above."""

    @pytest.fixture
    def bound(self, monkeypatch):
        tokens = {
            "tok-acme": ServerTokenBinding(tenant_id="acme", plan="free_plan"),
            "tok-globex": ServerTokenBinding(tenant_id="globex", plan="pro_plan"),
        }
        monkeypatch.setattr(server, "SERVER_TOKENS", tokens)
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        return tokens

    @staticmethod
    def _bump_load_and_get_row(stats_client, headers) -> dict:
        registry = server._flux._engine._registry
        model_id = registry.all_available_models()[0].model_id
        registry.update_load(model_id)
        try:
            body = stats_client.get("/v1/stats/registry", headers=headers).json()
            return next(r for r in body["data"] if r["model_id"] == model_id)
        finally:
            registry.reset_load_tracking()

    def test_tenant_bound_token_gets_null_load(self, stats_client, bound):
        row = self._bump_load_and_get_row(
            stats_client, {"Authorization": "Bearer tok-acme"}
        )
        assert row["current_load_rpm"] is None
        # Static catalog fields are untouched -- this is a load-only redaction.
        assert row["cost_per_1k_input"] is not None
        assert row["rate_limit_rpm"] is not None
        assert row["is_available"] is True

    def test_cross_tenant_load_is_not_the_real_value(self, stats_client, bound):
        """The exact Strix repro: tenant B's traffic (update_load) must not
        surface as a real number to tenant A's token."""
        row = self._bump_load_and_get_row(
            stats_client, {"Authorization": "Bearer tok-acme"}
        )
        assert row["current_load_rpm"] != 1

    def test_loopback_operator_still_sees_live_load(self, stats_client, monkeypatch):
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", False)
        row = self._bump_load_and_get_row(stats_client, {})
        assert row["current_load_rpm"] == 1

    def test_shared_token_still_sees_live_load(self, stats_client, monkeypatch):
        """Deliberately not a read scope -- same reasoning as
        TestSharedTokenReadsAreUnscoped / the config endpoint's precedent."""
        monkeypatch.setattr(server, "SERVER_TOKENS", {})
        monkeypatch.setattr(server, "SERVER_REQUIRE_AUTH", True)
        monkeypatch.setattr(server, "_SERVER_TOKEN", "shared-secret")
        row = self._bump_load_and_get_row(
            stats_client, {"Authorization": "Bearer shared-secret"}
        )
        assert row["current_load_rpm"] == 1
