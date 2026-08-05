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
    run_id: str | None = "run-1",
    latency: float | None = 100.0,
    savings: float | None = 0.09,
    usage_source: str = "estimated",
    cache_hit: bool = False,
    fallback_used: bool = False,
) -> UsageRecord:
    """A usage row `age_seconds` in the past, with dashboard-relevant fields set."""
    return UsageRecord(
        tenant_id=tenant,
        run_id=run_id,
        task_type=task,
        step_type="completion",
        model_id=model,
        cost_usd=cost,
        timestamp=time.time() - age_seconds,
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
    # Present a loopback peer: the spend endpoints refuse a non-loopback
    # caller when no auth is configured, and TestClient's default peer host
    # is the literal string "testclient", which is not a parseable address.
    return TestClient(server.app, client=("127.0.0.1", 50000))


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
        store.record(_rec(cost=0.01, age_seconds=10))
        store.record(_rec(cost=0.02, age_seconds=20))

        buckets = store.timeseries(bucket_seconds=86400)
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
    def _client_from(peer: str | None):
        return TestClient(server.app, client=(peer, 50000) if peer else None)

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

    def test_missing_peer_is_refused(self, monkeypatch, store):
        monkeypatch.setattr(server._flux._engine, "_attribution", CostAttribution(store=store))
        assert self._client_from(None).get("/v1/stats/summary").status_code == 403
