"""
File: router/tests/test_attribution.py

Purpose:
Tests for Task 7 — per-run and per-tenant cost attribution
(router/attribution.py) plus its wiring into Flux.complete() /
RoutingEngine._proxy_execute() and the proxy's GET /v1/usage and GET
/metrics endpoints.

How to run:
  pytest -v router/tests/test_attribution.py
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.attribution import CostAttribution, SqliteUsageStore, UsageRecord
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.flux import Flux
from router.model_registry import ModelRegistry
from router.provider_caller import ProviderResult
from router.routing_engine import RoutingEngine


def rr(coro):
    return asyncio.run(coro)


def _engine() -> RoutingEngine:
    registry = ModelRegistry()
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _flux() -> Flux:
    return Flux(_engine(), api_key="test-key")


def _pr(text: str) -> ProviderResult:
    return ProviderResult(
        text=text, input_tokens=None, output_tokens=None, usage_source="estimated"
    )


class TestSqliteUsageStore:
    def test_record_and_query_round_trip(self):
        store = SqliteUsageStore(":memory:")
        store.record(
            UsageRecord("tenant-a", "run-1", "code_generation", "plan", "m1", 0.01, 1000.0)
        )
        store.record(
            UsageRecord("tenant-b", "run-2", "summarization", "extract", "m2", 0.02, 1001.0)
        )

        all_records = store.query()
        assert len(all_records) == 2

        tenant_a = store.query(tenant_id="tenant-a")
        assert len(tenant_a) == 1
        assert tenant_a[0].model_id == "m1"

    def test_query_filters_by_run_id(self):
        store = SqliteUsageStore(":memory:")
        store.record(UsageRecord("t", "run-1", "x", "y", "m1", 0.01, 1000.0))
        store.record(UsageRecord("t", "run-2", "x", "y", "m2", 0.02, 1001.0))
        assert len(store.query(run_id="run-1")) == 1

    def test_pagination(self):
        store = SqliteUsageStore(":memory:")
        for i in range(10):
            store.record(UsageRecord("t", None, "x", "y", f"m{i}", 0.001, float(i)))
        page1 = store.query(limit=5, offset=0)
        page2 = store.query(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert store.count() == 10
        assert {r.model_id for r in page1}.isdisjoint({r.model_id for r in page2})

    def test_record_does_not_block_on_disk_write(self):
        """Regression: record() used to execute()+commit() (an fsync)
        synchronously on the calling thread — the HTTP proxy's hot request
        path. It must now return immediately even while the writer thread
        is mid-write; simulated here by holding the same lock the writer
        thread needs before it can commit."""
        import time

        store = SqliteUsageStore(":memory:")
        store._lock.acquire()
        try:
            t0 = time.perf_counter()
            store.record(UsageRecord("t", "r", "x", "y", "m", 0.01, 1000.0))
            elapsed = time.perf_counter() - t0
            # Generous bound on purpose: a real regression (record() waiting
            # on the held lock) doesn't run slow, it deadlocks — the pytest
            # timeout catches that. A tight 50ms bound only added CI-runner
            # scheduling noise as a failure mode.
            assert elapsed < 1.0, f"record() blocked the caller for {elapsed:.3f}s"
        finally:
            store._lock.release()

        # query() flushes first, so the record is still visible once queried.
        assert len(store.query()) == 1

    def test_record_survives_queue_overflow_without_raising(self, monkeypatch):
        """A caller (an async request handler) must never see an exception
        from record() just because the writer thread fell behind."""
        import router.attribution as attribution_module

        store = SqliteUsageStore(":memory:")
        # Simulate a full queue without actually enqueuing thousands of items.
        monkeypatch.setattr(store, "_queue", attribution_module.queue.Queue(maxsize=1))
        store._queue.put_nowait(UsageRecord("t", "r", "x", "y", "m0", 0.01, 0.0))
        store.record(UsageRecord("t", "r", "x", "y", "m1", 0.01, 1.0))  # must not raise
        assert store.dropped_records == 1

    def test_writer_batches_available_records_into_one_transaction(self):
        store = SqliteUsageStore(":memory:")
        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        store._lock.acquire()
        try:
            for i in range(5):
                store.record(UsageRecord("t", "r", "x", "y", f"m{i}", 0.01, float(i)))
        finally:
            store._lock.release()

        assert store.count() == 5
        assert sum(statement == "BEGIN " for statement in statements) <= 2

    def test_no_prompt_or_response_fields_exist(self):
        """UsageRecord has no field that could ever hold prompt/response text.

        This is an allowlist on purpose: adding a field here is a deliberate
        act that has to be justified against SECURITY_ARCHITECTURE.md, not
        something that happens by accident. Every entry below is a number, a
        bool, an identifier the operator configured, or a closed set of
        values — never anything derived from prompt or completion content.
        routing_priority in particular is a pydantic Literal on
        RoutingRequest (schemas.py), so it cannot carry caller free text.
        """
        fields = set(UsageRecord.__dataclass_fields__.keys())
        assert fields == {
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
            # Routing telemetry for the local dashboard.
            "latency_ms",
            "decision_latency_ms",
            "estimated_savings_usd",
            "complexity_score",
            "cache_hit",
            "routing_priority",
            "fallback_used",
        }


class TestSqliteUsageStoreMigration:
    """Regression: an on-disk usage.db predating usage_source/input_tokens/
    output_tokens must open cleanly and migrate in place — see
    MIGRATIONS.md's "Attribution Usage Database" section."""

    def _make_old_schema_db(self, path: str) -> None:
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute(
            """
            CREATE TABLE usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT,
                run_id TEXT,
                task_type TEXT,
                step_type TEXT,
                model_id TEXT NOT NULL,
                cost_usd REAL NOT NULL,
                timestamp REAL NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO usage (tenant_id, run_id, task_type, step_type, model_id, "
            "cost_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old-tenant", "old-run", "code_generation", "plan", "gpt-4o", 0.05, 1000.0),
        )
        conn.commit()
        conn.close()

    def test_old_schema_file_opens_and_migrates_cleanly(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        self._make_old_schema_db(db_path)

        store = SqliteUsageStore(db_path)  # must not raise
        records = store.query(tenant_id="old-tenant")
        assert len(records) == 1
        record = records[0]
        # A pre-migration row has no usage_source column at insert time, but
        # the ADD COLUMN ... DEFAULT 'estimated' backfills it automatically.
        assert record.usage_source == "estimated"
        assert record.input_tokens is None
        assert record.output_tokens is None
        assert record.cost_usd == 0.05

    def test_new_rows_after_migration_carry_actual_usage(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        self._make_old_schema_db(db_path)

        store = SqliteUsageStore(db_path)
        store.record(
            UsageRecord(
                "new-tenant", "new-run", "code_generation", "plan", "gpt-4o", 0.02, 2000.0,
                usage_source="provider", input_tokens=100, output_tokens=50,
            )
        )
        records = store.query(tenant_id="new-tenant")
        assert len(records) == 1
        assert records[0].usage_source == "provider"
        assert records[0].input_tokens == 100
        assert records[0].output_tokens == 50

    def test_migration_is_idempotent_across_reopen(self, tmp_path):
        db_path = str(tmp_path / "usage.db")
        self._make_old_schema_db(db_path)

        SqliteUsageStore(db_path)  # first open: migrates
        store2 = SqliteUsageStore(db_path)  # second open: must not raise (columns already exist)
        assert len(store2.query()) == 1


class TestCostAttribution:
    def test_record_updates_prometheus_counters(self):
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record(
            tenant_id="acme",
            run_id="r1",
            task_type="code_generation",
            step_type="plan",
            model_id="gpt-5",
            cost_usd=0.05,
        )
        body = attribution.render_prometheus()
        assert 'flux_cost_usd_total{tenant_id="acme",model_id="gpt-5"} 0.050000' in body
        assert 'flux_run_steps{tenant_id="acme"} 1' in body
        assert "flux_attribution_records_dropped_total 0" in body

    def test_budget_exceeded_counter(self):
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record_budget_exceeded("acme")
        attribution.record_budget_exceeded("acme")
        body = attribution.render_prometheus()
        assert 'flux_budget_exceeded_total{tenant_id="acme"} 2' in body

    def test_tenant_id_with_quote_is_escaped_not_injected(self):
        """Regression: X-Flux-Tenant-Id is caller-controlled and was
        interpolated into the Prometheus label unescaped — a tenant_id
        containing a `"` broke the label syntax for every line after it,
        which most scrapers reject wholesale rather than skip one metric."""
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record(
            tenant_id='foo"bar\\baz',
            run_id="r1",
            task_type="x",
            step_type="y",
            model_id="m",
            cost_usd=0.01,
        )
        body = attribution.render_prometheus()
        assert 'tenant_id="foo\\"bar\\\\baz"' in body
        # No stray unescaped quote breaks the rest of the line into a new label.
        assert 'tenant_id="foo"bar' not in body

    def test_missing_tenant_id_buckets_as_unknown(self):
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record(
            tenant_id=None, run_id=None, task_type="x", step_type="y", model_id="m", cost_usd=0.01
        )
        assert 'tenant_id="unknown"' in attribution.render_prometheus()

    def test_usage_returns_records_and_total_count(self):
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        for i in range(3):
            attribution.record(
                tenant_id="acme",
                run_id="r1",
                task_type="x",
                step_type="y",
                model_id=f"m{i}",
                cost_usd=0.01,
            )
        records, total = attribution.usage(tenant_id="acme", limit=2)
        assert total == 3
        assert len(records) == 2

    def test_cardinality_cap_overflows_new_label_combos(self, monkeypatch):
        import router.attribution as attribution_module

        monkeypatch.setattr(attribution_module, "ATTRIBUTION_METRICS_MAX_LABEL_COMBOS", 2)
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record(
            tenant_id="t1", run_id=None, task_type="x", step_type="y", model_id="m1", cost_usd=0.01
        )
        attribution.record(
            tenant_id="t2", run_id=None, task_type="x", step_type="y", model_id="m1", cost_usd=0.01
        )
        # Cap (2) reached — a third NEW (tenant, model) pair overflows.
        attribution.record(
            tenant_id="t3", run_id=None, task_type="x", step_type="y", model_id="m1", cost_usd=0.05
        )
        body = attribution.render_prometheus()
        assert 'tenant_id="t3"' not in body
        assert 'flux_cost_usd_total{tenant_id="_overflow_",model_id="_overflow_"} 0.050000' in body

    def test_repeat_label_combo_does_not_count_against_cap(self, monkeypatch):
        import router.attribution as attribution_module

        monkeypatch.setattr(attribution_module, "ATTRIBUTION_METRICS_MAX_LABEL_COMBOS", 1)
        attribution = CostAttribution(store=SqliteUsageStore(":memory:"))
        attribution.record(
            tenant_id="t1", run_id=None, task_type="x", step_type="y", model_id="m1", cost_usd=0.01
        )
        attribution.record(
            tenant_id="t1", run_id=None, task_type="x", step_type="y", model_id="m1", cost_usd=0.02
        )
        body = attribution.render_prometheus()
        assert 'flux_cost_usd_total{tenant_id="t1",model_id="m1"} 0.030000' in body


class TestAttributionWiredIntoFluxComplete:
    def test_complete_records_usage(self):
        flux = _flux()
        flux._call_model = AsyncMock(return_value=_pr("ok"))  # type: ignore[method-assign]
        rr(flux.complete("hi", user_id="u1", tenant_id="acme", exploration_rate=0.0))

        records, total = flux._engine._attribution.usage(tenant_id="acme")
        assert total == 1
        assert records[0].cost_usd >= 0.0

    def test_no_tenant_id_still_records_under_unknown(self):
        flux = _flux()
        flux._call_model = AsyncMock(return_value=_pr("ok"))  # type: ignore[method-assign]
        rr(flux.complete("hi", user_id="u1", exploration_rate=0.0))
        body = flux._engine._attribution.render_prometheus()
        assert 'tenant_id="unknown"' in body


class TestDashboardTelemetryColumns:
    """The columns added for the local dashboard (latency, savings, routing
    detail). Costs and metadata only — nothing here derives from prompt text."""

    def test_round_trip(self):
        store = SqliteUsageStore(":memory:")
        store.record(
            UsageRecord(
                tenant_id="acme",
                run_id="run-1",
                task_type="code_generation",
                step_type="completion",
                model_id="m1",
                cost_usd=0.01,
                timestamp=1000.0,
                latency_ms=250.5,
                decision_latency_ms=1.25,
                estimated_savings_usd=0.09,
                complexity_score=0.42,
                cache_hit=True,
                routing_priority="cost_optimized",
                fallback_used=True,
            )
        )

        rec = store.query()[0]
        assert rec.latency_ms == 250.5
        assert rec.decision_latency_ms == 1.25
        assert rec.estimated_savings_usd == 0.09
        assert rec.complexity_score == 0.42
        assert rec.routing_priority == "cost_optimized"

    def test_booleans_survive_sqlites_lack_of_a_bool_type(self):
        store = SqliteUsageStore(":memory:")
        store.record(
            UsageRecord(
                tenant_id=None, run_id=None, task_type="t", step_type="s", model_id="m",
                cost_usd=0.01, timestamp=1000.0, cache_hit=True, fallback_used=False,
            )
        )

        rec = store.query()[0]
        assert rec.cache_hit is True
        assert rec.fallback_used is False

    def test_telemetry_is_optional(self):
        """The library dispatch path doesn't measure wall-clock latency, so
        every one of these stays writable as None."""
        store = SqliteUsageStore(":memory:")
        store.record(
            UsageRecord(
                tenant_id=None, run_id=None, task_type="t", step_type="s", model_id="m",
                cost_usd=0.01, timestamp=1000.0,
            )
        )

        rec = store.query()[0]
        assert rec.latency_ms is None
        assert rec.decision_latency_ms is None
        assert rec.estimated_savings_usd is None
        assert rec.complexity_score is None
        assert rec.routing_priority is None
        assert rec.cache_hit is False
        assert rec.fallback_used is False


class TestSchemaMigration:
    """A database written by an older Flux must open in place, not blow up.
    Anyone self-hosting already has one; `flux serve` opens it on every start."""

    ORIGINAL_COLUMNS = (
        "tenant_id TEXT", "run_id TEXT", "task_type TEXT", "step_type TEXT",
        "model_id TEXT NOT NULL", "cost_usd REAL NOT NULL", "timestamp REAL NOT NULL",
    )

    def _legacy_db(self, path: str, *, rows: int = 1) -> None:
        """Create a database with the pre-telemetry schema and seed it."""
        import sqlite3

        conn = sqlite3.connect(path)
        conn.execute(
            "CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            + ", ".join(self.ORIGINAL_COLUMNS)
            + ")"
        )
        for i in range(rows):
            conn.execute(
                "INSERT INTO usage (tenant_id, run_id, task_type, step_type, model_id, "
                "cost_usd, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("acme", f"run-{i}", "code_generation", "completion", "m1", 0.01, 1000.0 + i),
            )
        conn.commit()
        conn.close()

    def test_old_database_opens_and_migrates(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        self._legacy_db(path)

        store = SqliteUsageStore(path)
        columns = {r[1] for r in store._conn.execute("PRAGMA table_info(usage)").fetchall()}
        for column, _ddl in SqliteUsageStore._MIGRATED_COLUMNS:
            assert column in columns

    def test_pre_migration_rows_read_back_as_null(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        self._legacy_db(path)

        rec = SqliteUsageStore(path).query()[0]
        assert rec.cost_usd == 0.01  # the original data is intact
        assert rec.latency_ms is None
        assert rec.estimated_savings_usd is None
        assert rec.routing_priority is None
        # NOT NULL DEFAULT columns backfill rather than reading NULL.
        assert rec.usage_source == "estimated"
        assert rec.cache_hit is False
        assert rec.fallback_used is False

    def test_new_rows_land_alongside_migrated_ones(self, tmp_path):
        path = str(tmp_path / "legacy.db")
        self._legacy_db(path)

        store = SqliteUsageStore(path)
        store.record(
            UsageRecord(
                tenant_id="acme", run_id="run-new", task_type="t", step_type="s",
                model_id="m1", cost_usd=0.02, timestamp=2000.0, latency_ms=99.0,
            )
        )

        assert store.count() == 2
        newest = store.query()[0]
        assert newest.run_id == "run-new"
        assert newest.latency_ms == 99.0

    def test_aggregates_tolerate_a_mix_of_old_and_new_rows(self, tmp_path):
        """Pre-migration rows have no latency; they must be skipped in the
        latency stats rather than counted as zero."""
        path = str(tmp_path / "legacy.db")
        self._legacy_db(path, rows=2)

        store = SqliteUsageStore(path)
        store.record(
            UsageRecord(
                tenant_id="acme", run_id="run-new", task_type="t", step_type="s",
                model_id="m1", cost_usd=0.02, timestamp=2000.0, latency_ms=100.0,
                estimated_savings_usd=0.08,
            )
        )

        summary = store.summary()
        assert summary["requests"] == 3
        assert summary["avg_latency_ms"] == 100.0
        assert summary["estimated_savings_usd"] == 0.08

    def test_reopening_an_already_migrated_database_is_a_no_op(self, tmp_path):
        path = str(tmp_path / "flux.db")
        first = SqliteUsageStore(path)
        first.record(
            UsageRecord(
                tenant_id="acme", run_id="r1", task_type="t", step_type="s", model_id="m1",
                cost_usd=0.01, timestamp=1000.0, latency_ms=50.0,
            )
        )
        first._flush()

        # What `flux serve` does on every restart.
        reopened = SqliteUsageStore(path)
        assert reopened.count() == 1
        assert reopened.query()[0].latency_ms == 50.0
