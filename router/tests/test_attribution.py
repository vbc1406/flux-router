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
            assert elapsed < 0.05, f"record() blocked the caller for {elapsed:.3f}s"
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

    def test_no_prompt_or_response_fields_exist(self):
        """UsageRecord has no field that could ever hold prompt/response text."""
        fields = set(UsageRecord.__dataclass_fields__.keys())
        assert fields == {
            "tenant_id",
            "run_id",
            "task_type",
            "step_type",
            "model_id",
            "cost_usd",
            "timestamp",
        }


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
        flux._call_model = AsyncMock(return_value="ok")  # type: ignore[method-assign]
        rr(flux.complete("hi", user_id="u1", tenant_id="acme", exploration_rate=0.0))

        records, total = flux._engine._attribution.usage(tenant_id="acme")
        assert total == 1
        assert records[0].cost_usd >= 0.0

    def test_no_tenant_id_still_records_under_unknown(self):
        flux = _flux()
        flux._call_model = AsyncMock(return_value="ok")  # type: ignore[method-assign]
        rr(flux.complete("hi", user_id="u1", exploration_rate=0.0))
        body = flux._engine._attribution.render_prometheus()
        assert 'tenant_id="unknown"' in body
