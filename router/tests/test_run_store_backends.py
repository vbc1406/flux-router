"""
File: router/tests/test_run_store_backends.py

Purpose:
Shared behavior tests for the RunStore Protocol (router/run_budget.py),
parametrized across every backend that implements it: InMemoryRunStore (the
default) and RedisRunStore (opt-in via FLUX_RUN_STORE=redis, for
multi-worker deployments). The point of this file is to prove both backends
honor the SAME contract — get/set/delete round-trip, LRU eviction at
max_entries, and TTL-based sweep_expired() — not just that each one works in
isolation.

RedisRunStore is exercised against fakeredis (an in-memory fake of the Redis
protocol) rather than a real server, so these tests run without any external
infrastructure. If fakeredis isn't installed, the redis-backed parametrization
is skipped (not failed) — see the `_backend` fixture below.

How to run:
  pytest -v router/tests/test_run_store_backends.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from router.run_budget import InMemoryRunStore, RunBudget, RunBudgetExceeded, RunLimits

try:
    import fakeredis

    from router.run_budget import RedisRunStore

    _HAS_FAKEREDIS = True
except ImportError:
    _HAS_FAKEREDIS = False


def _make_memory_store(max_entries: int = 200) -> InMemoryRunStore:
    return InMemoryRunStore(max_entries=max_entries)


def _make_redis_store(max_entries: int = 200) -> "RedisRunStore":
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisRunStore(redis_client=client, max_entries=max_entries, ttl_seconds=3600.0)


_BACKENDS = [
    pytest.param(_make_memory_store, id="in-memory"),
    pytest.param(
        _make_redis_store,
        id="redis",
        marks=pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed"),
    ),
]


@pytest.fixture(params=_BACKENDS)
def store_factory(request):
    return request.param


class TestRunStoreContract:
    """Backend-agnostic behavior every RunStore implementation must satisfy."""

    def test_get_on_unknown_run_returns_none(self, store_factory):
        store = store_factory()
        assert store.get("nonexistent") is None

    def test_set_then_get_round_trips_state(self, store_factory):
        store = store_factory()
        rb = RunBudget(store=store)
        rb.check_before_dispatch("run-1")
        rb.record_step("run-1", "gpt-cheap", 0.01, 50)
        rb.record_step("run-1", "gpt-cheap", 0.02, 60)

        cost, steps = rb.snapshot("run-1")
        assert cost == pytest.approx(0.03)
        assert steps == 2

    def test_delete_removes_state(self, store_factory):
        store = store_factory()
        rb = RunBudget(store=store)
        rb.check_before_dispatch("run-1")
        rb.record_step("run-1", "m", 0.01, 10)
        store.delete("run-1")
        assert store.get("run-1") is None
        cost, steps = rb.snapshot("run-1")
        assert (cost, steps) == (0.0, 0)

    def test_lru_cap_evicts_least_recently_touched(self, store_factory):
        store = store_factory(max_entries=3)
        rb = RunBudget(store=store)

        for run_id in ("a", "b", "c"):
            rb.check_before_dispatch(run_id)
        # Touch "a" again so it's no longer the least-recently-touched.
        rb.check_before_dispatch("a")
        # Adding a 4th distinct run must evict the LRU entry ("b").
        rb.check_before_dispatch("d")

        assert store.get("a") is not None
        assert store.get("d") is not None
        assert store.get("b") is None

    def test_sweep_expired_evicts_idle_runs_only(self, store_factory):
        store = store_factory()
        rb = RunBudget(store=store)
        rb.check_before_dispatch("idle-run")
        time.sleep(0.15)
        rb.check_before_dispatch("fresh-run")

        evicted = store.sweep_expired(ttl_seconds=0.05)

        assert evicted >= 1
        assert store.get("idle-run") is None
        assert store.get("fresh-run") is not None

    def test_degradation_ladder_behaves_identically_across_backends(self, store_factory):
        store = store_factory()
        rb = RunBudget(store=store)
        rb.start("run-1", RunLimits(max_cost_usd=1.0, max_steps=1000, max_tokens=10**9))

        rb.record_step("run-1", "m", 0.65, 10)  # 65% -> below RUN_DEGRADE_THRESHOLD (0.70)
        assert rb.check_before_dispatch("run-1") == "ok"

        rb.record_step("run-1", "m", 0.10, 10)  # 75% -> degraded
        assert rb.check_before_dispatch("run-1") == "degraded"

        rb.record_step("run-1", "m", 0.16, 10)  # 91% -> warning
        assert rb.check_before_dispatch("run-1") == "warning"

    def test_raises_before_dispatch_once_a_limit_is_met(self, store_factory):
        store = store_factory()
        rb = RunBudget(store=store)
        rb.start("run-1", RunLimits(max_cost_usd=0.01, max_steps=1000, max_tokens=10**9))
        rb.record_step("run-1", "m", 0.02, 10)

        with pytest.raises(RunBudgetExceeded) as excinfo:
            rb.check_before_dispatch("run-1")
        assert excinfo.value.summary["exceeded_reason"] == "cost"


@pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisRunStoreSpecifics:
    """Coverage for RedisRunStore behavior with no InMemoryRunStore analogue."""

    def test_requires_redis_package_or_explicit_client(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError("no redis installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="RedisRunStore requires"):
            RedisRunStore()

    def test_state_survives_a_fresh_store_instance_against_the_same_client(self):
        """Simulates two different worker processes sharing one Redis: a
        second RedisRunStore instance (standing in for a second worker) built
        against the SAME underlying client must see state the first one
        wrote — this is the entire point of the Redis backend."""
        client = fakeredis.FakeRedis(decode_responses=True)
        store_a = RedisRunStore(redis_client=client, max_entries=200, ttl_seconds=3600.0)
        rb_a = RunBudget(store=store_a)
        rb_a.check_before_dispatch("shared-run")
        rb_a.record_step("shared-run", "gpt-cheap", 0.05, 100)

        store_b = RedisRunStore(redis_client=client, max_entries=200, ttl_seconds=3600.0)
        rb_b = RunBudget(store=store_b)
        cost, steps = rb_b.snapshot("shared-run")

        assert cost == pytest.approx(0.05)
        assert steps == 1

    def test_independent_instances_atomically_reserve_last_step(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        rb_a = RunBudget(store=RedisRunStore(redis_client=client, ttl_seconds=3600.0))
        rb_b = RunBudget(store=RedisRunStore(redis_client=client, ttl_seconds=3600.0))
        rb_a.start("shared", RunLimits(max_steps=1))

        def check(rb):
            try:
                rb.check_before_dispatch("shared")
                return True
            except RunBudgetExceeded:
                return False

        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted = list(pool.map(check, (rb_a, rb_b)))
        assert accepted.count(True) == 1

    def test_independent_instances_do_not_lose_concurrent_records(self):
        client = fakeredis.FakeRedis(decode_responses=True)
        budgets = [
            RunBudget(store=RedisRunStore(redis_client=client, ttl_seconds=3600.0))
            for _ in range(8)
        ]
        budgets[0].start("shared", RunLimits(max_steps=1000))
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(
                pool.map(
                    lambda index: budgets[index % len(budgets)].record_step(
                        "shared", "m", 0.01, 1
                    ),
                    range(100),
                )
            )
        cost, steps = budgets[0].snapshot("shared")
        assert cost == pytest.approx(1.0)
        assert steps == 100

    def test_config_flag_selects_redis_backend(self, monkeypatch):
        import router.config as config_module
        import router.run_budget as rb_module

        monkeypatch.setattr(config_module, "RUN_STORE_BACKEND", "redis")
        monkeypatch.setattr(rb_module, "RUN_STORE_BACKEND", "redis")
        monkeypatch.setattr(
            rb_module,
            "RedisRunStore",
            lambda: RedisRunStore(redis_client=fakeredis.FakeRedis(decode_responses=True)),
        )

        rb = RunBudget()
        assert isinstance(rb._store, RedisRunStore)

    def test_default_backend_is_in_memory_when_unset(self, monkeypatch):
        import router.run_budget as rb_module

        monkeypatch.setattr(rb_module, "RUN_STORE_BACKEND", "memory")
        rb = RunBudget()
        assert isinstance(rb._store, InMemoryRunStore)
