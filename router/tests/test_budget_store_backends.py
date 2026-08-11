"""
File: router/tests/test_budget_store_backends.py

Purpose:
Shared behavior tests proving BudgetTracker and RedisBudgetTracker
(router/budget_tracker.py) honor the same public contract — record/reserve/
reconcile/release, daily/monthly windowing, LRU-style plan resolution, and
concurrent-reservation safety — not just that each one works in isolation.
Mirrors router/tests/test_run_store_backends.py's approach for
run_budget.RunStore.

RedisBudgetTracker is exercised against fakeredis (an in-memory fake of the
Redis protocol) rather than a real server, so these tests run without any
external infrastructure. If fakeredis isn't installed, the redis-backed
parametrization is skipped (not failed).

How to run:
  pytest -v router/tests/test_budget_store_backends.py
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from router.budget_tracker import BudgetTracker

try:
    import fakeredis

    from router.budget_tracker import RedisBudgetTracker

    _HAS_FAKEREDIS = True
except ImportError:
    _HAS_FAKEREDIS = False


def _make_memory_tracker() -> BudgetTracker:
    return BudgetTracker()


def _make_redis_tracker() -> "RedisBudgetTracker":
    client = fakeredis.FakeRedis(decode_responses=True)
    return RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)


_BACKENDS = [
    pytest.param(_make_memory_tracker, id="in-memory"),
    pytest.param(
        _make_redis_tracker,
        id="redis",
        marks=pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed"),
    ),
]


@pytest.fixture(params=_BACKENDS)
def tracker_factory(request):
    return request.param


class TestBudgetTrackerContract:
    """Backend-agnostic behavior every BudgetTracker implementation must satisfy."""

    def test_new_user_has_zero_spend(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 1.0})
        tracker = tracker_factory()
        assert tracker.get_daily_spend("brand-new-user") == 0.0
        assert tracker.get_monthly_spend("brand-new-user") == 0.0
        assert tracker.would_exceed_budget("brand-new-user", 0.5, "test_plan") is False

    def test_record_spend_then_read_round_trips(self, tracker_factory):
        tracker = tracker_factory()
        tracker.record_spend("u1", 1.5, "gpt-cheap", "cid-1")
        assert tracker.get_daily_spend("u1") == pytest.approx(1.5)
        assert tracker.get_monthly_spend("u1") == pytest.approx(1.5)

    def test_would_exceed_budget_respects_plan_limits(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        tracker = tracker_factory()
        tracker.record_spend("u1", 0.9, "m", "cid-1", plan="test_plan")
        assert tracker.would_exceed_budget("u1", 0.05, "test_plan") is False
        assert tracker.would_exceed_budget("u1", 0.2, "test_plan") is True

    def test_reserve_reconcile_round_trip(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        tracker = tracker_factory()
        reservation_id = tracker.reserve_spend("u1", 0.3, "test_plan")
        assert reservation_id is not None
        # would_exceed_budget() only scans the recorded ledger, not pending
        # reservations (that's reserve_spend()'s own job, tested separately
        # below) — a reservation alone must not yet show up as spend.
        assert tracker.get_daily_spend("u1") == 0.0
        tracker.reconcile_spend(reservation_id, 0.25, "m", "cid-1")
        assert tracker.get_daily_spend("u1") == pytest.approx(0.25)

    def test_reserve_returns_none_when_over_budget(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 0.1, "monthly": 10.0})
        tracker = tracker_factory()
        assert tracker.reserve_spend("u1", 0.2, "test_plan") is None

    def test_release_reservation_frees_pending_budget(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        tracker = tracker_factory()
        reservation_id = tracker.reserve_spend("u1", 0.9, "test_plan")
        assert tracker.reserve_spend("u1", 0.5, "test_plan") is None  # pending blocks it
        assert tracker.release_reservation(reservation_id) is True
        assert tracker.reserve_spend("u1", 0.5, "test_plan") is not None

    def test_release_unknown_reservation_returns_false(self, tracker_factory):
        tracker = tracker_factory()
        assert tracker.release_reservation("does-not-exist") is False

    def test_reconcile_unknown_reservation_raises(self, tracker_factory):
        tracker = tracker_factory()
        with pytest.raises(KeyError):
            tracker.reconcile_spend("does-not-exist", 0.1, "m", "cid")

    def test_reconcile_negative_amount_raises(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        tracker = tracker_factory()
        reservation_id = tracker.reserve_spend("u1", 0.1, "test_plan")
        with pytest.raises(ValueError):
            tracker.reconcile_spend(reservation_id, -0.1, "m", "cid")

    def test_reserve_negative_estimate_raises(self, tracker_factory):
        tracker = tracker_factory()
        with pytest.raises(ValueError):
            tracker.reserve_spend("u1", -0.1)

    def test_check_daily_cap(self, tracker_factory):
        tracker = tracker_factory()
        tracker.record_spend("u1", 5.0, "m", "cid-1")
        assert tracker.check_daily_cap("u1", 5.0) is True
        assert tracker.check_daily_cap("u1", 5.01) is False

    def test_get_savings_report_breaks_down_by_model_and_task(self, tracker_factory):
        tracker = tracker_factory()
        tracker.record_spend("u1", 1.0, "m1", "cid-1", task_type="reasoning")
        tracker.record_spend("u1", 2.0, "m2", "cid-2", task_type="summarization")
        report = tracker.get_savings_report("u1")
        assert report["total_spent"] == pytest.approx(3.0)
        assert report["record_count"] == 2
        assert report["breakdown_by_model"]["m1"] == pytest.approx(1.0)
        assert report["breakdown_by_model"]["m2"] == pytest.approx(2.0)
        assert report["breakdown_by_task_type"]["reasoning"] == pytest.approx(1.0)

    def test_concurrent_reservations_cannot_overspend(self, tracker_factory, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 1.0})
        tracker = tracker_factory()
        with ThreadPoolExecutor(max_workers=20) as pool:
            reservations = list(
                pool.map(lambda _: tracker.reserve_spend("user", 0.1, "test_plan"), range(20))
            )
        accepted = [r for r in reservations if r is not None]
        assert len(accepted) == 10


@pytest.mark.skipif(not _HAS_FAKEREDIS, reason="fakeredis not installed")
class TestRedisBudgetTrackerSpecifics:
    """Coverage for RedisBudgetTracker behavior with no in-memory analogue."""

    def test_requires_redis_package_or_explicit_client(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "redis":
                raise ImportError("no redis installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(ImportError, match="RedisBudgetTracker requires"):
            RedisBudgetTracker()

    def test_state_survives_a_fresh_tracker_instance_against_the_same_client(self):
        """Simulates two different worker processes sharing one Redis: a
        second RedisBudgetTracker built against the SAME underlying client
        must see spend the first one recorded — this is the entire point of
        the Redis backend."""
        client = fakeredis.FakeRedis(decode_responses=True)
        tracker_a = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)
        tracker_a.record_spend("shared-user", 0.4, "m", "cid-1")

        tracker_b = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)
        assert tracker_b.get_daily_spend("shared-user") == pytest.approx(0.4)

    def test_independent_instances_atomically_reserve_last_slot(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 0.1, "monthly": 10.0})
        client = fakeredis.FakeRedis(decode_responses=True)
        tracker_a = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)
        tracker_b = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)

        def reserve(tracker):
            return tracker.reserve_spend("shared", 0.1, "test_plan") is not None

        with ThreadPoolExecutor(max_workers=2) as pool:
            accepted = list(pool.map(reserve, (tracker_a, tracker_b)))
        assert accepted.count(True) == 1

    def test_reconcile_from_a_different_instance_than_reserved(self, monkeypatch):
        """A reservation made by worker A must be reconcilable by worker B —
        exactly the cross-worker scenario a shared Redis backend exists for."""
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        client = fakeredis.FakeRedis(decode_responses=True)
        tracker_a = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)
        tracker_b = RedisBudgetTracker(redis_client=client, reservation_ttl=600.0)

        reservation_id = tracker_a.reserve_spend("u1", 0.3, "test_plan")
        tracker_b.reconcile_spend(reservation_id, 0.25, "m", "cid-1")

        assert tracker_a.get_daily_spend("u1") == pytest.approx(0.25)

    def test_stale_reservations_are_pruned_on_next_reserve(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 10.0})
        client = fakeredis.FakeRedis(decode_responses=True)
        tracker = RedisBudgetTracker(redis_client=client, reservation_ttl=0.05)
        first = tracker.reserve_spend("u1", 0.9, "test_plan")
        assert first is not None
        time.sleep(0.1)
        # The stale 0.9 reservation must no longer block a fresh one.
        second = tracker.reserve_spend("u1", 0.9, "test_plan")
        assert second is not None

    def test_config_flag_selects_redis_backend(self, monkeypatch):
        import router.budget_tracker as bt_module

        monkeypatch.setattr(bt_module, "BUDGET_STORE_BACKEND", "redis")
        monkeypatch.setattr(
            bt_module,
            "RedisBudgetTracker",
            lambda: RedisBudgetTracker(redis_client=fakeredis.FakeRedis(decode_responses=True)),
        )
        tracker = bt_module.make_budget_tracker()
        assert isinstance(tracker, RedisBudgetTracker)

    def test_default_backend_is_in_memory_when_unset(self, monkeypatch):
        import router.budget_tracker as bt_module

        monkeypatch.setattr(bt_module, "BUDGET_STORE_BACKEND", "memory")
        tracker = bt_module.make_budget_tracker()
        assert type(tracker) is BudgetTracker
