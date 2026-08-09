"""
File: router/tests/test_budget_tracker.py

Purpose:
Unit tests for router/budget_tracker.py — BudgetTracker and
DailyBudgetTracker. Focused on two fixes: the ledger's outer dict (keyed by
user_id/customer_id) is now LRU-bounded (BUDGET_TRACKER_MAX_USERS), and
read-only budget checks (would_exceed_budget, get_daily_spend, ...) no
longer create a ledger entry for a user who has never spent anything —
previously every check (not just every record_spend) grew the ledger.

How to run:
  pytest -v router/tests/test_budget_tracker.py
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from router.budget_tracker import BudgetTracker, DailyBudgetTracker


class TestBudgetReservations:
    def test_concurrent_reservations_cannot_overspend(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setitem(bt.BUDGET_LIMITS, "test_plan", {"daily": 1.0, "monthly": 1.0})
        tracker = BudgetTracker()
        with ThreadPoolExecutor(max_workers=20) as pool:
            reservations = list(
                pool.map(lambda _: tracker.reserve_spend("user", 0.1, "test_plan"), range(20))
            )
        accepted = [reservation for reservation in reservations if reservation is not None]
        assert len(accepted) == 10

        tracker.reconcile_spend(accepted[0], 0.08, "model", "request")
        assert tracker.get_daily_spend("user") == 0.08
        assert tracker.release_reservation(accepted[1]) is True
        assert tracker.reserve_spend("user", 0.1, "test_plan") is not None


class TestBudgetTrackerReadDoesNotGrowLedger:
    def test_would_exceed_budget_does_not_create_entry_for_unknown_user(self):
        tracker = BudgetTracker()
        tracker.would_exceed_budget("brand-new-user", 0.01, "free_plan")
        assert "brand-new-user" not in tracker._ledger

    def test_get_daily_spend_does_not_create_entry_for_unknown_user(self):
        tracker = BudgetTracker()
        assert tracker.get_daily_spend("never-spent") == 0.0
        assert "never-spent" not in tracker._ledger

    def test_get_monthly_spend_does_not_create_entry_for_unknown_user(self):
        tracker = BudgetTracker()
        assert tracker.get_monthly_spend("never-spent") == 0.0
        assert "never-spent" not in tracker._ledger

    def test_record_spend_still_creates_entry(self):
        tracker = BudgetTracker()
        tracker.record_spend("u1", 1.0, "m1", "cid-1")
        assert "u1" in tracker._ledger


class TestBudgetTrackerLRUBound:
    def test_distinct_users_bounded_by_max_users(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setattr(bt, "BUDGET_TRACKER_MAX_USERS", 3)
        tracker = BudgetTracker()
        for i in range(10):
            tracker.record_spend(f"user-{i}", 0.01, "m1", f"cid-{i}")
        assert len(tracker._ledger) == 3
        assert len(tracker._plans) == 3
        # Only the most recently touched users survive.
        assert set(tracker._ledger.keys()) == {"user-7", "user-8", "user-9"}

    def test_evicted_user_starts_fresh_not_erroring(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setattr(bt, "BUDGET_TRACKER_MAX_USERS", 2)
        tracker = BudgetTracker()
        tracker.record_spend("u1", 5.0, "m1", "cid-1", plan="business_plan")
        tracker.record_spend("u2", 5.0, "m1", "cid-2", plan="business_plan")
        tracker.record_spend("u3", 5.0, "m1", "cid-3", plan="business_plan")  # evicts u1

        assert "u1" not in tracker._ledger
        # u1's history is gone, but querying it must not raise — it's just
        # treated as a fresh (never-spent) user again.
        assert tracker.get_daily_spend("u1") == 0.0

    def test_touching_a_user_keeps_them_from_eviction(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setattr(bt, "BUDGET_TRACKER_MAX_USERS", 2)
        tracker = BudgetTracker()
        tracker.record_spend("u1", 1.0, "m1", "cid-1")
        tracker.record_spend("u2", 1.0, "m1", "cid-2")
        tracker.get_daily_spend("u1")  # touch u1 -> most-recently-used
        tracker.record_spend("u3", 1.0, "m1", "cid-3")  # should evict u2, not u1

        assert "u1" in tracker._ledger
        assert "u2" not in tracker._ledger


class TestBudgetTrackerCorrectnessPreservedByBackwardScan:
    """The backward-scan-with-early-stop optimization must produce identical
    results to a full scan — these pin down that behavior with a mixed-date
    ledger."""

    def test_daily_and_monthly_sums_ignore_older_records(self, monkeypatch):
        from datetime import datetime, timedelta, timezone

        import router.budget_tracker as bt

        tracker = BudgetTracker()
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Old records (different day, different month) must not count.
        old = now - timedelta(days=40)
        tracker._ledger_for_write("u1")
        with tracker._lock:
            tracker._ledger["u1"].append(
                bt._SpendRecord(2.0, "m1", "x", "cid-old", old)
            )
            tracker._ledger["u1"].append(
                bt._SpendRecord(3.0, "m1", "x", "cid-today", now)
            )

        assert tracker.get_daily_spend("u1") == 3.0
        assert tracker.get_monthly_spend("u1") == 3.0


class TestDailyBudgetTrackerReadDoesNotGrowLedger:
    def test_get_daily_spend_does_not_create_entry(self):
        tracker = DailyBudgetTracker()
        assert tracker.get_daily_spend("never-spent") == 0.0
        assert "never-spent" not in tracker._ledger

    def test_get_report_does_not_create_entry(self):
        tracker = DailyBudgetTracker()
        report = tracker.get_report("never-spent")
        assert report["total_spend"] == 0.0
        assert report["request_count"] == 0
        assert "never-spent" not in tracker._ledger


class TestDailyBudgetTrackerLRUBound:
    def test_distinct_customers_bounded_by_max_users(self, monkeypatch):
        import router.budget_tracker as bt

        monkeypatch.setattr(bt, "BUDGET_TRACKER_MAX_USERS", 2)
        tracker = DailyBudgetTracker()
        tracker.record_spend("c1", 1.0, "m1", "cid-1")
        tracker.record_spend("c2", 1.0, "m1", "cid-2")
        tracker.record_spend("c3", 1.0, "m1", "cid-3")

        assert len(tracker._ledger) == 2
        assert "c1" not in tracker._ledger
