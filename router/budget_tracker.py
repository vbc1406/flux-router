"""
File: router/budget_tracker.py

Purpose:
In-memory per-user spend ledger with daily and monthly budget enforcement.
The routing engine consults BudgetTracker in Step 12 to determine whether
the chosen model would exceed the user's plan budget, and if so walks down
to a cheaper tier.  DailyBudgetTracker handles per-request daily caps
(request.max_daily_cost) set directly by callers.

Main Classes:
  BudgetTracker       — plan-level daily/monthly limits (used in routing Step 12)
  DailyBudgetTracker  — per-customer daily caps (used in routing Step 4b)

Config Dependencies (all in config.py):
  BUDGET_LIMITS           — daily/monthly limits per plan
  BUDGET_LEDGER_MAX_PER_USER — max spend records per user (bounded deque)

Key Methods:
  BudgetTracker.would_exceed_budget(user_id, cost) — call before committing to a model
  BudgetTracker.record_spend(user_id, amount, ...)  — call after a successful response
  DailyBudgetTracker.is_cap_exceeded(customer_id, daily_cap) — check daily cap

🔧 EXTENSION POINT: swap DailyBudgetTracker._ledger for a Redis client to share
  daily spend state across multiple service instances.  The public interface stays
  the same; only the storage backend changes.

Things NOT to change without discussion:
  - The lock pattern in would_exceed_budget() that reads plan, daily, and monthly
    spend under a single acquisition. This makes a single check internally
    consistent — but note that the check→record window across calls is NOT
    atomic: two concurrent requests can both pass the budget check before
    either records spend, so daily/monthly caps can be exceeded by a small
    margin under high concurrency. This is the standard pre-bill pattern;
    operators who need a hard cap should record_spend BEFORE the provider
    call and refund on failure, or wrap routing with their own mutex.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict, defaultdict, deque
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import Callable, NamedTuple

import structlog

from .config import BUDGET_LEDGER_MAX_PER_USER, BUDGET_LIMITS, BUDGET_TRACKER_MAX_USERS

log = structlog.get_logger(__name__)


# Treating empty/None user_ids as a real key would silently pool spend across
# all anonymous traffic; reject loudly instead so the caller bug is visible
# at the failure site rather than as mysterious shared budgets later.
_INVALID_USER_ID_MSG = "user_id must be a non-empty string"


def _validate_user_id(user_id: str | None) -> str:
    """Raise ValueError on empty/None/whitespace user_id. Returns the stripped id."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError(_INVALID_USER_ID_MSG)
    return user_id.strip()


class _SpendRecord(NamedTuple):
    amount: float
    model_id: str
    task_type: str
    correlation_id: str
    timestamp: datetime


def _sum_while(records: Iterable[_SpendRecord], predicate: Callable[[_SpendRecord], bool]) -> float:
    """Sum r.amount for the most-recent records matching predicate, scanning
    backward and stopping at the first non-match. Records are appended in
    non-decreasing timestamp order, so once one fails a "recent enough"
    predicate (e.g. "is today"), every earlier record fails it too — this
    turns a full O(ledger size) scan into O(matching records), which is
    typically far smaller than the per-user cap."""
    total = 0.0
    for r in reversed(records):
        if not predicate(r):
            break
        total += r.amount
    return total


class BudgetTracker:
    """
    In-memory spend ledger with daily and monthly rollup windows.

    Spend records are bucketed by (user_id, date) so daily reset is free —
    we just ignore records from prior dates when computing daily totals.
    Monthly totals look back at YYYY-MM buckets.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # user_id → bounded deque of spend records (auto-evicts oldest beyond
        # per-user cap). An OrderedDict, not a plain/defaultdict — the outer
        # dict itself is also LRU-bounded (BUDGET_TRACKER_MAX_USERS), since
        # nothing previously stopped the number of DISTINCT users tracked
        # from growing forever.
        self._ledger: OrderedDict[str, deque[_SpendRecord]] = OrderedDict()
        # user_id → plan (cached to avoid repeated lookups); evicted in
        # lockstep with _ledger so the two never disagree on which users
        # are still tracked.
        self._plans: dict[str, str] = {}

    # ── Internal: LRU-bounded ledger access ────────────────────────────────

    def _ledger_for_write(self, user_id: str) -> deque[_SpendRecord]:
        """Get-or-create user_id's ledger, evicting the least-recently-used
        user if the cap is hit. Must be called with self._lock held."""
        records = self._ledger.get(user_id)
        if records is None:
            if len(self._ledger) >= BUDGET_TRACKER_MAX_USERS:
                evicted_id, _ = self._ledger.popitem(last=False)
                self._plans.pop(evicted_id, None)
                log.info(
                    "budget_tracker_evicted_lru", limit=BUDGET_TRACKER_MAX_USERS, evicted=evicted_id
                )
            records = deque(maxlen=BUDGET_LEDGER_MAX_PER_USER)
        self._ledger[user_id] = records
        self._ledger.move_to_end(user_id)
        return records

    def _ledger_for_read(self, user_id: str) -> deque[_SpendRecord]:
        """Look up user_id's ledger WITHOUT creating one — a budget check
        for a user who has never spent anything must not itself grow the
        ledger (that was the actual unbounded-growth vector: every routing
        decision calls would_exceed_budget(), not just record_spend()).
        Must be called with self._lock held."""
        records = self._ledger.get(user_id)
        if records is not None:
            self._ledger.move_to_end(user_id)
            return records
        return deque()

    # ── Public API ──────────────────────────────────────────────────────────

    def record_spend(
        self,
        user_id: str,
        amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
        plan: str = "pro_plan",
    ) -> None:
        """Append a spend record. Also caches the user's plan for budget checks."""
        user_id = _validate_user_id(user_id)
        with self._lock:
            self._plans[user_id] = plan
            self._ledger_for_write(user_id).append(
                _SpendRecord(
                    amount=amount,
                    model_id=model_id,
                    task_type=task_type,
                    correlation_id=correlation_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )

    def get_daily_spend(self, user_id: str) -> float:
        """Sum of spend for user_id within the current UTC day."""
        user_id = _validate_user_id(user_id)
        today = datetime.now(timezone.utc).replace(tzinfo=None).date()
        with self._lock:
            records = self._ledger_for_read(user_id)
            return _sum_while(records, lambda r: r.timestamp.date() == today)

    def get_monthly_spend(self, user_id: str) -> float:
        """Sum of spend for user_id within the current UTC month."""
        user_id = _validate_user_id(user_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        with self._lock:
            records = self._ledger_for_read(user_id)
            return _sum_while(
                records, lambda r: r.timestamp.year == now.year and r.timestamp.month == now.month
            )

    def get_remaining_budget(self, user_id: str, plan: str) -> float:
        """
        Return the smaller of the remaining daily and monthly budget.
        Callers use this to know how much headroom is left before choosing a model.
        """
        user_id = _validate_user_id(user_id)
        limits = BUDGET_LIMITS.get(plan, BUDGET_LIMITS["pro_plan"])
        daily_remaining = limits["daily"] - self.get_daily_spend(user_id)
        monthly_remaining = limits["monthly"] - self.get_monthly_spend(user_id)
        return min(daily_remaining, monthly_remaining)

    def would_exceed_budget(
        self, user_id: str, estimated_cost: float, plan: str | None = None
    ) -> bool:
        """
        Return True if adding estimated_cost would push either the daily or
        monthly spend over the plan limit.

        All reads (plan, daily spend, monthly spend) happen under a single
        lock acquisition so this single check is internally consistent. The
        check→record window across calls is not atomic (see module docstring).
        """
        user_id = _validate_user_id(user_id)
        with self._lock:
            # Priority: explicitly-passed plan > previously-recorded plan > free_plan default.
            # Fail-closed (free_plan) for unknown users instead of fail-open (pro_plan).
            resolved_plan = plan or self._plans.get(user_id, "free_plan")
            limits = BUDGET_LIMITS.get(resolved_plan, BUDGET_LIMITS["free_plan"])
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            today = now.date()
            records = self._ledger_for_read(user_id)
            # Single backward pass: every record in the current month is a
            # superset of every record today, so one scan (stopping at the
            # first record outside the current month) computes both sums
            # instead of two independent O(ledger size) scans.
            daily_spend = 0.0
            monthly_spend = 0.0
            for r in reversed(records):
                if r.timestamp.year != now.year or r.timestamp.month != now.month:
                    break
                monthly_spend += r.amount
                if r.timestamp.date() == today:
                    daily_spend += r.amount
            remaining = min(
                limits["daily"] - daily_spend,
                limits["monthly"] - monthly_spend,
            )
        return estimated_cost > remaining

    def check_daily_cap(self, customer_id: str, daily_cap: float) -> bool:
        """
        Return True if customer_id has already reached or exceeded daily_cap today.
        Change 4: Used by the routing engine to enforce per-request max_daily_cost.
        """
        # get_daily_spend already validates the id.
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_savings_report(self, user_id: str) -> dict:
        """
        Compare actual spend vs what it would have cost using the most expensive
        model for every request.  Useful for ROI dashboards.
        """
        user_id = _validate_user_id(user_id)
        with self._lock:
            records = list(self._ledger_for_read(user_id))

        total_spent = sum(r.amount for r in records)

        # Aggregate by model and task_type
        by_model: dict[str, float] = defaultdict(float)
        by_task: dict[str, float] = defaultdict(float)
        for r in records:
            by_model[r.model_id] += r.amount
            by_task[r.task_type] += r.amount

        return {
            "total_spent": round(total_spent, 6),
            "breakdown_by_model": dict(by_model),
            "breakdown_by_task_type": dict(by_task),
            "record_count": len(records),
        }


# ── Change 4: DailyBudgetTracker ─────────────────────────────────────────────


class DailyBudgetTracker:
    """
    Lightweight in-memory tracker for per-customer daily spend caps.

    Change 4: Tracks actual spend per customer_id per UTC day.  The routing
    engine consults this when request.max_daily_cost is set; if the customer
    has already hit their cap the engine forces routing to the free tier
    (or returns a budget-exhausted error if the free tier is unavailable).

    This is intentionally separate from BudgetTracker (which enforces plan-level
    limits) so the two concerns stay decoupled.  Can be swapped to a persistent
    backend (Redis, DB) without touching routing logic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Same LRU-bounded-outer-dict pattern as BudgetTracker: bounds both
        # per-customer record count (BUDGET_LEDGER_MAX_PER_USER) and the
        # number of distinct customers tracked (BUDGET_TRACKER_MAX_USERS).
        self._ledger: OrderedDict[str, deque[_SpendRecord]] = OrderedDict()

    def _ledger_for_write(self, customer_id: str) -> deque[_SpendRecord]:
        """Must be called with self._lock held."""
        records = self._ledger.get(customer_id)
        if records is None:
            if len(self._ledger) >= BUDGET_TRACKER_MAX_USERS:
                evicted_id, _ = self._ledger.popitem(last=False)
                log.info(
                    "daily_budget_tracker_evicted_lru",
                    limit=BUDGET_TRACKER_MAX_USERS,
                    evicted=evicted_id,
                )
            records = deque(maxlen=BUDGET_LEDGER_MAX_PER_USER)
        self._ledger[customer_id] = records
        self._ledger.move_to_end(customer_id)
        return records

    def _ledger_for_read(self, customer_id: str) -> deque[_SpendRecord]:
        """Must be called with self._lock held. Never creates an entry — a
        cap check for a customer with no spend must not grow the ledger."""
        records = self._ledger.get(customer_id)
        if records is not None:
            self._ledger.move_to_end(customer_id)
            return records
        return deque()

    def record_spend(
        self,
        customer_id: str,
        amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
    ) -> None:
        """Record a spend event for customer_id."""
        customer_id = _validate_user_id(customer_id)
        with self._lock:
            self._ledger_for_write(customer_id).append(
                _SpendRecord(
                    amount=amount,
                    model_id=model_id,
                    task_type=task_type,
                    correlation_id=correlation_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
        log.debug(
            "daily_budget_spend_recorded",
            customer_id=customer_id,
            amount=amount,
            model_id=model_id,
        )

    def get_daily_spend(self, customer_id: str) -> float:
        """Sum of spend for customer_id within the current UTC day."""
        customer_id = _validate_user_id(customer_id)
        today = datetime.now(timezone.utc).replace(tzinfo=None).date()
        with self._lock:
            records = self._ledger_for_read(customer_id)
            return _sum_while(records, lambda r: r.timestamp.date() == today)

    def is_cap_exceeded(self, customer_id: str, daily_cap: float) -> bool:
        """
        Return True if customer_id has already spent >= daily_cap today.
        Called by the routing engine in Step 4 when request.max_daily_cost is set.
        """
        # get_daily_spend already validates the id.
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_report(self, customer_id: str) -> dict:
        """Today's spend summary for a customer."""
        customer_id = _validate_user_id(customer_id)
        today = date.today()
        def _is_today(r: _SpendRecord) -> bool:
            return r.timestamp.date() == today

        with self._lock:
            records = self._ledger_for_read(customer_id)
            # Same backward-scan-with-early-stop as _sum_while, but keeping
            # the matched records themselves (not just their sum).
            today_records = list(reversed(list(itertools.takewhile(_is_today, reversed(records)))))
        return {
            "customer_id": customer_id,
            "date": today.isoformat(),
            "total_spend": round(sum(r.amount for r in today_records), 6),
            "request_count": len(today_records),
        }
