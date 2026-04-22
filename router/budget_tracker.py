"""
Per-user spend tracking with daily and monthly budget enforcement.

Keeps everything in memory (fast) and optionally writes through to SQLite
for persistence across restarts.  The routing engine consults this before
finalising its model choice so it can downgrade if needed.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from datetime import date, datetime
from typing import NamedTuple

import structlog

from .config import BUDGET_LEDGER_MAX_PER_USER, BUDGET_LIMITS

log = structlog.get_logger(__name__)


class _SpendRecord(NamedTuple):
    amount: float
    model_id: str
    task_type: str
    correlation_id: str
    timestamp: datetime


class BudgetTracker:
    """
    In-memory spend ledger with daily and monthly rollup windows.

    Spend records are bucketed by (user_id, date) so daily reset is free —
    we just ignore records from prior dates when computing daily totals.
    Monthly totals look back at YYYY-MM buckets.
    """

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        # user_id → bounded deque of spend records (auto-evicts oldest beyond cap)
        self._ledger: dict[str, deque[_SpendRecord]] = defaultdict(
            lambda: deque(maxlen=BUDGET_LEDGER_MAX_PER_USER)
        )
        # user_id → plan (cached to avoid repeated lookups)
        self._plans: dict[str, str] = {}

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
        with self._lock:
            self._plans[user_id] = plan
            self._ledger[user_id].append(
                _SpendRecord(
                    amount         = amount,
                    model_id       = model_id,
                    task_type      = task_type,
                    correlation_id = correlation_id,
                    timestamp      = datetime.utcnow(),
                )
            )

    def get_daily_spend(self, user_id: str) -> float:
        """Sum of spend for user_id within the current UTC day."""
        today = datetime.utcnow().date()
        with self._lock:
            return sum(
                r.amount for r in self._ledger[user_id]
                if r.timestamp.date() == today
            )

    def get_monthly_spend(self, user_id: str) -> float:
        """Sum of spend for user_id within the current UTC month."""
        now = datetime.utcnow()
        with self._lock:
            return sum(
                r.amount for r in self._ledger[user_id]
                if r.timestamp.year == now.year and r.timestamp.month == now.month
            )

    def get_remaining_budget(self, user_id: str, plan: str) -> float:
        """
        Return the smaller of the remaining daily and monthly budget.
        Callers use this to know how much headroom is left before choosing a model.
        """
        limits = BUDGET_LIMITS.get(plan, BUDGET_LIMITS["pro_plan"])
        daily_remaining   = limits["daily"]   - self.get_daily_spend(user_id)
        monthly_remaining = limits["monthly"] - self.get_monthly_spend(user_id)
        return min(daily_remaining, monthly_remaining)

    def would_exceed_budget(self, user_id: str, estimated_cost: float) -> bool:
        """
        Return True if adding estimated_cost would push either the daily or
        monthly spend over the plan limit.

        All reads (plan, daily spend, monthly spend) happen under a single lock
        acquisition to avoid TOCTOU races under concurrent routing.
        """
        with self._lock:
            plan   = self._plans.get(user_id, "pro_plan")
            limits = BUDGET_LIMITS.get(plan, BUDGET_LIMITS["pro_plan"])
            now    = datetime.utcnow()
            today  = now.date()
            records = self._ledger[user_id]
            daily_spend = sum(
                r.amount for r in records if r.timestamp.date() == today
            )
            monthly_spend = sum(
                r.amount for r in records
                if r.timestamp.year == now.year and r.timestamp.month == now.month
            )
            remaining = min(
                limits["daily"]   - daily_spend,
                limits["monthly"] - monthly_spend,
            )
        return estimated_cost > remaining

    def check_daily_cap(self, customer_id: str, daily_cap: float) -> bool:
        """
        Return True if customer_id has already reached or exceeded daily_cap today.
        Change 4: Used by the routing engine to enforce per-request max_daily_cost.
        """
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_savings_report(self, user_id: str) -> dict:
        """
        Compare actual spend vs what it would have cost using the most expensive
        model for every request.  Useful for ROI dashboards.
        """
        with self._lock:
            records = list(self._ledger[user_id])

        total_spent = sum(r.amount for r in records)

        # Aggregate by model and task_type
        by_model: dict[str, float]     = defaultdict(float)
        by_task:  dict[str, float]     = defaultdict(float)
        for r in records:
            by_model[r.model_id] += r.amount
            by_task[r.task_type] += r.amount

        return {
            "total_spent":             round(total_spent, 6),
            "breakdown_by_model":      dict(by_model),
            "breakdown_by_task_type":  dict(by_task),
            "record_count":            len(records),
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
        self._lock  = threading.Lock()
        self._ledger: dict[str, deque[_SpendRecord]] = defaultdict(
            lambda: deque(maxlen=BUDGET_LEDGER_MAX_PER_USER)
        )

    def record_spend(
        self,
        customer_id: str,
        amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
    ) -> None:
        """Record a spend event for customer_id."""
        with self._lock:
            self._ledger[customer_id].append(
                _SpendRecord(
                    amount         = amount,
                    model_id       = model_id,
                    task_type      = task_type,
                    correlation_id = correlation_id,
                    timestamp      = datetime.utcnow(),
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
        today = datetime.utcnow().date()
        with self._lock:
            return sum(
                r.amount for r in self._ledger[customer_id]
                if r.timestamp.date() == today
            )

    def is_cap_exceeded(self, customer_id: str, daily_cap: float) -> bool:
        """
        Return True if customer_id has already spent >= daily_cap today.
        Called by the routing engine in Step 4 when request.max_daily_cost is set.
        """
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_report(self, customer_id: str) -> dict:
        """Today's spend summary for a customer."""
        today = date.today()
        with self._lock:
            today_records = [
                r for r in self._ledger[customer_id]
                if r.timestamp.date() == today
            ]
        return {
            "customer_id":  customer_id,
            "date":         today.isoformat(),
            "total_spend":  round(sum(r.amount for r in today_records), 6),
            "request_count": len(today_records),
        }
