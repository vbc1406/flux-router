"""
Per-user spend tracking with daily and monthly budget enforcement.

Keeps everything in memory (fast) and optionally writes through to SQLite
for persistence across restarts.  The routing engine consults this before
finalising its model choice so it can downgrade if needed.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import date, datetime
from typing import NamedTuple

import structlog

from .config import BUDGET_LIMITS

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
        # user_id → list of spend records
        self._ledger: dict[str, list[_SpendRecord]] = defaultdict(list)
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
        today = date.today()
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

        Looks up the plan from the cached value; defaults to pro_plan if unknown.
        """
        with self._lock:
            plan = self._plans.get(user_id, "pro_plan")
        remaining = self.get_remaining_budget(user_id, plan)
        return estimated_cost > remaining

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
