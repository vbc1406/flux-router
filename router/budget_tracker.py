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
  BudgetTracker.reserve_spend(user_id, cost) — atomically reserve before dispatch
  BudgetTracker.record_spend(user_id, amount, ...)  — call after a successful response
  DailyBudgetTracker.is_cap_exceeded(customer_id, daily_cap) — check daily cap

🔧 EXTENSION POINT: swap DailyBudgetTracker._ledger for a Redis client to share
  daily spend state across multiple service instances.  The public interface stays
  the same; only the storage backend changes.

Compatibility note: would_exceed_budget() and record_spend() remain available,
but callers needing a hard concurrent cap must use reserve_spend() followed by
reconcile_spend() or release_reservation().
"""

from __future__ import annotations

import itertools
import json
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, NamedTuple, Reversible

import structlog

from .config import (
    BUDGET_LEDGER_MAX_PER_USER,
    BUDGET_LIMITS,
    BUDGET_RESERVATION_MAX_AGE_SECONDS,
    BUDGET_STORE_BACKEND,
    BUDGET_TRACKER_MAX_USERS,
    REDIS_URL,
)

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


def _sum_while(
    records: Reversible[_SpendRecord], predicate: Callable[[_SpendRecord], bool]
) -> float:
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
        # reservation_id -> (user_id, estimated amount, plan, created_at).
        # Reservations are included in budget checks until reconciled with
        # actual spend or explicitly released. Bounded by a staleness sweep
        # (see _sweep_stale_reservations) rather than an LRU cap: a live
        # reservation must never be evicted while its dispatch is still in
        # flight, so age — not recency of access — is the only safe signal
        # for reclaiming one that was never reconciled/released.
        self._reservations: dict[str, tuple[str, float, str, float]] = {}

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

    def _sweep_stale_reservations(self) -> None:
        """Reclaim reservations that outlived any plausible dispatch — evidence
        of a leak (a dispatch path that raised without reconciling or
        releasing) rather than normal operation. Caller holds self._lock."""
        cutoff = time.monotonic() - BUDGET_RESERVATION_MAX_AGE_SECONDS
        stale = [rid for rid, (_, _, _, created) in self._reservations.items() if created < cutoff]
        for rid in stale:
            self._reservations.pop(rid, None)
        if stale:
            log.warning(
                "budget_tracker_swept_stale_reservations",
                count=len(stale),
                max_age_seconds=BUDGET_RESERVATION_MAX_AGE_SECONDS,
            )

    def reserve_spend(
        self, user_id: str, estimated_cost: float, plan: str | None = None
    ) -> str | None:
        """Atomically reserve estimated spend, returning an opaque id.

        Returns ``None`` when the estimate would exceed either limit. The
        caller must eventually pass the id to ``reconcile_spend`` after a
        successful provider call, or to ``release_reservation`` otherwise.
        """
        user_id = _validate_user_id(user_id)
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        with self._lock:
            self._sweep_stale_reservations()
            resolved_plan = plan or self._plans.get(user_id, "free_plan")
            limits = BUDGET_LIMITS.get(resolved_plan, BUDGET_LIMITS["free_plan"])
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            daily_spend = monthly_spend = 0.0
            for record in reversed(self._ledger_for_read(user_id)):
                if record.timestamp.year != now.year or record.timestamp.month != now.month:
                    break
                monthly_spend += record.amount
                if record.timestamp.date() == now.date():
                    daily_spend += record.amount
            pending = sum(
                amount for uid, amount, _, _ in self._reservations.values() if uid == user_id
            )
            if daily_spend + pending + estimated_cost > limits["daily"] or (
                monthly_spend + pending + estimated_cost > limits["monthly"]
            ):
                return None
            reservation_id = uuid.uuid4().hex
            self._reservations[reservation_id] = (
                user_id,
                estimated_cost,
                resolved_plan,
                time.monotonic(),
            )
            return reservation_id

    def reconcile_spend(
        self,
        reservation_id: str,
        actual_amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
    ) -> None:
        """Replace a reservation with one actual spend record atomically."""
        if actual_amount < 0:
            raise ValueError("actual_amount must be non-negative")
        with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                raise KeyError(f"unknown budget reservation: {reservation_id}")
            user_id, _, plan, _ = reservation
            self._plans[user_id] = plan
            self._ledger_for_write(user_id).append(
                _SpendRecord(
                    amount=actual_amount,
                    model_id=model_id,
                    task_type=task_type,
                    correlation_id=correlation_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )

    def release_reservation(self, reservation_id: str) -> bool:
        """Release unused estimated spend. Returns whether it existed."""
        with self._lock:
            return self._reservations.pop(reservation_id, None) is not None

    def check_daily_cap(self, customer_id: str, daily_cap: float) -> bool:
        """
        Return True if customer_id has already reached or exceeded daily_cap today.
        Change 4: Used by the routing engine to enforce per-request max_daily_cost.
        """
        # get_daily_spend already validates the id.
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_savings_report(self, user_id: str) -> dict[str, Any]:
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

    def get_report(self, customer_id: str) -> dict[str, Any]:
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


# ── Redis-backed BudgetTracker (multi-worker deployments) ───────────────────


@dataclass
class _UserBudgetState:
    """Everything RedisBudgetTracker needs about one user, stored as a single
    JSON blob (mirrors run_budget._RunState's one-blob-per-run shape)."""

    ledger: list[_SpendRecord] = field(default_factory=list)
    plan: str | None = None
    # reservation_id -> (amount, plan, created_at wall-clock seconds)
    reservations: dict[str, tuple[float, str, float]] = field(default_factory=dict)


def _serialize_budget_state(state: _UserBudgetState) -> str:
    return json.dumps(
        {
            "ledger": [
                [r.amount, r.model_id, r.task_type, r.correlation_id, r.timestamp.isoformat()]
                for r in state.ledger
            ],
            "plan": state.plan,
            "reservations": state.reservations,
        }
    )


def _deserialize_budget_state(raw: str) -> _UserBudgetState:
    d = json.loads(raw)
    ledger = [
        _SpendRecord(
            amount=amount,
            model_id=model_id,
            task_type=task_type,
            correlation_id=correlation_id,
            timestamp=datetime.fromisoformat(ts),
        )
        for amount, model_id, task_type, correlation_id, ts in d["ledger"]
    ]
    reservations = {
        rid: (float(v[0]), str(v[1]), float(v[2])) for rid, v in d.get("reservations", {}).items()
    }
    return _UserBudgetState(ledger=ledger, plan=d.get("plan"), reservations=reservations)


class RedisBudgetTracker(BudgetTracker):
    """
    Redis-backed BudgetTracker for multi-worker/multi-instance deployments.

    BudgetTracker's ledger/reservations are process-local, so under N server
    workers each one enforces daily/monthly plan budgets against only the
    spend IT personally handled — silently multiplying the effective budget
    by the worker count once requests for the same user_id land on different
    workers (the normal case behind a load balancer or `FLUX_SERVER_WORKERS`
    &gt; 1). Select this with FLUX_BUDGET_STORE=redis (config.BUDGET_STORE_BACKEND)
    via make_budget_tracker() rather than constructing it directly in most
    cases.

    Subclasses BudgetTracker (rather than implementing a separate Protocol)
    purely so existing `budget: BudgetTracker` type hints (e.g.
    routing_engine._budget_tier_walkdown) keep accepting it under mypy
    --strict — every public method is overridden below; none of the parent's
    in-memory attributes (_ledger, _plans, _reservations, _lock) are used or
    initialized, and __init__ deliberately skips super().__init__().

    Storage: one JSON blob per user at `{prefix}user:{user_id}`, updated via
    an optimistic Redis WATCH/MULTI transaction — the same concurrency-safe
    technique as run_budget.RedisRunStore.update(), so concurrent workers
    reserving against the same user's budget can't both observe "ok" before
    either commits. A separate short-lived key `{prefix}resv:{reservation_id}`
    -> user_id lets reconcile_spend()/release_reservation() find which user's
    blob to touch given only a reservation_id, since that's the public
    contract BudgetTracker already has (matching it, not extending it).
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        url: str = REDIS_URL,
        key_prefix: str = "flux:budget:",
        reservation_ttl: float = BUDGET_RESERVATION_MAX_AGE_SECONDS,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "RedisBudgetTracker requires the 'redis' package. Install it with "
                    "`pip install flux-router[redis]`, or pass an explicit "
                    "redis_client= (e.g. a fakeredis instance in tests)."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._user_prefix = key_prefix + "user:"
        self._resv_prefix = key_prefix + "resv:"
        self._reservation_ttl = reservation_ttl
        # Defense-in-depth backstop key TTL, mirroring RedisRunStore — not the
        # source of truth for reservation staleness (the prune below is);
        # just a bound on truly abandoned user keys.
        self._user_key_ttl = max(int(reservation_ttl) * 4, 86400)

    # ── Redis plumbing ────────────────────────────────────────────────────

    def _get_state(self, user_id: str) -> _UserBudgetState | None:
        raw = self._redis.get(self._user_prefix + user_id)
        return _deserialize_budget_state(raw) if raw is not None else None

    def _atomic_update(
        self,
        user_id: str,
        updater: Callable[[_UserBudgetState | None], tuple[_UserBudgetState | None, Any]],
    ) -> Any:
        """Optimistic Redis transaction, safe across independent workers —
        same retry-on-WatchError shape as RedisRunStore.update()."""
        import redis as redis_lib

        key = self._user_prefix + user_id
        while True:
            with self._redis.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    state, result = updater(
                        _deserialize_budget_state(raw) if raw is not None else None
                    )
                    if state is None:
                        pipe.unwatch()
                        return result
                    pipe.multi()
                    pipe.set(key, _serialize_budget_state(state), ex=self._user_key_ttl)
                    pipe.execute()
                    return result
                except redis_lib.exceptions.WatchError:
                    continue

    def _prune_stale_reservations(self, state: _UserBudgetState) -> None:
        cutoff = time.time() - self._reservation_ttl
        stale = [rid for rid, (_, _, created) in state.reservations.items() if created < cutoff]
        for rid in stale:
            state.reservations.pop(rid, None)

    # ── Public API (same signatures as BudgetTracker) ──────────────────────

    def record_spend(
        self,
        user_id: str,
        amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
        plan: str = "pro_plan",
    ) -> None:
        user_id = _validate_user_id(user_id)

        def upd(
            state: _UserBudgetState | None,
        ) -> tuple[_UserBudgetState | None, None]:
            if state is None:
                state = _UserBudgetState()
            state.plan = plan
            state.ledger.append(
                _SpendRecord(
                    amount=amount,
                    model_id=model_id,
                    task_type=task_type,
                    correlation_id=correlation_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            if len(state.ledger) > BUDGET_LEDGER_MAX_PER_USER:
                state.ledger = state.ledger[-BUDGET_LEDGER_MAX_PER_USER:]
            return state, None

        self._atomic_update(user_id, upd)

    def get_daily_spend(self, user_id: str) -> float:
        user_id = _validate_user_id(user_id)
        today = datetime.now(timezone.utc).replace(tzinfo=None).date()
        state = self._get_state(user_id)
        records = state.ledger if state is not None else []
        return _sum_while(records, lambda r: r.timestamp.date() == today)

    def get_monthly_spend(self, user_id: str) -> float:
        user_id = _validate_user_id(user_id)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        state = self._get_state(user_id)
        records = state.ledger if state is not None else []
        return _sum_while(
            records, lambda r: r.timestamp.year == now.year and r.timestamp.month == now.month
        )

    def get_remaining_budget(self, user_id: str, plan: str) -> float:
        user_id = _validate_user_id(user_id)
        limits = BUDGET_LIMITS.get(plan, BUDGET_LIMITS["pro_plan"])
        daily_remaining = limits["daily"] - self.get_daily_spend(user_id)
        monthly_remaining = limits["monthly"] - self.get_monthly_spend(user_id)
        return min(daily_remaining, monthly_remaining)

    def would_exceed_budget(
        self, user_id: str, estimated_cost: float, plan: str | None = None
    ) -> bool:
        user_id = _validate_user_id(user_id)
        state = self._get_state(user_id)
        resolved_plan = plan or (state.plan if state is not None else None) or "free_plan"
        limits = BUDGET_LIMITS.get(resolved_plan, BUDGET_LIMITS["free_plan"])
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        today = now.date()
        records = state.ledger if state is not None else []
        daily_spend = 0.0
        monthly_spend = 0.0
        for r in reversed(records):
            if r.timestamp.year != now.year or r.timestamp.month != now.month:
                break
            monthly_spend += r.amount
            if r.timestamp.date() == today:
                daily_spend += r.amount
        remaining = min(limits["daily"] - daily_spend, limits["monthly"] - monthly_spend)
        return estimated_cost > remaining

    def reserve_spend(
        self, user_id: str, estimated_cost: float, plan: str | None = None
    ) -> str | None:
        user_id = _validate_user_id(user_id)
        if estimated_cost < 0:
            raise ValueError("estimated_cost must be non-negative")
        reservation_id = uuid.uuid4().hex

        def upd(
            state: _UserBudgetState | None,
        ) -> tuple[_UserBudgetState | None, str | None]:
            if state is None:
                state = _UserBudgetState()
            self._prune_stale_reservations(state)
            resolved_plan = plan or state.plan or "free_plan"
            limits = BUDGET_LIMITS.get(resolved_plan, BUDGET_LIMITS["free_plan"])
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            daily_spend = monthly_spend = 0.0
            for record in reversed(state.ledger):
                if record.timestamp.year != now.year or record.timestamp.month != now.month:
                    break
                monthly_spend += record.amount
                if record.timestamp.date() == now.date():
                    daily_spend += record.amount
            pending = sum(amount for amount, _, _ in state.reservations.values())
            if daily_spend + pending + estimated_cost > limits["daily"] or (
                monthly_spend + pending + estimated_cost > limits["monthly"]
            ):
                return None, None
            state.reservations[reservation_id] = (estimated_cost, resolved_plan, time.time())
            return state, reservation_id

        result = self._atomic_update(user_id, upd)
        if result is not None:
            self._redis.set(
                self._resv_prefix + reservation_id, user_id, ex=max(int(self._reservation_ttl), 1)
            )
        return result  # type: ignore[no-any-return]

    def reconcile_spend(
        self,
        reservation_id: str,
        actual_amount: float,
        model_id: str,
        correlation_id: str,
        task_type: str = "unknown",
    ) -> None:
        if actual_amount < 0:
            raise ValueError("actual_amount must be non-negative")
        user_id = self._redis.get(self._resv_prefix + reservation_id)
        if user_id is None:
            raise KeyError(f"unknown budget reservation: {reservation_id}")

        def upd(
            state: _UserBudgetState | None,
        ) -> tuple[_UserBudgetState | None, None]:
            if state is None or reservation_id not in state.reservations:
                raise KeyError(f"unknown budget reservation: {reservation_id}")
            reservation = state.reservations.pop(reservation_id)
            state.plan = reservation[1]
            state.ledger.append(
                _SpendRecord(
                    amount=actual_amount,
                    model_id=model_id,
                    task_type=task_type,
                    correlation_id=correlation_id,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                )
            )
            if len(state.ledger) > BUDGET_LEDGER_MAX_PER_USER:
                state.ledger = state.ledger[-BUDGET_LEDGER_MAX_PER_USER:]
            return state, None

        self._atomic_update(user_id, upd)
        self._redis.delete(self._resv_prefix + reservation_id)

    def release_reservation(self, reservation_id: str) -> bool:
        user_id = self._redis.get(self._resv_prefix + reservation_id)
        if user_id is None:
            return False

        def upd(
            state: _UserBudgetState | None,
        ) -> tuple[_UserBudgetState | None, bool]:
            if state is None or reservation_id not in state.reservations:
                return None, False
            state.reservations.pop(reservation_id, None)
            return state, True

        existed = self._atomic_update(user_id, upd)
        self._redis.delete(self._resv_prefix + reservation_id)
        return existed  # type: ignore[no-any-return]

    def check_daily_cap(self, customer_id: str, daily_cap: float) -> bool:
        return self.get_daily_spend(customer_id) >= daily_cap

    def get_savings_report(self, user_id: str) -> dict[str, Any]:
        user_id = _validate_user_id(user_id)
        state = self._get_state(user_id)
        records = state.ledger if state is not None else []
        total_spent = sum(r.amount for r in records)
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


def make_budget_tracker() -> BudgetTracker:
    """Construct the BudgetTracker backend selected by FLUX_BUDGET_STORE —
    RedisBudgetTracker for "redis" (multi-worker deployments), the plain
    process-local BudgetTracker otherwise. Mirrors run_budget's
    _default_store() selection logic. Used by make_flux(); tests and
    single-worker/library use should keep constructing BudgetTracker()
    directly rather than going through this."""
    if BUDGET_STORE_BACKEND == "redis":
        return RedisBudgetTracker()
    return BudgetTracker()
