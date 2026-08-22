"""
File: router/run_budget.py

Purpose:
Run-scoped budget enforcement (the agent cost governance wedge).
A "run" is a correlation ID (run_id) grouping N routing decisions — a
multi-step agent trajectory, not a single request. Flux's existing per-user
and per-day budget trackers (budget_tracker.py) cannot stop a runaway agent
loop mid-trajectory; this module can, because it is checked BEFORE each
step's dispatch rather than after the fact.

Main Classes:
  RunBudget          — public API: check_before_dispatch(), record_step()
  RunBudgetExceeded  — raised when a run has exhausted any of its limits
  InMemoryRunStore    — default storage; LRU + TTL evicted (see RunStore)

Config Dependencies (all in config.py):
  RUN_MAX_COST_USD, RUN_MAX_STEPS, RUN_MAX_TOKENS, RUN_MAX_DURATION_SECONDS
  RUN_DEGRADE_THRESHOLD, RUN_WARN_THRESHOLD
  RUN_STORE_MAX_ENTRIES, RUN_TTL_SECONDS, RUN_HOUSEKEEPING_INTERVAL

Enforcement model:
  check_before_dispatch(run_id) is called BEFORE a step spends anything. It
  looks only at *prior, already-recorded* steps for this run:
    - If any limit is already met/exceeded, raises RunBudgetExceeded — the
      caller never dispatches this step, so the run is stopped before it
      spends, never after.
    - Otherwise returns "degraded" or "warning" once the ladder thresholds
      are crossed, so routing_engine.py can force cost-optimized routing
      for this step (and the caller can inspect budget_warning to decide
      whether to keep going).
  record_step(run_id, ...) is called AFTER a step's dispatch succeeds, and
  updates the run's cumulative cost/tokens/step-count for the *next* call to
  check_before_dispatch(). It never raises — the ladder is entirely a
  before-dispatch concern.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Protocol, TypeVar, cast

import structlog

from .config import (
    REDIS_URL,
    RUN_DEGRADE_THRESHOLD,
    RUN_HOUSEKEEPING_INTERVAL,
    RUN_MAX_COST_USD,
    RUN_MAX_DURATION_SECONDS,
    RUN_MAX_STEPS,
    RUN_MAX_TOKENS,
    RUN_STORE_BACKEND,
    RUN_STORE_MAX_ENTRIES,
    RUN_TTL_SECONDS,
    RUN_WARN_THRESHOLD,
)

log = structlog.get_logger(__name__)
_T = TypeVar("_T")


@dataclass
class StepRecord:
    """One completed step within a run, for the RunBudgetExceeded breakdown."""

    model_id: str
    cost_usd: float
    tokens: int
    timestamp: float


@dataclass
class RunLimits:
    """Per-run ceilings. Falls back to the global config defaults when unset."""

    max_cost_usd: float = RUN_MAX_COST_USD
    max_steps: int = RUN_MAX_STEPS
    max_tokens: int = RUN_MAX_TOKENS
    max_duration_seconds: float = RUN_MAX_DURATION_SECONDS


@dataclass
class _RunState:
    limits: RunLimits
    started_at: float = field(default_factory=time.monotonic)
    last_touched: float = field(default_factory=time.monotonic)
    steps: list[StepRecord] = field(default_factory=list)
    cost_so_far: float = 0.0
    tokens_so_far: int = 0
    # Steps that passed check_before_dispatch but haven't been recorded (or
    # released) yet — see RunBudget._key's sibling concern in
    # check_before_dispatch()/record_step()/release_reservation() below.
    # Counted toward max_steps so N concurrent requests against the same run
    # can't all pass the check before any of them records a step.
    pending_steps: int = 0


class RunBudgetExceeded(Exception):  # noqa: N818 — name matches the public API exactly
    """
    Raised by check_before_dispatch() when a run has already met or exceeded
    any of its limits. Carries a structured summary so the caller can return
    partial results instead of nothing.
    """

    def __init__(self, run_id: str, summary: dict[str, Any]) -> None:
        self.run_id = run_id
        self.summary = summary
        super().__init__(
            f"Run '{run_id}' exceeded its budget after {summary['steps_taken']} step(s): "
            f"${summary['total_cost_usd']:.4f} spent, "
            f"reason={summary['exceeded_reason']}"
        )


# 🔧 EXTENSION POINT: implement this Protocol against Redis (or any shared
# store) for multi-process deployments, where run state must be visible
# across workers. The in-memory default below is process-local only.
class RunStore(Protocol):
    def get(self, run_id: str) -> _RunState | None: ...
    def set(self, run_id: str, state: _RunState) -> None: ...
    def delete(self, run_id: str) -> None: ...
    def update(
        self, run_id: str, updater: Callable[[_RunState | None], tuple[_RunState | None, _T]]
    ) -> _T:
        """Atomically read, transform, and persist a run state."""
        ...
    def sweep_expired(self, ttl_seconds: float) -> int:
        """Evict entries idle longer than ttl_seconds. Returns count evicted."""
        ...


class InMemoryRunStore:
    """
    Thread-safe, process-local RunStore. Bounded by RUN_STORE_MAX_ENTRIES
    (LRU eviction of the least-recently-touched run) and by RUN_TTL_SECONDS
    (idle runs are swept away even under the cap), so an agent fleet that
    starts runs and never explicitly closes them cannot leak memory.
    """

    def __init__(self, max_entries: int = RUN_STORE_MAX_ENTRIES) -> None:
        self._states: OrderedDict[str, _RunState] = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get(self, run_id: str) -> _RunState | None:
        with self._lock:
            state = self._states.get(run_id)
            if state is not None:
                self._states.move_to_end(run_id)
            return state

    def set(self, run_id: str, state: _RunState) -> None:
        with self._lock:
            if run_id not in self._states and len(self._states) >= self._max_entries:
                evicted_id, _ = self._states.popitem(last=False)
                log.info("run_budget_evicted_lru", limit=self._max_entries, evicted=evicted_id)
            self._states[run_id] = state
            self._states.move_to_end(run_id)

    def delete(self, run_id: str) -> None:
        with self._lock:
            self._states.pop(run_id, None)

    def update(
        self, run_id: str, updater: Callable[[_RunState | None], tuple[_RunState | None, _T]]
    ) -> _T:
        with self._lock:
            state, result = updater(self._states.get(run_id))
            if state is None:
                return result
            if run_id not in self._states and len(self._states) >= self._max_entries:
                self._states.popitem(last=False)
            self._states[run_id] = state
            self._states.move_to_end(run_id)
            return result

    def sweep_expired(self, ttl_seconds: float) -> int:
        now = time.monotonic()
        with self._lock:
            expired = [
                run_id
                for run_id, state in self._states.items()
                if now - state.last_touched > ttl_seconds
            ]
            for run_id in expired:
                del self._states[run_id]
        if expired:
            log.info("run_budget_evicted_ttl", count=len(expired), ttl_seconds=ttl_seconds)
        return len(expired)


class RedisRunStore:
    """
    Redis-backed RunStore for multi-worker/multi-instance deployments, where
    InMemoryRunStore's process-local state lets each worker enforce a run's
    budget against only the steps IT personally handled — silently
    under-counting the run's true cumulative spend once requests for the
    same run_id land on different workers (the normal case behind a load
    balancer or multiple uvicorn workers). Select it with FLUX_RUN_STORE=redis
    (config.RUN_STORE_BACKEND) rather than constructing it directly in most
    cases — see RunBudget's default-store wiring below.

    Same eviction semantics as InMemoryRunStore:
      - LRU cap at `max_entries`: a Redis sorted set (`{key_prefix}lru`,
        member=run_id, score=last-touched wall-clock time) tracks recency;
        inserting a genuinely NEW run_id while at capacity evicts the
        lowest-scored (least-recently-touched) member first — mirrors
        InMemoryRunStore.set()'s OrderedDict.popitem(last=False).
      - `sweep_expired(ttl_seconds)` evicts anything idle longer than
        ttl_seconds, same contract as InMemoryRunStore, implemented via a
        ZRANGEBYSCORE cutoff query instead of a full scan.
      - A native Redis key TTL (`ttl_seconds` at construction) is set on each
        state entry purely as a defense-in-depth backstop for a worker that
        crashes before the next housekeeping sweep runs; the manual sweep
        above is the actual source of truth for the Protocol's eviction count.

    Concurrency note: the "is this run_id new" check and the LRU-cap eviction
    are two separate round trips, not one atomic transaction — under heavy
    concurrent writers from many workers, max_entries is a soft/approximate
    cap (as it effectively already is for InMemoryRunStore across a fleet,
    which only bounds ONE process's local state). Not pursuing a Lua-script
    atomic version here; correctness of budget enforcement itself does not
    depend on this bound being exact.

    Clock-domain note: `_RunState.started_at` / `.last_touched` are
    `time.monotonic()` values, which are only meaningful WITHIN the process
    that produced them — comparing one process's monotonic clock reading to
    another's is meaningless (each process's monotonic epoch is arbitrary).
    Values are translated to wall-clock time before being written to Redis,
    and translated back to an equivalent-for-THIS-process monotonic value on
    read, so cross-process elapsed-time math (duration limits, TTL sweep)
    stays correct within normal clock-computation tolerances (sub-millisecond
    in practice) rather than being silently nonsensical.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        max_entries: int = RUN_STORE_MAX_ENTRIES,
        ttl_seconds: float = RUN_TTL_SECONDS,
        key_prefix: str = "flux:run_budget:",
        url: str = REDIS_URL,
    ) -> None:
        if redis_client is not None:
            self._redis = redis_client
        else:
            try:
                import redis
            except ImportError as exc:
                raise ImportError(
                    "RedisRunStore requires the 'redis' package. Install it with "
                    "`pip install -e '.[redis]'`, or pass an explicit "
                    "redis_client= (e.g. a fakeredis instance in tests)."
                ) from exc
            self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._state_prefix = key_prefix + "state:"
        self._lru_key = key_prefix + "lru"

    # ── Clock translation (see class docstring) ─────────────────────────────

    @staticmethod
    def _mono_to_wall(mono_value: float) -> float:
        return mono_value + (time.time() - time.monotonic())

    @staticmethod
    def _wall_to_mono(wall_value: float) -> float:
        return wall_value - (time.time() - time.monotonic())

    def _serialize(self, state: _RunState) -> str:
        d = asdict(state)
        d["started_at"] = self._mono_to_wall(d["started_at"])
        d["last_touched"] = self._mono_to_wall(d["last_touched"])
        return json.dumps(d)

    def _deserialize(self, raw: str) -> _RunState:
        d = json.loads(raw)
        return _RunState(
            limits=RunLimits(**d["limits"]),
            started_at=self._wall_to_mono(d["started_at"]),
            last_touched=self._wall_to_mono(d["last_touched"]),
            steps=[StepRecord(**s) for s in d["steps"]],
            cost_so_far=d["cost_so_far"],
            tokens_so_far=d["tokens_so_far"],
            pending_steps=d.get("pending_steps", 0),
        )

    # ── RunStore Protocol ────────────────────────────────────────────────────

    def get(self, run_id: str) -> _RunState | None:
        raw = self._redis.get(self._state_prefix + run_id)
        if raw is None:
            return None
        # Recency bump for LRU purposes only, mirroring InMemoryRunStore.get()'s
        # move_to_end() — last_touched itself is only ever refreshed by set().
        self._redis.zadd(self._lru_key, {run_id: time.time()})
        return self._deserialize(raw)

    def set(self, run_id: str, state: _RunState) -> None:
        is_new = self._redis.zscore(self._lru_key, run_id) is None
        if is_new and self._redis.zcard(self._lru_key) >= self._max_entries:
            evicted = self._redis.zpopmin(self._lru_key)
            if evicted:
                evicted_id = evicted[0][0]
                self._redis.delete(self._state_prefix + evicted_id)
                log.info("run_budget_evicted_lru", limit=self._max_entries, evicted=evicted_id)
        self._redis.set(
            self._state_prefix + run_id,
            self._serialize(state),
            ex=max(int(self._ttl_seconds), 1),
        )
        self._redis.zadd(self._lru_key, {run_id: time.time()})

    def delete(self, run_id: str) -> None:
        self._redis.delete(self._state_prefix + run_id)
        self._redis.zrem(self._lru_key, run_id)

    def update(
        self, run_id: str, updater: Callable[[_RunState | None], tuple[_RunState | None, _T]]
    ) -> _T:
        """Optimistic Redis transaction, safe across independent workers."""
        import redis

        state_key = self._state_prefix + run_id
        while True:
            with self._redis.pipeline() as pipe:
                try:
                    pipe.watch(state_key)
                    raw = pipe.get(state_key)
                    state, result = updater(self._deserialize(raw) if raw is not None else None)
                    if state is None:
                        pipe.unwatch()
                        return result
                    pipe.multi()
                    pipe.set(state_key, self._serialize(state), ex=max(int(self._ttl_seconds), 1))
                    pipe.zadd(self._lru_key, {run_id: time.time()})
                    pipe.execute()
                    if self._redis.zcard(self._lru_key) > self._max_entries:
                        evicted = self._redis.zpopmin(self._lru_key)
                        if evicted:
                            self._redis.delete(self._state_prefix + evicted[0][0])
                    return result
                except redis.exceptions.WatchError:
                    continue

    def sweep_expired(self, ttl_seconds: float) -> int:
        cutoff = time.time() - ttl_seconds
        expired = self._redis.zrangebyscore(self._lru_key, "-inf", cutoff)
        if not expired:
            return 0
        for run_id in expired:
            self._redis.delete(self._state_prefix + run_id)
        self._redis.zrem(self._lru_key, *expired)
        log.info("run_budget_evicted_ttl", count=len(expired), ttl_seconds=ttl_seconds)
        return len(expired)


def _default_store() -> RunStore:
    """Bugfix: RunBudget() used to always hardcode InMemoryRunStore, which is
    unsafe for multi-worker deployments (see RedisRunStore's docstring) with
    no way to opt out short of manually constructing RunBudget(store=...).
    FLUX_RUN_STORE=redis now selects RedisRunStore here instead."""
    if RUN_STORE_BACKEND == "redis":
        return RedisRunStore()
    return InMemoryRunStore()


class RunBudget:
    """Public API for run-scoped budget enforcement. See module docstring."""

    def __init__(self, store: RunStore | None = None) -> None:
        self._store: RunStore = store or _default_store()
        self._lock = threading.Lock()
        self._call_count = 0

    @staticmethod
    def _key(run_id: str, tenant_id: str | None) -> str:
        """Storage key for a run. run_id is client-controlled (X-Flux-Run-Id)
        and is NOT namespaced by itself — two different tenants (or two
        callers under no-tenant auth) supplying the same run_id would
        otherwise share one run's budget state, letting one exhaust or
        inspect another's run accounting. When tenant_id is known (verified
        server-side in FLUX_SERVER_TOKENS mode — see server.py), the storage
        key is scoped by it. tenant_id is never itself attacker-controlled in
        that mode, so this closes the collision. Falls back to the raw run_id
        when tenant_id is unset, unchanged from prior behavior."""
        return f"{tenant_id}\x00{run_id}" if tenant_id else run_id

    def start(
        self, run_id: str, limits: RunLimits | None = None, tenant_id: str | None = None
    ) -> None:
        """Explicitly (re)initialize a run with its own limits. Optional —
        check_before_dispatch() lazily creates a run with global defaults if
        start() was never called."""
        self._store.set(self._key(run_id, tenant_id), _RunState(limits=limits or RunLimits()))

    def finish(self, run_id: str, tenant_id: str | None = None) -> None:
        """Drop a run's state immediately rather than waiting for TTL eviction."""
        self._store.delete(self._key(run_id, tenant_id))

    def check_before_dispatch(
        self, run_id: str, tenant_id: str | None = None
    ) -> Literal["ok", "degraded", "warning"]:
        """
        Evaluate this run's state from steps recorded (and reserved — see
        below) so far, and return the budget_state for the step about to be
        dispatched: "ok", "degraded", or "warning". Raises RunBudgetExceeded
        if any limit is already met/exceeded — the caller must not dispatch
        in that case.

        Concurrency: cost_so_far/tokens_so_far are only known once a step's
        actual provider call finishes (record_step), which can be seconds
        after this check — the same "pre-bill" limitation documented in
        budget_tracker.py applies to those two dimensions here. The steps
        dimension does NOT have this gap: every non-exceeded outcome of this
        call reserves one step slot (state.pending_steps) under self._lock
        BEFORE returning, so N concurrent callers against the same run can't
        all observe "ok" before any of them records a step — the Nth caller
        past max_steps sees the reservations made by the first N-1 and is
        correctly blocked. The caller MUST eventually pair this with exactly
        one of record_step() (dispatch succeeded) or release_reservation()
        (dispatch was never attempted or failed) to release the slot.
        """
        self._maybe_housekeep()
        key = self._key(run_id, tenant_id)
        def reserve(
            state: _RunState | None,
        ) -> tuple[_RunState, tuple[str, dict[str, Any] | None]]:
            if state is None:
                state = _RunState(limits=RunLimits())
            state.last_touched = time.monotonic()

            frac, reason = _worst_fraction(state)
            if frac >= 1.0:
                # Return the exceeded summary as data instead of raising here:
                # self._store.update() is what actually persists `state` (with
                # the refreshed last_touched/TTL) — raising inside the updater
                # would make RedisRunStore's transaction bail out before its
                # pipe.set(), silently skipping the TTL refresh and letting an
                # over-budget run's key expire and reset to zero spend.
                return state, ("exceeded", _summary(run_id, state, reason))

            state.pending_steps += 1
            if frac >= RUN_WARN_THRESHOLD:
                result = "warning"
            elif frac >= RUN_DEGRADE_THRESHOLD:
                result = "degraded"
            else:
                result = "ok"
            return state, (result, None)
        kind, payload = self._store.update(key, reserve)
        if kind == "exceeded":
            # reserve() only pairs "exceeded" with a real summary dict (see
            # its body above) — payload is None only alongside "ok"/
            # "degraded"/"warning".
            assert payload is not None
            # run_id here is the caller-facing (unscoped) id — never leak the
            # tenant-scoped storage key into the exception/response.
            raise RunBudgetExceeded(run_id, payload)
        # `reserve()`'s inner `result` is only ever "ok"/"degraded"/"warning"
        # (the "exceeded" case returns via the branch above instead) — the
        # generic RunStore.update() plumbing types its return as bare str,
        # so this narrows back to what's actually always true at runtime.
        return cast(Literal["ok", "degraded", "warning"], kind)

    def record_step(
        self,
        run_id: str,
        model_id: str,
        cost_usd: float,
        tokens: int,
        tenant_id: str | None = None,
    ) -> None:
        """Record a completed step's actual cost/tokens, releasing the
        reservation check_before_dispatch() made for it. Never raises."""
        key = self._key(run_id, tenant_id)
        def record(state: _RunState | None) -> tuple[_RunState, None]:
            if state is None:
                state = _RunState(limits=RunLimits())
            state.steps.append(
                StepRecord(
                    model_id=model_id, cost_usd=cost_usd, tokens=tokens, timestamp=time.monotonic()
                )
            )
            state.pending_steps = max(0, state.pending_steps - 1)
            state.cost_so_far += cost_usd
            state.tokens_so_far += tokens
            state.last_touched = time.monotonic()
            return state, None
        self._store.update(key, record)

    def release_reservation(self, run_id: str, tenant_id: str | None = None) -> None:
        """Release a step slot reserved by check_before_dispatch() without a
        matching record_step() — call this whenever a step that passed the
        gate was never actually dispatched, or its dispatch failed outright
        (see Flux._dispatch_with_fallback / _complete_cascade,
        router/server.py's proxy handlers). Never raises; a no-op if the run
        is already gone (e.g. TTL-swept)."""
        key = self._key(run_id, tenant_id)
        def release(state: _RunState | None) -> tuple[_RunState | None, None]:
            if state is None:
                return None, None
            state.pending_steps = max(0, state.pending_steps - 1)
            return state, None
        self._store.update(key, release)

    def snapshot(self, run_id: str, tenant_id: str | None = None) -> tuple[float, int]:
        """Return (cost_so_far, steps_so_far) for a run, or (0.0, 0) if unknown."""
        state = self._store.get(self._key(run_id, tenant_id))
        if state is None:
            return 0.0, 0
        return state.cost_so_far, len(state.steps)

    def _maybe_housekeep(self) -> None:
        with self._lock:
            self._call_count += 1
            due = self._call_count % RUN_HOUSEKEEPING_INTERVAL == 0
        if due:
            self._store.sweep_expired(RUN_TTL_SECONDS)


def _fraction(value: float, limit: float) -> float:
    """value/limit, treating a limit of exactly 0 as "no budget at all" (blocks
    immediately) rather than "unset" — plain falsiness would silently disable
    the dimension instead, since 0 is falsy in Python. A negative limit is not
    a meaningful ceiling and is treated as unset."""
    if limit > 0:
        return value / limit
    if limit == 0:
        return 1.0
    return 0.0


def _worst_fraction(state: _RunState) -> tuple[float, str]:
    """Return (fraction, reason) for whichever limit is closest to being hit.

    The steps dimension counts pending_steps (checked-out but not yet
    recorded/released reservations) alongside completed steps — see
    RunBudget.check_before_dispatch()'s docstring for why.
    """
    elapsed = time.monotonic() - state.started_at
    candidates = [
        (_fraction(state.cost_so_far, state.limits.max_cost_usd), "cost"),
        (_fraction(len(state.steps) + state.pending_steps, state.limits.max_steps), "steps"),
        (_fraction(state.tokens_so_far, state.limits.max_tokens), "tokens"),
        (_fraction(elapsed, state.limits.max_duration_seconds), "duration"),
    ]
    return max(candidates, key=lambda c: c[0])


def _summary(run_id: str, state: _RunState, exceeded_reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "steps_taken": len(state.steps),
        "total_cost_usd": state.cost_so_far,
        "total_tokens": state.tokens_so_far,
        "duration_seconds": time.monotonic() - state.started_at,
        "exceeded_reason": exceeded_reason,
        "step_breakdown": [
            {
                "model_id": s.model_id,
                "cost_usd": s.cost_usd,
                "tokens": s.tokens,
            }
            for s in state.steps
        ],
    }
