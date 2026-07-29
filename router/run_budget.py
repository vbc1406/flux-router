"""
File: router/run_budget.py

Purpose:
Run-scoped budget enforcement (Task 3: the agent cost governance wedge).
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

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Protocol

import structlog

from .config import (
    RUN_DEGRADE_THRESHOLD,
    RUN_HOUSEKEEPING_INTERVAL,
    RUN_MAX_COST_USD,
    RUN_MAX_DURATION_SECONDS,
    RUN_MAX_STEPS,
    RUN_MAX_TOKENS,
    RUN_STORE_MAX_ENTRIES,
    RUN_TTL_SECONDS,
    RUN_WARN_THRESHOLD,
)

log = structlog.get_logger(__name__)


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


class RunBudgetExceeded(Exception):  # noqa: N818 — name matches the Task 3 public API exactly
    """
    Raised by check_before_dispatch() when a run has already met or exceeded
    any of its limits. Carries a structured summary so the caller can return
    partial results instead of nothing.
    """

    def __init__(self, run_id: str, summary: dict) -> None:
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


class RunBudget:
    """Public API for run-scoped budget enforcement. See module docstring."""

    def __init__(self, store: RunStore | None = None) -> None:
        self._store: RunStore = store or InMemoryRunStore()
        self._lock = threading.Lock()
        self._call_count = 0

    def start(self, run_id: str, limits: RunLimits | None = None) -> None:
        """Explicitly (re)initialize a run with its own limits. Optional —
        check_before_dispatch() lazily creates a run with global defaults if
        start() was never called."""
        self._store.set(run_id, _RunState(limits=limits or RunLimits()))

    def finish(self, run_id: str) -> None:
        """Drop a run's state immediately rather than waiting for TTL eviction."""
        self._store.delete(run_id)

    def check_before_dispatch(self, run_id: str) -> str:
        """
        Evaluate this run's state from steps recorded so far and return the
        budget_state for the step about to be dispatched: "ok", "degraded",
        or "warning". Raises RunBudgetExceeded if any limit is already
        met/exceeded — the caller must not dispatch in that case.
        """
        self._maybe_housekeep()
        state = self._store.get(run_id)
        if state is None:
            state = _RunState(limits=RunLimits())
        state.last_touched = time.monotonic()
        self._store.set(run_id, state)

        frac, reason = _worst_fraction(state)
        if frac >= 1.0:
            raise RunBudgetExceeded(run_id, _summary(run_id, state, reason))
        if frac >= RUN_WARN_THRESHOLD:
            return "warning"
        if frac >= RUN_DEGRADE_THRESHOLD:
            return "degraded"
        return "ok"

    def record_step(self, run_id: str, model_id: str, cost_usd: float, tokens: int) -> None:
        """Record a completed step's actual cost/tokens. Never raises."""
        state = self._store.get(run_id)
        if state is None:
            state = _RunState(limits=RunLimits())
        state.steps.append(
            StepRecord(
                model_id=model_id, cost_usd=cost_usd, tokens=tokens, timestamp=time.monotonic()
            )
        )
        state.cost_so_far += cost_usd
        state.tokens_so_far += tokens
        state.last_touched = time.monotonic()
        self._store.set(run_id, state)

    def snapshot(self, run_id: str) -> tuple[float, int]:
        """Return (cost_so_far, steps_so_far) for a run, or (0.0, 0) if unknown."""
        state = self._store.get(run_id)
        if state is None:
            return 0.0, 0
        return state.cost_so_far, len(state.steps)

    def _maybe_housekeep(self) -> None:
        with self._lock:
            self._call_count += 1
            due = self._call_count % RUN_HOUSEKEEPING_INTERVAL == 0
        if due:
            self._store.sweep_expired(RUN_TTL_SECONDS)


def _worst_fraction(state: _RunState) -> tuple[float, str]:
    """Return (fraction, reason) for whichever limit is closest to being hit."""
    elapsed = time.monotonic() - state.started_at
    candidates = [
        (
            state.cost_so_far / state.limits.max_cost_usd if state.limits.max_cost_usd else 0.0,
            "cost",
        ),
        (len(state.steps) / state.limits.max_steps if state.limits.max_steps else 0.0, "steps"),
        (
            state.tokens_so_far / state.limits.max_tokens if state.limits.max_tokens else 0.0,
            "tokens",
        ),
        (
            elapsed / state.limits.max_duration_seconds
            if state.limits.max_duration_seconds
            else 0.0,
            "duration",
        ),
    ]
    return max(candidates, key=lambda c: c[0])


def _summary(run_id: str, state: _RunState, exceeded_reason: str) -> dict:
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
