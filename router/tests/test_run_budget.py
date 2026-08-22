"""
File: router/tests/test_run_budget.py

Purpose:
Tests for run-scoped budget enforcement (router/run_budget.py) at
two levels:
  - Unit tests against RunBudget/InMemoryRunStore directly — the
    degradation ladder, the raise-before-not-after invariant, per-step
    breakdown, LRU + TTL eviction at scale, and cross-run isolation.
  - Integration tests through Flux.start_run()/Flux.complete() — a
    simulated multi-step agent loop that degrades to cheap models and then
    stops before overspending.

How to run:
  pytest -v router/tests/test_run_budget.py

How to add a test:
  1. Unit-level: build a RunBudget() (optionally with tight RunLimits) and
     drive check_before_dispatch()/record_step() directly.
  2. Integration-level: use _flux() for a mocked Flux instance (patches
     _call_model so no real HTTP happens), then flux.start_run(...) +
     repeated flux.complete(..., run_id=run_id).
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.flux import Flux
from router.model_registry import ModelRegistry
from router.provider_caller import ProviderResult
from router.routing_engine import RoutingEngine
from router.run_budget import InMemoryRunStore, RunBudget, RunBudgetExceeded, RunLimits

# ── Helpers ─────────────────────────────────────────────────────────────────


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
    return Flux(_engine(), api_key="sk-test")


def _pr(text: str) -> ProviderResult:
    return ProviderResult(
        text=text, input_tokens=None, output_tokens=None, usage_source="estimated"
    )


# ── Unit tests: degradation ladder + raise-before-not-after ────────────────


class TestDegradationLadder:
    def test_ok_then_degraded_then_warning_then_exceeded(self):
        rb = RunBudget()
        limits = RunLimits(
            max_cost_usd=1.0, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
        )
        rb.start("run-a", limits)

        # Nothing recorded yet.
        assert rb.check_before_dispatch("run-a") == "ok"
        rb.record_step("run-a", "cheap-model", 0.75, 100)  # cumulative 0.75

        # 0.75 >= RUN_DEGRADE_THRESHOLD (0.70) -> degraded
        assert rb.check_before_dispatch("run-a") == "degraded"
        rb.record_step("run-a", "cheap-model", 0.16, 100)  # cumulative 0.91

        # 0.91 >= RUN_WARN_THRESHOLD (0.90) -> warning
        assert rb.check_before_dispatch("run-a") == "warning"
        rb.record_step("run-a", "cheap-model", 0.10, 100)  # cumulative 1.01

        # Now at/over 1.0 -> raises, never returns a state string
        with pytest.raises(RunBudgetExceeded):
            rb.check_before_dispatch("run-a")

    def test_zero_limit_blocks_immediately_instead_of_disabling(self):
        """A limit explicitly set to 0 must mean 'no budget at all', not
        'unset' — 0 is falsy in Python, so a naive `if limit` check would
        silently disable that dimension instead of blocking on it."""
        rb = RunBudget()
        limits = RunLimits(
            max_cost_usd=0.0, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
        )
        rb.start("run-zero-cost", limits)
        with pytest.raises(RunBudgetExceeded) as exc_info:
            rb.check_before_dispatch("run-zero-cost")
        assert exc_info.value.summary["exceeded_reason"] == "cost"

    def test_never_exceeds_before_raising(self):
        """The step that would push cost over budget must never be recorded —
        check_before_dispatch() raises BEFORE that step's dispatch, so the
        run's recorded spend never exceeds its cap."""
        rb = RunBudget()
        limits = RunLimits(
            max_cost_usd=0.10, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
        )
        rb.start("run-b", limits)

        total_recorded = 0.0
        raised = False
        for _ in range(50):
            try:
                rb.check_before_dispatch("run-b")
            except RunBudgetExceeded as exc:
                raised = True
                # The exception's own summary must match what was actually recorded —
                # i.e. it reflects state as of BEFORE this (blocked) step.
                assert exc.summary["total_cost_usd"] == pytest.approx(total_recorded)
                assert exc.summary["total_cost_usd"] <= limits.max_cost_usd
                break
            rb.record_step("run-b", "m", 0.02, 50)
            total_recorded += 0.02

        assert raised, (
            "expected RunBudgetExceeded within 50 steps at $0.02/step against a $0.10 budget"
        )
        assert total_recorded <= limits.max_cost_usd

    def test_exceeded_summary_has_per_step_breakdown(self):
        rb = RunBudget()
        rb.start(
            "run-c",
            RunLimits(
                max_cost_usd=0.05, max_steps=1000, max_tokens=10**9, max_duration_seconds=10**9
            ),
        )
        rb.record_step("run-c", "model-1", 0.03, 10)
        rb.record_step("run-c", "model-2", 0.03, 10)

        with pytest.raises(RunBudgetExceeded) as excinfo:
            rb.check_before_dispatch("run-c")

        summary = excinfo.value.summary
        assert summary["steps_taken"] == 2
        assert summary["total_cost_usd"] == pytest.approx(0.06)
        assert summary["exceeded_reason"] == "cost"
        assert [s["model_id"] for s in summary["step_breakdown"]] == ["model-1", "model-2"]
        assert summary["step_breakdown"][0]["cost_usd"] == pytest.approx(0.03)

    def test_step_limit_triggers_before_cost_limit(self):
        rb = RunBudget()
        rb.start(
            "run-d",
            RunLimits(
                max_cost_usd=1000.0, max_steps=2, max_tokens=10**9, max_duration_seconds=10**9
            ),
        )
        rb.record_step("run-d", "m", 0.0001, 1)
        rb.record_step("run-d", "m", 0.0001, 1)

        with pytest.raises(RunBudgetExceeded) as excinfo:
            rb.check_before_dispatch("run-d")
        assert excinfo.value.summary["exceeded_reason"] == "steps"

    def test_run_with_no_start_call_uses_global_defaults(self):
        rb = RunBudget()
        # No rb.start() — check_before_dispatch must lazily create the run.
        assert rb.check_before_dispatch("run-lazy") == "ok"
        cost, steps = rb.snapshot("run-lazy")
        assert (cost, steps) == (0.0, 0)


class TestConcurrentDispatchReservation:
    """Regression: check_before_dispatch() used to only look at already-
    RECORDED steps, so N requests dispatched concurrently against the same
    run (record_step() only happens after each one's slow provider call
    finishes) could all pass the max_steps gate before any of them recorded
    anything — a caller could blow straight through the step cap by firing
    requests in parallel instead of sequentially. check_before_dispatch()
    now reserves a step slot atomically as part of the same check, so a
    concurrent burst is correctly throttled at the limit."""

    def test_burst_of_concurrent_checks_stops_at_max_steps(self):
        rb = RunBudget()
        rb.start(
            "run-burst",
            RunLimits(
                max_cost_usd=1000.0, max_steps=5, max_tokens=10**9, max_duration_seconds=10**9
            ),
        )

        allowed = 0
        blocked = 0
        lock = threading.Lock()

        def worker():
            nonlocal allowed, blocked
            try:
                rb.check_before_dispatch("run-burst")
            except RunBudgetExceeded:
                with lock:
                    blocked += 1
                return
            with lock:
                allowed += 1
            # Simulate a slow provider call finishing well after every other
            # thread's check_before_dispatch() has already run.
            rb.record_step("run-burst", "m", 0.0001, 1)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert allowed == 5
        assert blocked == 15

    def test_release_reservation_frees_the_slot_for_a_failed_dispatch(self):
        rb = RunBudget()
        rb.start(
            "run-release",
            RunLimits(
                max_cost_usd=1000.0, max_steps=1, max_tokens=10**9, max_duration_seconds=10**9
            ),
        )
        assert rb.check_before_dispatch("run-release") == "ok"
        # Simulate the dispatch failing outright (never reaches record_step).
        rb.release_reservation("run-release")
        # The freed slot is available again — not permanently leaked.
        assert rb.check_before_dispatch("run-release") == "ok"
        rb.record_step("run-release", "m", 0.0001, 1)
        with pytest.raises(RunBudgetExceeded):
            rb.check_before_dispatch("run-release")


# ── Unit tests: eviction at scale ───────────────────────────────────────────


class TestEviction:
    def test_lru_cap_keeps_store_bounded_across_10000_abandoned_runs(self):
        store = InMemoryRunStore(max_entries=200)
        rb = RunBudget(store=store)

        for i in range(10_000):
            rb.check_before_dispatch(f"abandoned-{i}")  # never touched again

        assert len(store._states) <= 200

    def test_ttl_sweep_evicts_idle_runs(self, monkeypatch):
        import time

        import router.run_budget as rb_module

        # A tiny real TTL + housekeeping on every call, so a real (short) sleep
        # is enough to exercise the sweep without faking the clock — the
        # RunLimits/_RunState default_factory=time.monotonic binding is
        # resolved at import time, so patching time.monotonic afterwards
        # would not affect already-defined dataclass field factories anyway.
        monkeypatch.setattr(rb_module, "RUN_TTL_SECONDS", 0.05)
        monkeypatch.setattr(rb_module, "RUN_HOUSEKEEPING_INTERVAL", 1)

        store = InMemoryRunStore(max_entries=1000)
        rb = RunBudget(store=store)
        rb.check_before_dispatch("idle-run")
        assert len(store._states) == 1

        time.sleep(0.15)  # well past the 0.05s TTL
        rb.check_before_dispatch("fresh-run")  # triggers housekeeping (interval=1)

        assert "idle-run" not in store._states
        assert "fresh-run" in store._states

    def test_concurrent_runs_do_not_leak_state(self):
        rb = RunBudget()
        run_ids = [f"concurrent-{i}" for i in range(50)]
        errors: list[Exception] = []

        def worker(run_id: str) -> None:
            try:
                for _step in range(20):
                    rb.check_before_dispatch(run_id)
                    rb.record_step(run_id, "m", 0.001, 10)
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(rid,)) for rid in run_ids]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for run_id in run_ids:
            cost, steps = rb.snapshot(run_id)
            assert steps == 20, f"{run_id} saw {steps} steps, expected 20 (no cross-run leakage)"
            assert cost == pytest.approx(0.02)


# ── Integration tests: Flux.start_run() + Flux.complete() ──────────────────


class TestAgentLoopIntegration:
    def test_50_step_loop_degrades_then_stops_before_exceeding(self):
        flux = _flux()
        mock_call = AsyncMock(return_value=_pr("ok"))
        flux._call_model = mock_call  # type: ignore[method-assign]

        budget_states: list[str] = []
        priorities_used: list[str] = []
        exceeded: RunBudgetExceeded | None = None

        # Bugfix note: this cap must stay comfortably below 50 steps *
        # (per-step cost of the CHEAPEST available cost-optimized model),
        # not just below the cost of whatever model happened to be cheapest
        # when this test was written — the model catalog grows over time
        # (see models.json) and a newly-added cheaper model must not make
        # this test flake by never crossing the cap in 50 steps.
        max_cost_usd = 0.01

        async def run_loop():
            nonlocal exceeded
            with flux.start_run(max_cost_usd=max_cost_usd, max_steps=50) as run_id:
                for _ in range(50):
                    try:
                        resp = await flux.complete(
                            "Take the next action.", run_id=run_id, user_id="agent-1"
                        )
                    except RunBudgetExceeded as exc:
                        exceeded = exc
                        return
                    budget_states.append(resp.decision.budget_state)
                    priorities_used.append(resp.decision.priority_applied)

        rr(run_loop())

        assert exceeded is not None, (
            f"50 steps should exceed a ${max_cost_usd} run budget well before step 50"
        )
        # The summary reflects cumulative spend as of the LAST successfully
        # dispatched step (check_before_dispatch only looks at prior steps,
        # so a step's own cost can tick the total slightly past the cap —
        # the invariant is that the NEXT step is blocked, not that the total
        # can never nominally cross the line by up to one step's cost).
        assert exceeded.summary["total_cost_usd"] < max_cost_usd * 1.5
        assert exceeded.summary["steps_taken"] == len(budget_states)
        assert len(exceeded.summary["step_breakdown"]) == exceeded.summary["steps_taken"]
        # The ladder must have kicked in before the hard stop.
        assert "degraded" in budget_states or "warning" in budget_states
        assert "cost-optimized" in priorities_used

    def test_run_state_is_isolated_between_two_concurrent_runs(self):
        flux = _flux()
        flux._call_model = AsyncMock(return_value=_pr("ok"))  # type: ignore[method-assign]

        async def drive(run_id: str, n: int):
            for _ in range(n):
                await flux.complete("step", run_id=run_id, user_id="u")

        async def both():
            # Kept as nested `with` rather than one parenthesized statement —
            # this repo's local dev interpreter predates the 3.10 syntax for that.
            with flux.start_run(run_id="run-x", max_cost_usd=10.0, max_steps=1000) as rx:  # noqa: SIM117
                with flux.start_run(run_id="run-y", max_cost_usd=10.0, max_steps=1000) as ry:
                    await asyncio.gather(drive(rx, 5), drive(ry, 3))
                    # Snapshot INSIDE the context — start_run() drops state on exit.
                    run_budget = flux._engine._run_budget
                    return run_budget.snapshot(rx), run_budget.snapshot(ry)

        (cost_x, steps_x), (cost_y, steps_y) = rr(both())
        assert steps_x == 5
        assert steps_y == 3


# ── Token ceiling with mixed actual/estimated steps ─────────────────────────


class TestTokenCeilingWithMixedUsageSources:
    """Regression: run_budget.record_step()'s token count now reflects the
    provider's own reported usage when available (see flux.py's
    billed_tokens), not always the ~4-chars/token estimate. A run mixing
    actual-usage steps and estimated-fallback steps must still trip
    RUN_MAX_TOKENS correctly against the true cumulative total."""

    def test_run_trips_token_ceiling_using_actual_usage(self):
        flux = _flux()
        calls: list[int] = []

        async def mock_call(model, request):
            calls.append(1)
            if len(calls) == 1:
                # Actual usage: 100 + 100 = 200 tokens.
                return ProviderResult(
                    text="step one", input_tokens=100, output_tokens=100, usage_source="provider"
                )
            # No provider usage -> falls back to char-math estimate. 100
            # chars -> ~25 estimated tokens; combined with step one's 200
            # ACTUAL tokens (225 total), this must trip a 220-token ceiling
            # on the third call — a char-math guess for step one's own
            # 8-char text ("step one" -> ~2 tokens) would never get there.
            return ProviderResult(
                text="x" * 100, input_tokens=None, output_tokens=None, usage_source="estimated"
            )

        flux._call_model = mock_call  # type: ignore[method-assign]

        exceeded: RunBudgetExceeded | None = None

        async def run_loop():
            nonlocal exceeded
            with flux.start_run(
                max_cost_usd=1000.0, max_steps=1000, max_tokens=220
            ) as run_id:
                for _ in range(5):
                    try:
                        await flux.complete("go", run_id=run_id, user_id="u_tokens")
                    except RunBudgetExceeded as exc:
                        exceeded = exc
                        return

        rr(run_loop())

        assert exceeded is not None, "220-token ceiling should trip within 5 steps"
        # Step 1 alone (200 actual tokens) is under the 250 ceiling, so the
        # ceiling could only have tripped because step 1's ACTUAL 200 tokens
        # were counted, not a smaller char-math guess for "step one" (an
        # 8-char string -> ~2 estimated tokens, which would never trip a
        # 250-token ceiling within 5 short steps at all).
        assert exceeded.summary["total_tokens"] >= 200

    def test_step_breakdown_reflects_actual_tokens_not_char_math(self):
        flux = _flux()

        async def mock_call(model, request):
            return ProviderResult(
                text="x", input_tokens=500, output_tokens=500, usage_source="provider"
            )

        flux._call_model = mock_call  # type: ignore[method-assign]

        async def run_loop():
            with flux.start_run(max_cost_usd=1000.0, max_steps=1000, max_tokens=100) as run_id:
                with pytest.raises(RunBudgetExceeded) as exc_info:
                    for _ in range(3):
                        await flux.complete("go", run_id=run_id, user_id="u_tokens2")
                return exc_info.value

        exc = rr(run_loop())
        # "x" alone char-math estimates to ~1 token — only the actual 1000
        # (500+500) reported by the provider explains tripping a 100-token
        # ceiling on the very first step.
        assert exc.summary["step_breakdown"][0]["tokens"] == 1000
