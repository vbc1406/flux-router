"""
File: router/tests/test_adaptive_weights.py

Purpose:
Comprehensive tests for AdaptiveWeights — EMA correctness, rollback/snapshot
integrity, signal-stats poisoning resistance, drift detection, and config
constant wiring.

How to run:
  pytest -v router/tests/test_adaptive_weights.py
  pytest -v router/tests/test_adaptive_weights.py::TestRollbackRestoration

How to add a test:
  1. Use AdaptiveWeights(weight_file=None) (in-memory, no disk I/O).
  2. Call aw.record(model, task, quality, latency) to feed signals.
  3. Assert on aw.get_adjusted_score(model, task) or internal _state fields.
  4. For snapshot/rollback tests, call aw._maybe_snapshot() then corrupt _state.

Test classes:
  TestMetrics               — EMA convergence, decay, per-customer isolation
  TestRollbackRestoration   — rollback restores _state and _signal_stats
  TestSignalStatsPoisoning  — Welford stats survive injected outliers
  TestConfigConstants       — ADAPTIVE_MIN_SAMPLES, PER_CUSTOMER_MIN_SAMPLES wiring
  TestSlowDriftDetection    — gradual drift triggers rollback between snapshots
  TestSnapshotIntegrity     — snapshot structure, capping at _MAX_SNAPSHOTS
  TestEndToEndIntegration   — realistic multi-model, multi-task timeline

Covers:
  - Rollback restoration of _state and _signal_stats
  - Signal stats poisoning scenario (the regression we're guarding)
  - Config constant usage (ADAPTIVE_MIN_SAMPLES, PER_CUSTOMER_MIN_SAMPLES)
  - Slow drift detection between snapshots
  - Snapshot integrity (structure, capping)
  - End-to-end integration timeline
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any

import pytest

import router.adaptive_weights as aw_mod
from router.adaptive_weights import (
    _MAX_SNAPSHOTS,
    _OUTLIER_STD_MULTIPLIER,
    _QUALITY_FLOOR,
    _ROLLBACK_DROP_THRESHOLD,
    AdaptiveWeights,
)
from router.config import ADAPTIVE_MIN_SAMPLES, PER_CUSTOMER_MIN_SAMPLES

# ── Metrics helpers ──────────────────────────────────────────────────────────


@dataclass
class TestMetrics:
    test_name: str
    initial_state: dict[str, Any]
    action_description: str
    final_state: dict[str, Any]
    expected_value: float
    actual_value: float
    tolerance: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return abs(self.actual_value - self.expected_value) <= self.tolerance

    def print_report(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        bar = "=" * 60
        print(f"\n{bar}")
        print(f"TEST: {self.test_name}")
        print(bar)
        for k, v in self.initial_state.items():
            print(f"  Initial {k}: {v}")
        print(f"  Action:  {self.action_description}")
        for k, v in self.final_state.items():
            print(f"  Final   {k}: {v}")
        print(f"  Expected: {self.expected_value}  ±{self.tolerance}")
        print(f"  Actual:   {self.actual_value:.6f}")
        print(f"  Deviation: {abs(self.actual_value - self.expected_value):.6f}")
        for k, v in self.extra.items():
            print(f"  {k}: {v}")
        print(f"  {'✅' if self.passed else '❌'} {status}")


_ALL_METRICS: list[TestMetrics] = []


def _record_metric(m: TestMetrics) -> TestMetrics:
    _ALL_METRICS.append(m)
    m.print_report()
    return m


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def aw() -> AdaptiveWeights:
    """Fresh AdaptiveWeights with no disk I/O."""
    return AdaptiveWeights(state_file=None)


@pytest.fixture
def aw_snap10(monkeypatch) -> AdaptiveWeights:
    """AdaptiveWeights that snapshots every 10 signals (speeds up snapshot tests)."""
    monkeypatch.setattr(aw_mod, "_SNAPSHOT_INTERVAL", 10)
    return AdaptiveWeights(state_file=None)


# ── Helpers ───────────────────────────────────────────────────────────────────


def inject_signals(
    aw: AdaptiveWeights,
    model: str,
    task: str,
    n: int,
    mean: float,
    std: float = 0.02,
    seed: int = 42,
) -> list[float]:
    """Record n quality scores drawn from N(mean, std), return the values used."""
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(n):
        v = max(0.0, min(1.0, rng.gauss(mean, std)))
        aw.record(model, task, v)
        values.append(v)
    return values


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def avg_quality(state: dict) -> float:
    vals = [v.get("avg_quality", 0.5) for v in state.values()]
    return sum(vals) / len(vals) if vals else 0.5


def print_comparison_table(rows: list[tuple[str, Any, Any]]) -> None:
    """Print a before/after comparison table."""
    col_w = 28
    print(f"\n  {'Metric':<{col_w}} {'Before':>12}  {'After':>12}")
    print(f"  {'-' * col_w} {'-' * 12}  {'-' * 12}")
    for label, before, after in rows:
        before_s = f"{before:.6f}" if isinstance(before, float) else str(before)
        after_s = f"{after:.6f}" if isinstance(after, float) else str(after)
        print(f"  {label:<{col_w}} {before_s:>12}  {after_s:>12}")


# ── Part 1: Rollback Restoration Tests ───────────────────────────────────────


class TestRollbackRestoration:
    """Rollback must faithfully restore both _state and _signal_stats."""

    def test_rollback_restores_state(self, aw_snap10):
        aw = aw_snap10
        model, task = "gpt-4o", "summarization"
        key = f"{model}:{task}"

        # Build enough signals to trigger a snapshot (interval = 10)
        inject_signals(aw, model, task, 12, mean=0.85)
        assert len(aw._snapshots) >= 1

        pre_avg = avg_quality(aw._state)

        # Corrupt state to trigger the rollback threshold
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.05

        corrupted_avg = avg_quality(aw._state)
        aw._check_for_corruption()
        restored_avg = avg_quality(aw._state)

        m = _record_metric(
            TestMetrics(
                test_name="Rollback Restores _state avg_quality",
                initial_state={"pre_corruption_avg": round(pre_avg, 4)},
                action_description="Corrupt all avg_quality to 0.05, call _check_for_corruption()",
                final_state={
                    "corrupted_avg": round(corrupted_avg, 4),
                    "restored_avg": round(restored_avg, 4),
                },
                expected_value=pre_avg,
                actual_value=restored_avg,
                tolerance=0.05,
            )
        )
        assert m.passed
        assert restored_avg > corrupted_avg

    def test_rollback_restores_signal_stats_mean(self, aw_snap10):
        aw = aw_snap10
        model, task = "claude-sonnet", "reasoning"
        key = f"{model}:{task}"

        inject_signals(aw, model, task, 12, mean=0.82, std=0.03)
        assert len(aw._snapshots) >= 1

        snap_mean = aw._snapshots[-1]["signal_stats"].get(key, {}).get("mean")
        assert snap_mean is not None, "snapshot must contain signal_stats for key"

        # Poison signal_stats
        aw._signal_stats[key]["mean"] = 0.01
        aw._signal_stats[key]["variance"] = 999.0
        poisoned_mean = aw._signal_stats[key]["mean"]

        # Corrupt state to trigger rollback
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.05
        aw._check_for_corruption()

        restored_mean = aw._signal_stats[key]["mean"]

        m = _record_metric(
            TestMetrics(
                test_name="Rollback Restores signal_stats.mean",
                initial_state={"snap_mean": round(snap_mean, 4)},
                action_description="Poison signal_stats.mean=0.01, variance=999; trigger rollback",
                final_state={
                    "poisoned_mean": poisoned_mean,
                    "restored_mean": round(restored_mean, 4),
                },
                expected_value=snap_mean,
                actual_value=restored_mean,
                tolerance=0.05,
            )
        )
        assert m.passed

    def test_rollback_restores_signal_stats_variance(self, aw_snap10):
        aw = aw_snap10
        model, task = "claude-sonnet", "code_generation"
        key = f"{model}:{task}"

        inject_signals(aw, model, task, 12, mean=0.80, std=0.03)
        snap_var = aw._snapshots[-1]["signal_stats"].get(key, {}).get("variance", 0.0)

        aw._signal_stats[key]["variance"] = 999.0
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.05
        aw._check_for_corruption()

        restored_var = aw._signal_stats[key]["variance"]

        m = _record_metric(
            TestMetrics(
                test_name="Rollback Restores signal_stats.variance",
                initial_state={"snap_variance": round(snap_var, 6)},
                action_description="Poison variance=999; trigger rollback",
                final_state={"restored_variance": round(restored_var, 6)},
                expected_value=snap_var,
                actual_value=restored_var,
                tolerance=0.01,
            )
        )
        assert m.passed

    def test_post_rollback_signals_use_restored_bounds(self, aw_snap10):
        """After rollback, new signals must be evaluated against restored outlier bounds."""
        aw = aw_snap10
        model, task = "gpt-4o-mini", "classification"
        key = f"{model}:{task}"

        # Build signal stats with mean ~0.85, std ~0.02 (very tight)
        inject_signals(aw, model, task, 12, mean=0.85, std=0.02)
        pre_count = aw._state[key]["sample_count"]

        # Force rollback by corrupting state
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.05
        aw._check_for_corruption()
        post_rollback_count = aw._state[key]["sample_count"]

        # 0.5 is many σ away from mean 0.85 with std≈0.02 → should be rejected
        aw.record(model, task, 0.50)
        count_after_outlier = aw._state[key]["sample_count"]

        # 0.84 is within 2σ of mean 0.85 → should be accepted
        aw.record(model, task, 0.84)
        count_after_normal = aw._state[key]["sample_count"]

        print_comparison_table(
            [
                ("pre_rollback sample_count", pre_count, post_rollback_count),
                ("after outlier (0.50)", post_rollback_count, count_after_outlier),
                ("after normal (0.84)", count_after_outlier, count_after_normal),
            ]
        )

        # The outlier (0.50) must be rejected (count unchanged)
        assert count_after_outlier == post_rollback_count, (
            f"Outlier 0.50 should be rejected post-rollback (count was {post_rollback_count}, now {count_after_outlier})"
        )
        # The normal signal (0.84) must be accepted
        assert count_after_normal == post_rollback_count + 1, (
            "Normal 0.84 should be accepted post-rollback"
        )


# ── Part 2: Signal Stats Poisoning Test ──────────────────────────────────────


class TestSignalStatsPoisoning:
    """The core regression: rollback must restore signal_stats, not just _state."""

    def test_outliers_rejected_before_stats_update(self, aw_snap10):
        """
        50 good signals → snapshot → try injecting 10 bad outliers → verify
        they don't corrupt signal_stats.mean (they're rejected first).
        """
        aw = aw_snap10
        model, task = "gpt-4o", "analysis"
        key = f"{model}:{task}"

        good_vals = inject_signals(aw, model, task, 50, mean=0.85, std=0.02)
        assert len(aw._snapshots) >= 1

        before_mean = aw._signal_stats[key]["mean"]
        before_count = aw._signal_stats[key]["count"]

        # Attempt to inject 10 bad outliers — must be rejected
        rejected = 0
        for _ in range(10):
            cnt_before = aw._state[key]["sample_count"]
            aw.record(model, task, 0.10)
            if aw._state[key]["sample_count"] == cnt_before:
                rejected += 1

        after_mean = aw._signal_stats[key]["mean"]
        after_count = aw._signal_stats[key]["count"]
        mean_deviation = abs(after_mean - before_mean)

        m = _record_metric(
            TestMetrics(
                test_name="Outliers Rejected — signal_stats.mean Unchanged",
                initial_state={
                    "good_signals": 50,
                    "signal_mean_before": round(before_mean, 4),
                    "signal_count_before": before_count,
                },
                action_description="Inject 10 outliers at 0.10 (far from mean ~0.85)",
                final_state={
                    "rejected": rejected,
                    "mean_after": round(after_mean, 4),
                    "count_after": after_count,
                    "mean_deviation": round(mean_deviation, 6),
                },
                expected_value=before_mean,
                actual_value=after_mean,
                tolerance=0.05,
                extra={"outliers_rejected": rejected},
            )
        )
        assert rejected >= 8, f"Expected most outliers rejected, got {rejected}/10"
        assert m.passed

    def test_rollback_restores_clean_mean_after_corruption(self, aw_snap10):
        """
        1. Record 50 good signals (mean≈0.85) → snapshot taken at signal 10.
        2. Manually corrupt signal_stats.mean (simulates a bug that bypassed outlier guard).
        3. Corrupt _state to trip the rollback threshold.
        4. Verify rollback restores mean to ~0.85, not the poisoned value.
        """
        aw = aw_snap10
        model, task = "gpt-4o", "code_review"
        key = f"{model}:{task}"

        inject_signals(aw, model, task, 50, mean=0.85, std=0.02)
        assert len(aw._snapshots) >= 1

        snap_mean = aw._snapshots[-1]["signal_stats"][key]["mean"]
        before_mean = aw._signal_stats[key]["mean"]

        # Simulate corruption that bypassed the outlier guard
        aw._signal_stats[key]["mean"] = 0.05
        aw._signal_stats[key]["variance"] = 500.0
        poisoned_mean = aw._signal_stats[key]["mean"]

        # Trip rollback threshold
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.04
        aw._check_for_corruption()

        restored_mean = aw._signal_stats[key]["mean"]
        deviation = abs(restored_mean - snap_mean)

        m = _record_metric(
            TestMetrics(
                test_name="Signal Stats Poisoning — Rollback Restores Clean Mean",
                initial_state={
                    "good_signals_recorded": 50,
                    "signal_mean_before_poison": round(before_mean, 4),
                    "snapshot_mean": round(snap_mean, 4),
                },
                action_description="Corrupt signal_stats.mean=0.05, trip rollback, call _check_for_corruption()",
                final_state={
                    "poisoned_mean": poisoned_mean,
                    "restored_mean": round(restored_mean, 4),
                    "deviation_from_snapshot": round(deviation, 6),
                },
                expected_value=snap_mean,
                actual_value=restored_mean,
                tolerance=0.01,
                extra={
                    "next_outlier_bound_approx": f"{restored_mean:.3f} ± {_OUTLIER_STD_MULTIPLIER}σ"
                },
            )
        )
        assert m.passed, (
            f"signal_stats.mean not restored: got {restored_mean:.4f}, expected ~{snap_mean:.4f}"
        )

    def test_new_signals_post_rollback_use_correct_outlier_bounds(self, aw_snap10):
        """
        After restoring signal_stats, the next batch of signals must be
        evaluated against the restored bounds — not the poisoned ones.
        """
        aw = aw_snap10
        model, task = "gpt-4o", "extraction"
        key = f"{model}:{task}"

        inject_signals(aw, model, task, 50, mean=0.85, std=0.02)
        snap_mean = aw._snapshots[-1]["signal_stats"][key]["mean"]
        snap_std = math.sqrt(max(0, aw._snapshots[-1]["signal_stats"][key]["variance"]))

        # Poison + rollback
        aw._signal_stats[key]["mean"] = 0.50  # poison to center of range
        aw._signal_stats[key]["variance"] = 0.09  # std≈0.30 — huge window
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.04
        aw._check_for_corruption()

        # 0.60 is >2σ from restored mean 0.85 (with std≈0.02) → should be rejected
        count_pre = aw._state[key]["sample_count"]
        aw.record(model, task, 0.60)
        count_post = aw._state[key]["sample_count"]
        rejected_by_restored_bounds = count_post == count_pre

        print(f"\n  Restored mean: {snap_mean:.4f}, std≈{snap_std:.4f}")
        print(f"  2σ bound: [{snap_mean - 2 * snap_std:.4f}, {snap_mean + 2 * snap_std:.4f}]")
        print(f"  Signal 0.60 rejected: {rejected_by_restored_bounds}")

        assert rejected_by_restored_bounds, (
            "Signal 0.60 should be rejected after rollback restores tight bounds around 0.85"
        )


# ── Part 3: Config Constant Tests ────────────────────────────────────────────


class TestConfigConstants:
    """get_adjusted_score must use constants from config, not hardcoded literals."""

    def test_get_adjusted_score_uses_adaptive_min_samples(self, monkeypatch):
        """Verify get_adjusted_score checks ADAPTIVE_MIN_SAMPLES, not a literal 20."""
        # Default is 20; verify at exactly threshold - 1, score is base
        aw = AdaptiveWeights(state_file=None)
        model, task, base = "m", "t", 0.70

        for _ in range(ADAPTIVE_MIN_SAMPLES - 1):
            aw.record(model, task, 0.95, base)

        score_before = aw.get_adjusted_score(model, task, base)

        # One more → at threshold → adaptive kicks in
        aw.record(model, task, 0.95, base)
        score_after = aw.get_adjusted_score(model, task, base)

        m = _record_metric(
            TestMetrics(
                test_name=f"get_adjusted_score Uses ADAPTIVE_MIN_SAMPLES ({ADAPTIVE_MIN_SAMPLES})",
                initial_state={
                    "samples_recorded": ADAPTIVE_MIN_SAMPLES - 1,
                    "score_at_N-1": score_before,
                },
                action_description=f"Record 1 more signal to reach ADAPTIVE_MIN_SAMPLES={ADAPTIVE_MIN_SAMPLES}",
                final_state={
                    "samples_recorded": ADAPTIVE_MIN_SAMPLES,
                    "score_at_N": round(score_after, 4),
                },
                expected_value=base,
                actual_value=score_before,
                tolerance=1e-9,
            )
        )
        assert score_before == base, f"Score should be base before {ADAPTIVE_MIN_SAMPLES} samples"
        assert score_after > base, f"Score should be adaptive at {ADAPTIVE_MIN_SAMPLES} samples"

    def test_lowering_adaptive_min_samples_affects_activation(self, monkeypatch):
        """Patch ADAPTIVE_MIN_SAMPLES to 5; verify adaptive kicks in after 5 signals."""
        monkeypatch.setattr(aw_mod, "ADAPTIVE_MIN_SAMPLES", 5)
        aw = AdaptiveWeights(state_file=None)
        model, task, base = "m", "t", 0.60

        for _ in range(4):
            aw.record(model, task, 0.95, base)
        score_at_4 = aw.get_adjusted_score(model, task, base)

        aw.record(model, task, 0.95, base)
        score_at_5 = aw.get_adjusted_score(model, task, base)

        m = _record_metric(
            TestMetrics(
                test_name="Lowering ADAPTIVE_MIN_SAMPLES to 5 Activates Earlier",
                initial_state={"patched_threshold": 5, "base_score": base},
                action_description="Record 4 signals (below new threshold), then 1 more",
                final_state={
                    "score_at_4_samples": score_at_4,
                    "score_at_5_samples": round(score_at_5, 4),
                },
                expected_value=base,
                actual_value=score_at_4,
                tolerance=1e-9,
            )
        )
        assert score_at_4 == base
        assert score_at_5 > base

    def test_per_customer_min_samples_threshold(self, monkeypatch):
        """Per-customer adaptive activates exactly at PER_CUSTOMER_MIN_SAMPLES."""
        aw = AdaptiveWeights(state_file=None)
        model, task, base, cid = "m", "t", 0.65, "cust-threshold-test"

        for _ in range(PER_CUSTOMER_MIN_SAMPLES - 1):
            aw.record(model, task, 0.92, base, customer_id=cid)
        score_before = aw.get_adjusted_score(model, task, base, customer_id=cid)

        aw.record(model, task, 0.92, base, customer_id=cid)
        score_after = aw.get_adjusted_score(model, task, base, customer_id=cid)

        m = _record_metric(
            TestMetrics(
                test_name=f"PER_CUSTOMER_MIN_SAMPLES={PER_CUSTOMER_MIN_SAMPLES} Activation",
                initial_state={
                    "threshold": PER_CUSTOMER_MIN_SAMPLES,
                    "samples_at_N-1": PER_CUSTOMER_MIN_SAMPLES - 1,
                },
                action_description=f"Record N={PER_CUSTOMER_MIN_SAMPLES}th customer sample",
                final_state={
                    "score_at_N-1": score_before,
                    "score_at_N": round(score_after, 4),
                },
                expected_value=base,
                actual_value=score_before,
                tolerance=1e-9,
            )
        )
        assert score_before == base, "Should use base score before threshold"
        assert score_after > base, "Should use per-customer adaptive score at threshold"

    def test_adaptive_min_samples_constant_not_hardcoded_20(self, monkeypatch):
        """Patch threshold to 7; confirm behavior changes — proves no hardcoded 20."""
        monkeypatch.setattr(aw_mod, "ADAPTIVE_MIN_SAMPLES", 7)
        aw = AdaptiveWeights(state_file=None)
        model, task, base = "probe", "analysis", 0.55

        for _ in range(7):
            aw.record(model, task, 0.90, base)
        score = aw.get_adjusted_score(model, task, base)

        m = _record_metric(
            TestMetrics(
                test_name="No Hardcoded 20: ADAPTIVE_MIN_SAMPLES=7 Works",
                initial_state={"patched_threshold": 7, "base_score": base},
                action_description="Record 7 signals with threshold patched to 7",
                final_state={"adjusted_score": round(score, 4)},
                expected_value=base + 0.001,  # just above base — adaptive must have kicked in
                actual_value=score,
                tolerance=0.3,
            )
        )
        assert score > base, (
            f"Adaptive should activate at 7 samples when threshold is 7, got {score}"
        )


# ── Part 4: Slow Drift Detection ─────────────────────────────────────────────


class TestSlowDriftDetection:
    """
    Gradual quality decline must trigger rollback on _check_for_corruption(),
    which is called on every record() — not just at snapshot boundaries.
    """

    def test_gradual_decline_triggers_rollback(self, aw_snap10):
        aw = aw_snap10
        model, task = "gpt-4o", "code_review"
        key = f"{model}:{task}"

        # Establish a clean snapshot
        inject_signals(aw, model, task, 12, mean=0.90)
        baseline = aw._last_snapshot_avg
        assert baseline is not None

        # Simulate slow drift: each step degrades slightly
        # Target: just below rollback threshold (90% of baseline)
        target = baseline * _ROLLBACK_DROP_THRESHOLD - 0.01
        steps = []
        current = baseline
        for step in range(1, 8):
            current = current * 0.98  # 2% degradation per step
            for k in aw._state:
                aw._state[k]["avg_quality"] = current
            steps.append((step, round(current, 4)))
            if current < baseline * _ROLLBACK_DROP_THRESHOLD:
                break

        pre_rollback_avg = avg_quality(aw._state)
        aw._check_for_corruption()
        post_rollback_avg = avg_quality(aw._state)

        print(f"\n  Baseline (snapshot avg): {baseline:.4f}")
        print(f"  Rollback threshold:      {baseline * _ROLLBACK_DROP_THRESHOLD:.4f}")
        print("  Degradation steps:")
        for step, q in steps:
            marker = " ← trigger" if q < baseline * _ROLLBACK_DROP_THRESHOLD else ""
            print(f"    step {step}: {q:.4f}{marker}")
        print(f"  Pre-rollback avg:  {pre_rollback_avg:.4f}")
        print(f"  Post-rollback avg: {post_rollback_avg:.4f}")

        m = _record_metric(
            TestMetrics(
                test_name="Slow Drift Triggers Rollback Before Critical Threshold",
                initial_state={
                    "baseline": round(baseline, 4),
                    "threshold": round(baseline * _ROLLBACK_DROP_THRESHOLD, 4),
                },
                action_description="Degrade avg_quality 2% per step until below rollback threshold",
                final_state={
                    "pre_rollback_avg": round(pre_rollback_avg, 4),
                    "post_rollback_avg": round(post_rollback_avg, 4),
                },
                expected_value=baseline,
                actual_value=post_rollback_avg,
                tolerance=0.10,
            )
        )
        assert post_rollback_avg > pre_rollback_avg, (
            "Rollback should restore avg above degraded level"
        )

    def test_rollback_not_triggered_before_threshold(self, aw_snap10):
        """Quality at exactly 91% of snapshot avg must NOT trigger rollback."""
        aw = aw_snap10
        model, task = "gpt-4o", "translation"
        inject_signals(aw, model, task, 12, mean=0.90)
        baseline = aw._last_snapshot_avg

        # 91% > 90% → rollback threshold not crossed
        safe_degraded = baseline * (_ROLLBACK_DROP_THRESHOLD + 0.01)
        for k in aw._state:
            aw._state[k]["avg_quality"] = safe_degraded

        aw._check_for_corruption()
        post_avg = avg_quality(aw._state)

        m = _record_metric(
            TestMetrics(
                test_name="No Rollback When Drift Is Within Threshold",
                initial_state={
                    "baseline": round(baseline, 4),
                    "threshold_pct": _ROLLBACK_DROP_THRESHOLD,
                },
                action_description=f"Set avg to {_ROLLBACK_DROP_THRESHOLD + 0.01:.0%} of baseline (above threshold)",
                final_state={"post_check_avg": round(post_avg, 4)},
                expected_value=safe_degraded,
                actual_value=post_avg,
                tolerance=1e-6,
            )
        )
        assert abs(post_avg - safe_degraded) < 1e-6, "Should NOT rollback when above threshold"

    def test_last_snapshot_avg_reset_after_rollback(self, aw_snap10):
        """After rollback, _last_snapshot_avg must equal the snapshot's avg_quality."""
        aw = aw_snap10
        model, task = "gpt-4o", "vision"
        inject_signals(aw, model, task, 12, mean=0.90)

        snap_avg = aw._snapshots[-1]["avg_quality"]
        pre_last_snap = aw._last_snapshot_avg

        for k in aw._state:
            aw._state[k]["avg_quality"] = 0.01
        aw._check_for_corruption()

        m = _record_metric(
            TestMetrics(
                test_name="_last_snapshot_avg Reset to Snapshot After Rollback",
                initial_state={
                    "snap_avg": round(snap_avg, 4),
                    "pre_rollback_last_snap": round(pre_last_snap, 4),
                },
                action_description="Corrupt state, trigger rollback",
                final_state={"post_rollback_last_snapshot_avg": round(aw._last_snapshot_avg, 4)},
                expected_value=snap_avg,
                actual_value=aw._last_snapshot_avg,
                tolerance=1e-9,
            )
        )
        assert aw._last_snapshot_avg == snap_avg


# ── Part 5: Snapshot Integrity Tests ─────────────────────────────────────────


class TestSnapshotIntegrity:
    """Snapshots must contain both _state and _signal_stats, and be capped correctly."""

    def test_snapshot_contains_state_and_signal_stats(self, aw_snap10):
        aw = aw_snap10
        inject_signals(aw, "gpt-4o", "reasoning", 12, mean=0.85)

        assert len(aw._snapshots) >= 1
        snap = aw._snapshots[-1]

        print(f"\n  Snapshot keys: {list(snap.keys())}")
        assert "state" in snap, "Snapshot must contain 'state'"
        assert "signal_stats" in snap, "Snapshot must contain 'signal_stats'"
        assert "avg_quality" in snap, "Snapshot must contain 'avg_quality'"

        # Deep copies — mutations to live state must not affect snapshot
        snap_state_id = id(snap["state"])
        snap_stats_id = id(snap["signal_stats"])
        assert snap_state_id != id(aw._state), "snapshot state must be a deep copy"
        assert snap_stats_id != id(aw._signal_stats), "snapshot signal_stats must be a deep copy"

    def test_snapshot_capped_at_max_snapshots(self, monkeypatch):
        """Taking >_MAX_SNAPSHOTS snapshots keeps only the most recent _MAX_SNAPSHOTS."""
        monkeypatch.setattr(aw_mod, "_SNAPSHOT_INTERVAL", 5)
        aw = AdaptiveWeights(state_file=None)

        # Take _MAX_SNAPSHOTS + 3 snapshots
        total_signals = (_MAX_SNAPSHOTS + 3) * 5
        inject_signals(aw, "m", "t", total_signals, mean=0.80)

        m = _record_metric(
            TestMetrics(
                test_name=f"Snapshot Cap at _MAX_SNAPSHOTS={_MAX_SNAPSHOTS}",
                initial_state={"expected_max": _MAX_SNAPSHOTS, "signals_sent": total_signals},
                action_description=f"Take {_MAX_SNAPSHOTS + 3} snapshots",
                final_state={"actual_snapshot_count": len(aw._snapshots)},
                expected_value=_MAX_SNAPSHOTS,
                actual_value=len(aw._snapshots),
                tolerance=0,
            )
        )
        assert len(aw._snapshots) == _MAX_SNAPSHOTS, (
            f"Expected {_MAX_SNAPSHOTS} snapshots, got {len(aw._snapshots)}"
        )

    def test_snapshot_signal_stats_is_independent_copy(self, aw_snap10):
        """Modifying signal_stats after snapshot must not alter the stored snapshot."""
        aw = aw_snap10
        model, task = "gpt-4o", "classification"
        key = f"{model}:{task}"

        inject_signals(aw, model, task, 12, mean=0.80)
        snap_mean_before_mutation = aw._snapshots[-1]["signal_stats"][key]["mean"]

        # Mutate live signal_stats
        aw._signal_stats[key]["mean"] = 0.01

        snap_mean_after_mutation = aw._snapshots[-1]["signal_stats"][key]["mean"]

        print_comparison_table(
            [
                (
                    "snapshot mean (before mutation)",
                    snap_mean_before_mutation,
                    snap_mean_after_mutation,
                ),
                ("live signal_stats mean", snap_mean_before_mutation, 0.01),
            ]
        )

        assert abs(snap_mean_before_mutation - snap_mean_after_mutation) < 1e-9, (
            "Snapshot signal_stats must not be aliased to live signal_stats"
        )

    def test_snapshot_avg_quality_matches_state(self, aw_snap10):
        """Snapshot's pre-computed avg_quality must match _avg_quality(state)."""
        aw = aw_snap10
        inject_signals(aw, "gpt-4o", "summarization", 12, mean=0.88)
        snap = aw._snapshots[-1]

        computed_avg = avg_quality(snap["state"])
        stored_avg = snap["avg_quality"]

        m = _record_metric(
            TestMetrics(
                test_name="Snapshot avg_quality Matches State Mean",
                initial_state={"signals": 12, "target_mean": 0.88},
                action_description="Take snapshot at signal 10",
                final_state={
                    "stored_avg": round(stored_avg, 6),
                    "computed_avg": round(computed_avg, 6),
                },
                expected_value=computed_avg,
                actual_value=stored_avg,
                tolerance=1e-9,
            )
        )
        assert abs(stored_avg - computed_avg) < 1e-9


# ── Part 6: Integration Test ──────────────────────────────────────────────────


class TestEndToEndIntegration:
    """
    Full lifecycle: 100 good samples → 20 outliers → corrupt → rollback → 50 new samples.
    Prints a timeline: [initial] → [corrupted] → [rollback] → [recovered]
    """

    def test_full_lifecycle_timeline(self, monkeypatch):
        monkeypatch.setattr(aw_mod, "_SNAPSHOT_INTERVAL", 20)
        aw = AdaptiveWeights(state_file=None)
        model, task = "gpt-4o", "reasoning"
        key = f"{model}:{task}"

        # ── Phase 1: Record 100 quality samples with known distribution ──────
        print("\n\n" + "=" * 70)
        print("INTEGRATION TEST: Full Lifecycle Timeline")
        print("=" * 70)

        good_vals = inject_signals(aw, model, task, 100, mean=0.85, std=0.02, seed=1)
        phase1_count = aw._state[key]["sample_count"]
        phase1_avg = aw._state[key]["avg_quality"]
        phase1_mean = aw._signal_stats[key]["mean"]
        snap_count_1 = len(aw._snapshots)

        print("\n[INITIAL] 100 good samples recorded")
        print(f"  sample_count:       {phase1_count}")
        print(f"  avg_quality:        {phase1_avg:.4f}")
        print(f"  signal_stats.mean:  {phase1_mean:.4f}")
        print(f"  snapshots taken:    {snap_count_1}")

        assert snap_count_1 >= 1, "Should have at least 1 snapshot after 100 signals (interval=20)"

        # ── Phase 2: Record 20 outliers (outside 2σ) ─────────────────────────
        outliers_rejected = 0
        for _ in range(20):
            cnt_pre = aw._state[key]["sample_count"]
            aw.record(model, task, 0.10)
            if aw._state[key]["sample_count"] == cnt_pre:
                outliers_rejected += 1

        phase2_count = aw._state[key]["sample_count"]
        phase2_avg = aw._state[key]["avg_quality"]
        phase2_mean = aw._signal_stats[key]["mean"]

        print("\n[AFTER OUTLIERS] Tried 20 outliers at 0.10")
        print(f"  outliers rejected:  {outliers_rejected}/20")
        print(f"  sample_count:       {phase2_count}  (delta: +{phase2_count - phase1_count})")
        print(f"  avg_quality:        {phase2_avg:.4f}")
        print(f"  signal_stats.mean:  {phase2_mean:.4f}")

        assert outliers_rejected >= 15, (
            f"Expected ≥15/20 outliers rejected, got {outliers_rejected}"
        )

        # ── Phase 3: Corrupt state to simulate degradation ───────────────────
        snap_state = copy.deepcopy(aw._snapshots[-1]["state"])
        snap_stats = copy.deepcopy(aw._snapshots[-1]["signal_stats"])
        snap_avg = aw._snapshots[-1]["avg_quality"]

        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.05
        phase3_avg = avg_quality(aw._state)

        print("\n[CORRUPTED] Artificially set all avg_quality = 0.05")
        print(f"  current avg:        {phase3_avg:.4f}")
        print(f"  snapshot avg:       {snap_avg:.4f}")
        print(f"  threshold:          {snap_avg * _ROLLBACK_DROP_THRESHOLD:.4f}")

        # ── Phase 4: Trigger rollback ─────────────────────────────────────────
        aw._check_for_corruption()
        phase4_avg = avg_quality(aw._state)
        phase4_count = aw._state[key]["sample_count"]
        phase4_mean = aw._signal_stats[key]["mean"]

        print("\n[ROLLBACK] _check_for_corruption() called")
        print(f"  restored avg:       {phase4_avg:.4f}")
        print(f"  restored count:     {phase4_count}")
        print(f"  restored stat mean: {phase4_mean:.4f}")
        print(f"  last_snapshot_avg:  {aw._last_snapshot_avg:.4f}")

        assert phase4_avg > phase3_avg, "Rollback must restore avg above corrupted level"
        assert abs(phase4_mean - snap_stats[key]["mean"]) < 0.01, (
            "signal_stats.mean must be restored from snapshot"
        )

        # ── Phase 5: Record 50 new samples and verify correct processing ──────
        new_vals = inject_signals(aw, model, task, 50, mean=0.86, std=0.02, seed=99)
        phase5_count = aw._state[key]["sample_count"]
        phase5_avg = aw._state[key]["avg_quality"]

        new_accepted = phase5_count - phase4_count
        print("\n[RECOVERED] 50 new samples at mean=0.86")
        print(f"  new signals accepted: {new_accepted}/50")
        print(f"  sample_count:         {phase5_count}")
        print(f"  avg_quality:          {phase5_avg:.4f}")

        assert new_accepted > 40, f"Expected most of 50 new signals accepted, got {new_accepted}"
        assert phase5_avg >= _QUALITY_FLOOR, "avg_quality must be above floor"

        # ── Timeline summary ──────────────────────────────────────────────────
        print("\n" + "─" * 70)
        print("TIMELINE SUMMARY")
        print("─" * 70)
        stages = [
            ("INITIAL", phase1_count, f"{phase1_avg:.4f}", f"{phase1_mean:.4f}"),
            ("CORRUPTED", phase2_count, f"{phase3_avg:.4f}", f"{phase2_mean:.4f}"),
            ("ROLLBACK", phase4_count, f"{phase4_avg:.4f}", f"{phase4_mean:.4f}"),
            (
                "RECOVERED",
                phase5_count,
                f"{phase5_avg:.4f}",
                f"{aw._signal_stats[key]['mean']:.4f}",
            ),
        ]
        print(f"  {'Stage':<12} {'Samples':>8} {'avg_quality':>12} {'stats.mean':>12}")
        print(f"  {'-' * 12} {'-' * 8} {'-' * 12} {'-' * 12}")
        for stage, cnt, aqv, smean in stages:
            print(f"  {stage:<12} {cnt:>8} {aqv:>12} {smean:>12}")

        _record_metric(
            TestMetrics(
                test_name="Integration: Full Lifecycle (100 good → 20 outliers → corrupt → rollback → 50 new)",
                initial_state={"phase1_avg": round(phase1_avg, 4), "phase1_count": phase1_count},
                action_description="Outlier rejection → corruption → rollback → recovery",
                final_state={
                    "phase5_avg": round(phase5_avg, 4),
                    "phase5_count": phase5_count,
                    "outliers_rejected": outliers_rejected,
                    "new_accepted": new_accepted,
                },
                expected_value=0.80,
                actual_value=phase5_avg,
                tolerance=0.20,
            )
        )


# ── Summary Reporter ──────────────────────────────────────────────────────────


def pytest_sessionfinish(session, exitstatus):
    """Print a final summary table of all recorded TestMetrics."""
    if not _ALL_METRICS:
        return
    print("\n\n" + "=" * 90)
    print("ADAPTIVE WEIGHTS TEST METRICS SUMMARY")
    print("=" * 90)
    col = 52
    print(f"  {'Test Name':<{col}} {'Expected':>9} {'Actual':>9} {'Tol':>6} {'Result':>6}")
    print(f"  {'-' * col} {'-' * 9} {'-' * 9} {'-' * 6} {'-' * 6}")
    passed = failed = 0
    for m in _ALL_METRICS:
        status = "PASS" if m.passed else "FAIL"
        if m.passed:
            passed += 1
        else:
            failed += 1
        name = m.test_name[:col]
        print(
            f"  {name:<{col}} {m.expected_value:>9.4f} {m.actual_value:>9.4f}"
            f" {m.tolerance:>6.4f} {'✅' if m.passed else '❌':>6}"
        )
    print(f"\n  Total: {len(_ALL_METRICS)}  Passed: {passed}  Failed: {failed}")
    print("=" * 90)
