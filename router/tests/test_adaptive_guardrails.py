"""
File: router/tests/test_adaptive_guardrails.py

Purpose:
Tests for the six guardrail improvements made to AdaptiveWeights.  Each test
class maps to one issue area.

How to run:
  pytest -v router/tests/test_adaptive_guardrails.py
  pytest -v router/tests/test_adaptive_guardrails.py::TestDriftDetection

How to add a test:
  1. Identify the issue area (drift, decay, clamping, threshold, log wrap, migration).
  2. Add to the corresponding class, or create a new class for a new guardrail.
  3. Follow the Arrange / Act / Assert pattern (see examples below).

Example:
  def test_my_guardrail_behaviour():
      # Arrange
      aw = _aw()
      for _ in range(1000):
          aw.record("model", "task", 0.90, 0.80)
      # Act
      aw._state["model:task"]["avg_quality"] = 0.01  # inject corruption
      aw._check_for_corruption()
      # Assert
      assert aw._state["model:task"]["avg_quality"] > 0.01

Test classes:
  TestOutlierFiltering      — outlier signals (>2σ) are rejected
  TestMinimumSampleThreshold — adaptive score inactive until ADAPTIVE_MIN_SAMPLES
  TestQualityFloor          — avg_quality never falls below _QUALITY_FLOOR
  TestSnapshotRollback      — corruption rollback restores state + signal_stats
  TestDriftDetection        — Issue 1: gradual drift triggers rollback between snapshots
  TestCustomDecayFactor     — Issue 2: set_decay_factor() overrides per-key EMA decay
  TestQualityScoreClamping  — Issue 3: out-of-range scores are clamped, not dropped
  TestPerCustomerThreshold  — Issue 4: PER_CUSTOMER_MIN_SAMPLES = 10
  TestCustomerLogWrap       — Issue 5: customer log cap + debug log on wrap
  TestDataMigration         — Issue 6: v0/v1/v2 on-disk format migration
"""

from __future__ import annotations

import json
import time

import pytest

import router.adaptive_weights as aw_mod
from router.adaptive_weights import (
    _FORMAT_VERSION,
    _QUALITY_FLOOR,
    _ROLLBACK_DROP_THRESHOLD,
    _WRITE_INTERVAL,
    AdaptiveWeights,
)
from router.config import ADAPTIVE_CUSTOMER_LOG_MAX, PER_CUSTOMER_MIN_SAMPLES


def _aw() -> AdaptiveWeights:
    return AdaptiveWeights(state_file=None)


class TestOutlierFiltering:
    def test_outlier_rejected(self):
        aw = _aw()
        # Seed 15 signals with natural variance around 0.85 so std > 0.01
        import random

        random.seed(42)
        for v in [
            0.80,
            0.82,
            0.83,
            0.84,
            0.85,
            0.85,
            0.86,
            0.86,
            0.87,
            0.87,
            0.88,
            0.88,
            0.89,
            0.90,
            0.91,
        ]:
            aw.record("gpt-4o", "code_generation", v)
        key = "gpt-4o:code_generation"
        count_before = aw._state[key]["sample_count"]
        # Inject extreme outlier — 0.0 is far beyond 2σ from mean ≈ 0.86
        aw.record("gpt-4o", "code_generation", 0.0)
        count_after = aw._state[key]["sample_count"]
        # Outlier should be rejected — sample count unchanged
        assert count_after == count_before

    def test_normal_signal_accepted(self):
        aw = _aw()
        for v in [
            0.80,
            0.82,
            0.83,
            0.84,
            0.85,
            0.85,
            0.86,
            0.86,
            0.87,
            0.87,
            0.88,
            0.88,
            0.89,
            0.90,
            0.91,
        ]:
            aw.record("gpt-4o", "code_generation", v)
        key = "gpt-4o:code_generation"
        count_before = aw._state[key]["sample_count"]
        # Signal within 2σ of mean — should be accepted
        aw.record("gpt-4o", "code_generation", 0.83)
        count_after = aw._state[key]["sample_count"]
        assert count_after == count_before + 1


class TestMinimumSampleThreshold:
    def test_minimum_sample_threshold(self):
        aw = _aw()
        base = 0.75
        # Record 19 signals — just below the 20-sample threshold
        for _ in range(19):
            aw.record("claude-sonnet-4-20250514", "reasoning", 0.95, base)
        # Should still use base score (< 20 samples)
        score = aw.get_adjusted_score("claude-sonnet-4-20250514", "reasoning", base)
        assert score == base
        # One more record → 20 samples, adaptive score should now kick in
        aw.record("claude-sonnet-4-20250514", "reasoning", 0.95, base)
        score_after = aw.get_adjusted_score("claude-sonnet-4-20250514", "reasoning", base)
        assert score_after > base  # learned adjustment applied


class TestQualityFloor:
    def test_quality_floor(self):
        aw = _aw()
        base = 0.80
        # Feed 25 catastrophically bad scores
        for _ in range(25):
            aw.record("gpt-4o-mini", "code_generation", 0.01, base)
        # Adaptive score must never fall below _QUALITY_FLOOR
        score = aw.get_adjusted_score("gpt-4o-mini", "code_generation", base)
        assert score >= _QUALITY_FLOOR

    def test_quality_floor_in_ema(self):
        aw = _aw()
        # Even the internal EMA avg_quality must not go below floor
        for _ in range(30):
            aw.record("gpt-4o-mini", "reasoning", 0.0)
        key = "gpt-4o-mini:reasoning"
        assert aw._state[key]["avg_quality"] >= _QUALITY_FLOOR


class TestSnapshotRollback:
    def test_snapshot_rollback(self):
        aw = _aw()
        base = 0.80
        # Seed good data so we have a valid snapshot to roll back to
        for _ in range(1000):
            aw.record("gpt-4o", "reasoning", 0.90, base)

        snapshot_count = len(aw._snapshots)
        assert snapshot_count >= 1

        # Capture the avg quality before corruption
        def avg_q(state):
            vals = [v.get("avg_quality", 0.5) for v in state.values()]
            return sum(vals) / len(vals) if vals else 0.5

        pre_corrupt_avg = avg_q(aw._state)

        # Inject corrupted state directly (simulate data corruption)
        for key in list(aw._state.keys()):
            aw._state[key]["avg_quality"] = 0.10

        # Trigger corruption check
        aw._check_for_corruption()

        # Should have rolled back to snapshot with higher avg_quality
        post_rollback_avg = avg_q(aw._state)
        assert post_rollback_avg > 0.10

    def test_rollback_restores_signal_stats(self):
        aw = _aw()
        base = 0.80
        key = "gpt-4o:reasoning"
        # Seed 1000 signals to trigger a snapshot
        for _ in range(1000):
            aw.record("gpt-4o", "reasoning", 0.90, base)

        assert len(aw._snapshots) >= 1
        # Snapshot must include signal_stats
        assert "signal_stats" in aw._snapshots[-1]
        snap_stats = aw._snapshots[-1]["signal_stats"].get(key)
        assert snap_stats is not None

        # Record the pre-corruption stats mean
        original_mean = snap_stats["mean"]

        # Poison signal_stats directly (simulates a bad burst corrupting outlier detection)
        aw._signal_stats[key]["mean"] = 0.01
        aw._signal_stats[key]["variance"] = 999.0

        # Also corrupt state to trigger rollback threshold
        for k in list(aw._state.keys()):
            aw._state[k]["avg_quality"] = 0.10

        aw._check_for_corruption()

        # signal_stats must be restored — mean should be back near original
        restored_mean = aw._signal_stats[key]["mean"]
        assert abs(restored_mean - original_mean) < 0.05, (
            f"signal_stats not restored: got mean={restored_mean}, expected ~{original_mean}"
        )
        # Variance must not be the poisoned value
        assert aw._signal_stats[key]["variance"] < 10.0


# ── Issue #1: Slow drift detection ──────────────────────────────────────────


class TestDriftDetection:
    def test_gradual_drift_triggers_rollback_between_snapshots(self):
        """Quality that degrades slowly (below snapshot_avg × 0.9) must roll back
        on any record() call, not only when a new snapshot is taken."""
        aw = _aw()
        for _ in range(1000):
            aw.record("gpt-4o", "code_review", 0.90, 0.80)

        baseline = aw._last_snapshot_avg
        assert baseline is not None and baseline > 0.70

        # Degrade state to just below the rollback threshold — simulate slow drift
        # across many signals rather than an instant collapse
        degraded = baseline * _ROLLBACK_DROP_THRESHOLD - 0.01
        for k in aw._state:
            aw._state[k]["avg_quality"] = degraded

        # Must fire on the very next check, not wait for the next 1000-signal snapshot
        aw._check_for_corruption()

        restored_avg = AdaptiveWeights._avg_quality(aw._state)
        assert restored_avg > degraded, "rollback did not restore state above degraded level"

    def test_last_snapshot_avg_reset_after_rollback(self):
        """After a rollback _last_snapshot_avg must equal the snapshot's avg_quality
        so subsequent signals are compared against the clean baseline."""
        aw = _aw()
        for _ in range(1000):
            aw.record("gpt-4o", "code_review", 0.90, 0.80)

        snap_avg = aw._snapshots[-1]["avg_quality"]

        for k in aw._state:
            aw._state[k]["avg_quality"] = 0.10
        aw._check_for_corruption()

        assert aw._last_snapshot_avg == snap_avg

    def test_no_rollback_before_first_snapshot(self):
        """_check_for_corruption must be a no-op until at least one snapshot exists."""
        aw = _aw()
        aw.record("gpt-4o", "code_review", 0.90)
        # Corrupt state before any snapshot
        aw._state["gpt-4o:code_review"]["avg_quality"] = 0.01
        aw._check_for_corruption()
        # State should remain corrupted — no snapshot to roll back to
        assert aw._state["gpt-4o:code_review"]["avg_quality"] == 0.01


# ── Issue #2: Configurable decay factor ─────────────────────────────────────


class TestCustomDecayFactor:
    def test_faster_decay_converges_quicker(self):
        """A key with decay=0.50 should reach a new quality target faster than
        the default decay=0.95 after the same number of fresh signals."""
        base = 0.50

        # Warm up both instances to quality ~0.50
        aw_fast = _aw()
        aw_slow = _aw()
        for _ in range(30):
            aw_fast.record("m", "t", 0.50, base)
            aw_slow.record("m", "t", 0.50, base)

        # Override decay only on the fast instance
        aw_fast.set_decay_factor("m", "t", 0.50)

        # Inject high-quality signals — fast decay should embrace them sooner
        for _ in range(10):
            aw_fast.record("m", "t", 0.90, base)
            aw_slow.record("m", "t", 0.90, base)

        fast_avg = aw_fast._state["m:t"]["avg_quality"]
        slow_avg = aw_slow._state["m:t"]["avg_quality"]
        assert fast_avg > slow_avg, (
            f"fast decay ({fast_avg:.4f}) should exceed slow decay ({slow_avg:.4f})"
        )

    def test_set_decay_factor_stored_per_key(self):
        aw = _aw()
        aw.set_decay_factor("model-a", "vision", 0.70)
        aw.set_decay_factor("model-b", "vision", 0.80)
        assert aw._decay_overrides["model-a:vision"] == 0.70
        assert aw._decay_overrides["model-b:vision"] == 0.80
        # Unrelated key is absent — will fall back to ADAPTIVE_DECAY_FACTOR
        assert "model-a:reasoning" not in aw._decay_overrides

    def test_set_decay_factor_rejects_invalid_range(self):
        aw = _aw()
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                aw.set_decay_factor("m", "t", bad)

    def test_decay_override_used_in_record(self):
        """Verify the override is actually picked up by record(), not just stored."""
        aw = _aw()
        aw.set_decay_factor("m", "t", 0.10)  # extreme: almost no history weight
        aw.record("m", "t", 0.90, 0.50)
        # With decay=0.10 starting from base 0.50: avg ≈ 0.10*0.50 + 0.90*0.90 = 0.86
        assert aw._state["m:t"]["avg_quality"] > 0.80


# ── Issue #3: Quality score validation ──────────────────────────────────────


class TestQualityScoreClamping:
    def test_score_above_1_is_clamped_and_accepted(self):
        """Scores > 1.0 must be clamped to 1.0 and recorded, not silently dropped."""
        aw = _aw()
        aw.record("gpt-4o", "summarization", 1.5)
        assert "gpt-4o:summarization" in aw._state
        assert aw._state["gpt-4o:summarization"]["avg_quality"] <= 1.0

    def test_score_below_0_is_clamped_and_accepted(self):
        aw = _aw()
        aw.record("gpt-4o", "summarization", -0.3)
        assert "gpt-4o:summarization" in aw._state
        assert aw._state["gpt-4o:summarization"]["avg_quality"] >= _QUALITY_FLOOR

    def test_boundary_scores_accepted_without_clamping(self):
        """Exactly 0.0 and 1.0 are valid — must not be clamped or rejected."""
        aw = _aw()
        aw.record("m", "t", 0.0)
        aw.record("m", "t", 1.0)
        assert aw._state["m:t"]["sample_count"] == 2

    def test_clamped_score_emits_debug_log(self, monkeypatch):
        debug_events: list[str] = []
        monkeypatch.setattr(aw_mod.log, "debug", lambda ev, **_: debug_events.append(ev))
        aw = AdaptiveWeights(state_file=None)
        aw.record("m", "t", 2.0)
        assert "adaptive_quality_clamped" in debug_events


# ── Issue #4: Per-customer threshold ────────────────────────────────────────


class TestPerCustomerThreshold:
    def test_threshold_is_10(self):
        assert PER_CUSTOMER_MIN_SAMPLES == 10

    def test_custom_weights_activate_at_10_samples(self):
        """Per-customer EMA must activate exactly at 10 samples, not before."""
        aw = _aw()
        base, cid = 0.70, "cust-alpha"
        for _ in range(9):
            aw.record("gpt-4o-mini", "classification", 0.92, base, customer_id=cid)

        # 9 samples — per-customer path skipped; global also has 9 < ADAPTIVE_MIN_SAMPLES
        score = aw.get_adjusted_score("gpt-4o-mini", "classification", base, customer_id=cid)
        assert score == base

        # 10th sample → per-customer path activates
        aw.record("gpt-4o-mini", "classification", 0.92, base, customer_id=cid)
        score = aw.get_adjusted_score("gpt-4o-mini", "classification", base, customer_id=cid)
        assert score > base


# ── Issue #5: Customer log wrap detection ───────────────────────────────────


class TestCustomerLogWrap:
    def test_log_length_capped_at_maxlen(self):
        aw = _aw()
        cid = "cust-fill"
        for i in range(ADAPTIVE_CUSTOMER_LOG_MAX + 10):
            aw.record_routing_event(cid, "gpt-4o", "summarization", 0.01 * i)
        assert len(aw._customer_log[cid]) == ADAPTIVE_CUSTOMER_LOG_MAX

    def test_wrap_emits_debug_log(self, monkeypatch):
        debug_events: list[str] = []
        monkeypatch.setattr(aw_mod.log, "debug", lambda ev, **_: debug_events.append(ev))
        aw = AdaptiveWeights(state_file=None)
        cid = "cust-wrap"
        # Fill exactly to capacity, then one more to trigger the wrap path
        for _ in range(ADAPTIVE_CUSTOMER_LOG_MAX):
            aw.record_routing_event(cid, "gpt-4o", "reasoning", 0.01)
        aw.record_routing_event(cid, "gpt-4o", "reasoning", 0.01)
        assert "adaptive_customer_log_wrapped" in debug_events

    def test_no_wrap_log_below_capacity(self, monkeypatch):
        debug_events: list[str] = []
        monkeypatch.setattr(aw_mod.log, "debug", lambda ev, **_: debug_events.append(ev))
        aw = AdaptiveWeights(state_file=None)
        cid = "cust-partial"
        for _ in range(ADAPTIVE_CUSTOMER_LOG_MAX - 1):
            aw.record_routing_event(cid, "gpt-4o", "reasoning", 0.01)
        assert "adaptive_customer_log_wrapped" not in debug_events


# ── Issue #6: Data migration ─────────────────────────────────────────────────


class TestDataMigration:
    def test_v0_flat_format_migrated(self, tmp_path):
        """v0 flat-dict format must be loaded and flagged for immediate flush."""
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "gpt-4o:reasoning": {
                        "avg_quality": 0.80,
                        "adjustment": 0.30,
                        "sample_count": 25,
                    }
                }
            )
        )
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        assert "gpt-4o:reasoning" in aw._state
        # Migration must schedule an immediate flush
        assert aw._dirty >= _WRITE_INTERVAL

    def test_v1_format_migrated(self, tmp_path):
        """v1 format (global/customers present, no 'version' key) must trigger migration."""
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "global": {"gpt-4o:reasoning": {"avg_quality": 0.80, "sample_count": 5}},
                    "customers": {},
                }
            )
        )
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        assert "gpt-4o:reasoning" in aw._state
        assert aw._dirty >= _WRITE_INTERVAL

    def test_v2_format_no_migration(self, tmp_path):
        """Current v2 format must load cleanly without triggering a migration flush."""
        state_file = tmp_path / "state.json"
        state_file.write_text(
            json.dumps(
                {
                    "version": _FORMAT_VERSION,
                    "global": {"gpt-4o:reasoning": {"avg_quality": 0.80, "sample_count": 5}},
                    "customers": {},
                }
            )
        )
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        assert "gpt-4o:reasoning" in aw._state
        assert aw._dirty == 0  # no migration needed — file is already current format

    def test_flush_writes_version_marker(self, tmp_path):
        """_flush must embed the version key so future loads know the format."""
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        aw.record("gpt-4o", "reasoning", 0.80)
        aw.flush()
        data = json.loads(state_file.read_text())
        assert data.get("version") == _FORMAT_VERSION


class TestFlushLockContention:
    """Regression: record()'s periodic flush used to hold self._lock for the
    entire atomic_write_json() call (fsync + rename), blocking every other
    record()/get_adjusted_score() caller for the duration of a disk write
    that scales with state size. The write must happen after self._lock is
    released."""

    def test_periodic_flush_does_not_hold_state_lock_during_io(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)

        lock_held_during_write = []
        real_write = aw._write_to_disk

        def spying_write(data):
            lock_held_during_write.append(aw._lock.locked())
            real_write(data)

        monkeypatch.setattr(aw, "_write_to_disk", spying_write)

        for _ in range(_WRITE_INTERVAL):
            aw.record("gpt-4o", "reasoning", 0.8)

        assert lock_held_during_write == [False]

    def test_explicit_flush_does_not_hold_state_lock_during_io(self, tmp_path, monkeypatch):
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        aw.record("gpt-4o", "reasoning", 0.8)

        lock_held_during_write = []
        real_write = aw._write_to_disk

        def spying_write(data):
            lock_held_during_write.append(aw._lock.locked())
            real_write(data)

        monkeypatch.setattr(aw, "_write_to_disk", spying_write)
        aw.flush()

        assert lock_held_during_write == [False]

    def test_record_not_blocked_by_concurrent_disk_write(self, tmp_path):
        """A record() call that only needs self._lock must not wait on
        _io_lock, even while a (simulated slow) write is in progress."""
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)

        aw._io_lock.acquire()
        try:
            t0 = time.perf_counter()
            aw.record("gpt-4o", "reasoning", 0.8)
            aw.get_adjusted_score("gpt-4o", "reasoning", 0.5)
            elapsed = time.perf_counter() - t0
        finally:
            aw._io_lock.release()

        assert elapsed < 0.05, f"record()/get_adjusted_score() blocked for {elapsed:.3f}s"

    def test_flush_still_round_trips_correctly(self, tmp_path):
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        for _ in range(5):
            aw.record("gpt-4o", "reasoning", 0.8, customer_id="cust-1")
        aw.flush()

        reloaded = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        assert reloaded._state["gpt-4o:reasoning"]["sample_count"] == 5
        assert reloaded._customer_state["cust-1"]["gpt-4o:reasoning"]["sample_count"] == 5

    def test_snapshot_is_isolated_from_later_mutation(self, tmp_path):
        """The flush snapshot must be a deep copy — entry dicts are mutated
        in place by _update_ema(), so a shallow copy could let a later
        record() call corrupt data that's still being written to disk."""
        state_file = tmp_path / "state.json"
        aw = AdaptiveWeights(state_file=str(state_file), base_dir=tmp_path)
        aw.record("gpt-4o", "reasoning", 0.8, customer_id="cust-1")

        snapshot = aw._prepare_flush()
        assert snapshot is not None
        before = snapshot["global"]["gpt-4o:reasoning"]["sample_count"]

        aw.record("gpt-4o", "reasoning", 0.9, customer_id="cust-1")

        assert snapshot["global"]["gpt-4o:reasoning"]["sample_count"] == before
