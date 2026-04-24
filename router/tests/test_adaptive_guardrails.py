"""Tests for adaptive learning guardrails (Fix 3)."""
from __future__ import annotations

import pytest

from router.adaptive_weights import AdaptiveWeights, _QUALITY_FLOOR


def _aw() -> AdaptiveWeights:
    return AdaptiveWeights(state_file=None)


class TestOutlierFiltering:
    def test_outlier_rejected(self):
        aw = _aw()
        # Seed 15 signals with natural variance around 0.85 so std > 0.01
        import random
        random.seed(42)
        for v in [0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.90, 0.91]:
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
        for v in [0.80, 0.82, 0.83, 0.84, 0.85, 0.85, 0.86, 0.86, 0.87, 0.87, 0.88, 0.88, 0.89, 0.90, 0.91]:
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
