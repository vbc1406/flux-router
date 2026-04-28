"""
File: router/tests/test_cache.py

Purpose:
Tests for ResponseCache (TTL-based response memoisation), the prompt fingerprint
function, and the is_cache_eligible predicate.

How to run:
  pytest -v router/tests/test_cache.py
  pytest -v router/tests/test_cache.py::TestResponseCache

How to add a test:
  1. Find the class for the area you are testing.
  2. Instantiate cache with ResponseCache(ttl_seconds=...) for time-sensitive cases.
  3. Use _model() to build a throwaway ModelOption fixture.

Test classes:
  TestFingerprint        — determinism, sensitivity to prompt/system changes
  TestCacheEligibility   — which request types are skipped (non-deterministic, etc.)
  TestResponseCache      — get/set, TTL expiry, hit-rate counters, disabled mode
"""
from __future__ import annotations

import time

import pytest

from router.cache import ResponseCache, fingerprint, is_cache_eligible
from router.schemas import ModelOption


def _model(mid: str = "test-model") -> ModelOption:
    return ModelOption(
        provider="test", model_id=mid, display_name="Test",
        tier="cheap", cost_per_1k_input=0.001, cost_per_1k_output=0.002,
        max_context_window=4096, max_output_tokens=1000, capabilities=[],
    )


# ── Fingerprinting ────────────────────────────────────────────────────────────

class TestFingerprint:
    def test_deterministic(self):
        fp1 = fingerprint("hello world", None, [], None)
        fp2 = fingerprint("hello world", None, [], None)
        assert fp1 == fp2

    def test_hex_string_length(self):
        fp = fingerprint("hello", None, [], None)
        assert len(fp) == 64  # SHA-256 hex

    def test_temperature_none_vs_zero(self):
        fp_none = fingerprint("hello", None, [], None)
        fp_zero = fingerprint("hello", None, [], 0.0)
        assert fp_none != fp_zero

    def test_temperature_affects_fingerprint(self):
        fp_lo = fingerprint("What is 2+2?", None, [], 0.0)
        fp_hi = fingerprint("What is 2+2?", None, [], 1.0)
        assert fp_lo != fp_hi

    def test_filler_normalised_away(self):
        fp1 = fingerprint("What is 2+2?",         None, [], None)
        fp2 = fingerprint("Please What is 2+2?",  None, [], None)
        fp3 = fingerprint("Can you What is 2+2?", None, [], None)
        # All three normalise to the same core question
        assert fp1 == fp2 == fp3

    def test_case_insensitive(self):
        fp1 = fingerprint("HELLO WORLD", None, [], None)
        fp2 = fingerprint("hello world", None, [], None)
        assert fp1 == fp2

    def test_whitespace_collapse(self):
        fp1 = fingerprint("hello   world",  None, [], None)
        fp2 = fingerprint("hello world",    None, [], None)
        assert fp1 == fp2

    def test_date_removal(self):
        fp1 = fingerprint("What happened today?",       None, [], None)
        fp2 = fingerprint("What happened?",             None, [], None)
        fp3 = fingerprint("What happened currently?",   None, [], None)
        # "today" and "currently" stripped → similar hash as the bare question
        assert fp1 == fp2 == fp3

    def test_system_prompt_included(self):
        fp1 = fingerprint("Hello", "You are a pirate", [],  None)
        fp2 = fingerprint("Hello", "You are helpful",  [],  None)
        fp3 = fingerprint("Hello", None,               [],  None)
        assert fp1 != fp2
        assert fp1 != fp3

    def test_history_last_two_included(self):
        hist_a = [{"role": "user", "content": "prev question A"}]
        hist_b = [{"role": "user", "content": "prev question B"}]
        fp1 = fingerprint("current", None, hist_a, None)
        fp2 = fingerprint("current", None, hist_b, None)
        assert fp1 != fp2

    def test_history_beyond_two_ignored(self):
        """Adding extra old messages beyond 2 should not change the fingerprint."""
        old_history = [
            {"role": "user", "content": "old msg 1"},
            {"role": "user", "content": "old msg 2"},
        ]
        last_two = [
            {"role": "user", "content": "recent A"},
            {"role": "user", "content": "recent B"},
        ]
        fp1 = fingerprint("now", None, last_two, None)
        fp2 = fingerprint("now", None, old_history + last_two, None)
        assert fp1 == fp2


# ── Cache eligibility ─────────────────────────────────────────────────────────

class TestCacheEligibility:
    def test_simple_qa_eligible(self):
        assert is_cache_eligible("simple_qa", None)  is True
        assert is_cache_eligible("simple_qa", 0.0)   is True
        assert is_cache_eligible("simple_qa", 0.3)   is True

    def test_simple_qa_high_temp_not_eligible(self):
        assert is_cache_eligible("simple_qa", 0.31) is False
        assert is_cache_eligible("simple_qa", 1.0)  is False

    def test_translation_eligible(self):
        assert is_cache_eligible("translation", None) is True

    def test_classification_eligible(self):
        assert is_cache_eligible("classification", 0.2) is True

    def test_extraction_eligible(self):
        assert is_cache_eligible("extraction", None) is True

    def test_summarization_eligible(self):
        assert is_cache_eligible("summarization", None) is True

    def test_creative_not_eligible(self):
        assert is_cache_eligible("creative_writing", 0.0)  is False
        assert is_cache_eligible("creative_writing", None) is False

    def test_conversation_not_eligible(self):
        assert is_cache_eligible("conversation", None) is False

    def test_vision_not_eligible(self):
        assert is_cache_eligible("vision", None) is False

    def test_code_gen_not_eligible(self):
        # code_generation not in CACHEABLE_TASK_TYPES
        assert is_cache_eligible("code_generation", 0.0) is False

    def test_analysis_not_eligible(self):
        assert is_cache_eligible("analysis", 0.0) is False


# ── ResponseCache ─────────────────────────────────────────────────────────────

class TestResponseCache:
    def setup_method(self):
        self.cache = ResponseCache(max_entries=5, ttl_seconds=60, enabled=True)
        self.model = _model()

    def test_miss_on_empty(self):
        assert self.cache.get("nonexistent") is None

    def test_set_and_get(self):
        self.cache.set("fp1", "Hello!", self.model, 0.001)
        result = self.cache.get("fp1")
        assert result is not None
        assert result.response_text == "Hello!"

    def test_model_preserved(self):
        self.cache.set("fp2", "response", self.model, 0.002)
        result = self.cache.get("fp2")
        assert result.model_used.model_id == self.model.model_id

    def test_cost_preserved(self):
        self.cache.set("fp3", "resp", self.model, 0.012345)
        result = self.cache.get("fp3")
        assert result.original_cost == pytest.approx(0.012345)

    def test_ttl_expiry(self):
        cache = ResponseCache(ttl_seconds=1, enabled=True)
        cache.set("fp_ttl", "data", self.model, 0.001)
        assert cache.get("fp_ttl") is not None
        time.sleep(1.1)
        assert cache.get("fp_ttl") is None

    def test_lru_eviction(self):
        cache = ResponseCache(max_entries=3, ttl_seconds=3600, enabled=True)
        for i in range(4):
            cache.set(f"fp{i}", f"resp{i}", self.model, 0.001)
        # First entry should have been evicted
        assert cache.get("fp0") is None
        # Last entry should still be present
        assert cache.get("fp3") is not None

    def test_invalidate(self):
        self.cache.set("fp_inv", "data", self.model, 0.001)
        assert self.cache.get("fp_inv") is not None
        assert self.cache.invalidate("fp_inv") is True
        assert self.cache.get("fp_inv") is None

    def test_invalidate_missing_returns_false(self):
        assert self.cache.invalidate("does_not_exist") is False

    def test_stats_hit_rate(self):
        self.cache.set("fp_s", "data", self.model, 0.001)
        self.cache.get("fp_s")   # hit
        self.cache.get("fp_s")   # hit
        self.cache.get("missing") # miss
        stats = self.cache.stats()
        assert stats["total_hits"]   == 2
        assert stats["total_misses"] == 1
        assert abs(stats["hit_rate"] - 2/3) < 0.01

    def test_disabled_cache_always_misses(self):
        cache = ResponseCache(enabled=False)
        cache.set("fp", "data", self.model, 0.001)
        assert cache.get("fp") is None

    def test_stats_entries_count(self):
        for i in range(3):
            self.cache.set(f"fp_cnt_{i}", f"r{i}", self.model, 0.001)
        assert self.cache.stats()["entries_count"] >= 3
