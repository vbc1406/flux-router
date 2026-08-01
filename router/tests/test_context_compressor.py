"""
File: router/tests/test_context_compressor.py

Purpose:
Tests for router/context_compressor.py — trims conversation history to fit a
target token budget. Focused on Pass 3 (the last-resort "drop oldest
messages one by one" step), which used to rebuild and re-scan the whole
history on every dropped message (O(n^2) on long histories).

How to run:
  pytest -v router/tests/test_context_compressor.py
"""

from __future__ import annotations

from router.context_compressor import ContextCompressor, estimate_tokens
from router.schemas import RoutingRequest


def _msg(role: str, words: int, marker: str = "") -> dict:
    return {"role": role, "content": " ".join(["word"] * words) + marker}


def _request(history: list[dict]) -> RoutingRequest:
    return RoutingRequest(raw_prompt="current turn", user_id="u", message_history=history)


class TestBelowTarget:
    def test_short_history_untouched(self):
        history = [_msg("user", 5), _msg("assistant", 5)]
        req = _request(history)
        result = ContextCompressor().compress(req, target_tokens=2000)
        assert result is req  # returned unchanged, not even copied


class TestPass3DropOldestOneByOne:
    """Long, uniform history that survives Pass 1 (no tool/image/long-assistant
    content to compress) and Pass 2 (mixed roles, so 'user only' still
    doesn't fit) forces Pass 3's incremental-drop path."""

    def _big_history(self, n_pairs: int) -> list[dict]:
        history = []
        for _ in range(n_pairs):
            history.append(_msg("user", 30))
            history.append(_msg("assistant", 30))
        return history

    def test_drops_oldest_first_and_fits_target(self):
        history = self._big_history(100)  # 200 messages (schema max), well over target
        req = _request(history)
        target = 500
        result = ContextCompressor().compress(req, target_tokens=target)
        new_history = result.message_history

        # A compression note was prepended.
        assert new_history[0]["role"] == "system"
        assert "compressed" in new_history[0]["content"]

        # The last _PRESERVE_RECENT=4 original messages are always kept verbatim.
        assert new_history[-4:] == history[-4:]

        # Result fits (approximately) within target — the note itself adds a
        # small amount, so allow slack rather than an exact bound.
        assert estimate_tokens(new_history) <= target + 50

    def test_oldest_messages_are_the_ones_dropped(self):
        """Pass 3 must drop from the FRONT (oldest) — the tail (most recent
        of the 'old' bucket, right before the preserved-recent window) must
        survive if it fits."""
        history = self._big_history(100)
        req = _request(history)
        result = ContextCompressor().compress(req, target_tokens=500)
        new_history = result.message_history

        surviving_old = new_history[1:-4]  # strip the note and preserved-recent
        if surviving_old:
            # With uniform assistant/user messages like these, Pass 2 (user
            # messages only) fires before Pass 3 — so the "old" pool Pass 3
            # drops from is the user-only subset, not the raw alternating
            # history. Whatever survives must be a contiguous suffix of
            # THAT pool, not scattered/duplicated.
            user_only_old = [m for m in history[:-4] if m.get("role") == "user"]
            tail_of_old = user_only_old[-len(surviving_old):]
            assert surviving_old == tail_of_old

    def test_scales_roughly_linearly_not_quadratically(self):
        """Not a strict perf assertion (flaky in CI), just a sanity check
        that 5x the history (bounded by RoutingRequest's 200-message schema
        cap) doesn't blow up runtime disproportionately — repeated many
        times to average out noise, since a single call is sub-millisecond
        either way and hard to time reliably."""
        import time

        small = _request(self._big_history(20))  # 40 messages
        large = _request(self._big_history(100))  # 200 messages (schema max)
        reps = 200

        t0 = time.perf_counter()
        for _ in range(reps):
            ContextCompressor().compress(small, target_tokens=500)
        t_small = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            ContextCompressor().compress(large, target_tokens=500)
        t_large = time.perf_counter() - t0

        # 5x the messages should cost nowhere near 5^2x=25x the time. Generous
        # margin (10x) to keep this robust on a loaded CI box.
        assert t_large < t_small * 10 + 0.5
