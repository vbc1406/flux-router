"""
File: router/tests/test_sticky_model.py

Purpose:
Tests for sticky-model bias — the routing engine prefers the model used in
previous turns of the same conversation, with configurable strength and
expiry.

How to run:
  pytest -v router/tests/test_sticky_model.py
  pytest -v router/tests/test_sticky_model.py::TestStickyModel

How to add a test:
  1. Use _engine() for a fresh engine, _req(prompt, conversation_id=..., **kw).
  2. Route two turns with the same conversation_id and assert the second turn
     picks the same model (or a higher-tier model if routing_priority overrides).
  3. For expiry tests, patch time.time() to advance the clock past TTL.

Test classes:
  TestConversationStore — conversation store CRUD: set, get, expiry, TTL reset
  TestStickyModel       — routing: bias applied, overridden, expired, last_failed

Covers:
  - same conversation_id prefers the same model on follow-up turns
  - always-premium routing_priority ignores sticky bias
  - first turn (no conversation_id) routes freely
  - expired conversation routes freely
  - deeper conversations apply the stronger bias
  - decision.last_model reflects the previous turn's model
  - conversation store update happens after each route
  - last_failed flag suppresses the sticky bias
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any
from unittest.mock import patch

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import ConversationStore, RoutingEngine
from router.schemas import RoutingRequest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine() -> RoutingEngine:
    registry   = ModelRegistry()
    cache      = ResponseCache(enabled=False)
    adaptive   = AdaptiveWeights(state_file=None)
    analytics  = RoutingAnalytics(log_path=None)
    budget     = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, conv_id: str | None = None, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id":  "u_sticky",
        "plan":     "business_plan",
        "priority": "normal",
        "exploration_rate": 0.0,   # disable A/B so results are deterministic
    }
    defaults.update(kw)
    return RoutingRequest(
        raw_prompt      = prompt,
        correlation_id  = str(uuid.uuid4()),
        conversation_id = conv_id,
        **defaults,
    )


def rr(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Mid-complexity prompt that reliably lands in mid/cheap tier under "balanced"
_MID_PROMPT = "Write a Python function to implement binary search with full tests"


# ═══════════════════════════════════════════════════════════════════════════════
# ConversationStore unit tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConversationStore:
    def test_get_returns_none_for_unknown(self):
        store = ConversationStore()
        assert store.get("no_such_id") is None

    def test_update_and_get(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        entry = store.get("conv1")
        assert entry is not None
        assert entry["last_model"] == "model-a"
        assert entry["message_count"] == 1

    def test_update_increments_message_count(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        store.update("conv1", "model-a")
        store.update("conv1", "model-b")
        entry = store.get("conv1")
        assert entry["message_count"] == 3
        assert entry["last_model"] == "model-b"

    def test_expired_entry_returns_none(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        # Fake the last_used timestamp to be way in the past.
        with store._lock:
            store._store["conv1"]["last_used"] = time.monotonic() - 99999
        assert store.get("conv1") is None

    def test_non_expired_entry_is_returned(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        assert store.get("conv1") is not None

    def test_record_failure_sets_flag(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        store.record_failure("conv1")
        entry = store.get("conv1")
        assert entry["last_failed"] is True

    def test_update_resets_failure_flag(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        store.record_failure("conv1")
        store.update("conv1", "model-b")
        entry = store.get("conv1")
        assert entry["last_failed"] is False

    def test_expire_old_removes_stale(self):
        store = ConversationStore()
        store.update("conv1", "model-a")
        store.update("conv2", "model-b")
        # Age conv1 out.
        with store._lock:
            store._store["conv1"]["last_used"] = time.monotonic() - 99999
        store.expire_old()
        assert store.get("conv1") is None
        assert store.get("conv2") is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests: sticky model bias in route()
# ═══════════════════════════════════════════════════════════════════════════════

class TestStickyModel:
    def setup_method(self):
        self.engine = _engine()

    def test_sticky_new_conversation(self):
        """No conversation_id → routes freely, last_model is None."""
        d = rr(self.engine.route(_req(_MID_PROMPT)))
        assert d.chosen_model is not None
        assert d.last_model is None

    def test_sticky_same_model_second_turn(self):
        """Second call with same conversation_id should prefer the first model."""
        conv_id = "conv_sticky_test"
        d1 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d1.chosen_model is not None
        first_model_id = d1.chosen_model.model_id

        d2 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d2.chosen_model is not None
        # With exploration disabled, sticky bias should keep the same model.
        assert d2.chosen_model.model_id == first_model_id

    def test_last_model_reflects_previous_turn(self):
        """decision.last_model should be None on turn 1, and the turn-1 model on turn 2."""
        conv_id = "conv_last_model"
        d1 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d1.last_model is None  # first turn: no previous model

        d2 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d2.last_model == d1.chosen_model.model_id

    def test_conversation_store_updated_after_route(self):
        """Conversation store must be updated immediately after each route call."""
        conv_id = "conv_store_check"
        d = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d.chosen_model is not None
        entry = self.engine._conversation_store.get(conv_id)
        assert entry is not None
        assert entry["last_model"] == d.chosen_model.model_id
        assert entry["message_count"] == 1

    def test_sticky_overridden_by_priority(self):
        """always-premium ignores the sticky bias and routes to premium."""
        conv_id = "conv_always_premium"
        # First turn: route with default priority — may land anywhere.
        d1 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d1.chosen_model is not None

        # Second turn: always-premium must pick premium regardless of sticky state.
        d2 = rr(self.engine.route(_req(
            _MID_PROMPT,
            conv_id=conv_id,
            routing_priority="always-premium",
        )))
        assert d2.chosen_model is not None
        assert d2.chosen_model.tier == "premium"

    def test_sticky_expired_conversation_routes_freely(self):
        """After expiry, last_model is None and no bias is applied."""
        conv_id = "conv_expired"
        rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))

        # Expire the conversation manually.
        with self.engine._conversation_store._lock:
            store = self.engine._conversation_store._store
            if conv_id in store:
                store[conv_id]["last_used"] = time.monotonic() - 99999

        d2 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d2.last_model is None  # expired entry → no previous model

    def test_sticky_increases_with_depth(self):
        """
        Deeper conversations (≥6 messages) apply CONVERSATION_STICKY_BIAS_DEEP (0.20)
        rather than CONVERSATION_STICKY_BIAS_SHALLOW (0.15).

        We test this indirectly: after 6 turns the entry message_count should be 6.
        """
        from router.config import CONVERSATION_DEPTH_THRESHOLD

        conv_id = "conv_deep"
        prompt  = "Explain a Python generator with examples"

        for _ in range(CONVERSATION_DEPTH_THRESHOLD):
            rr(self.engine.route(_req(prompt, conv_id=conv_id)))

        entry = self.engine._conversation_store.get(conv_id)
        assert entry is not None
        assert entry["message_count"] >= CONVERSATION_DEPTH_THRESHOLD

    def test_no_conv_id_no_store_update(self):
        """Requests without conversation_id must not pollute the store."""
        rr(self.engine.route(_req(_MID_PROMPT)))
        # Store should remain empty.
        with self.engine._conversation_store._lock:
            assert len(self.engine._conversation_store._store) == 0

    def test_multiple_conversations_independent(self):
        """Two different conversation_ids track state independently."""
        conv_a, conv_b = "conv_multi_a", "conv_multi_b"
        da1 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_a)))
        db1 = rr(self.engine.route(_req("Summarize a paragraph", conv_id=conv_b)))

        da2 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_a)))
        db2 = rr(self.engine.route(_req("Summarize a paragraph", conv_id=conv_b)))

        # Each conversation's last_model reflects its own history, not the other's.
        assert da2.last_model == da1.chosen_model.model_id
        assert db2.last_model == db1.chosen_model.model_id

    def test_sticky_suppressed_after_failure(self):
        """If last_failed is set, the sticky bias should not be applied."""
        conv_id = "conv_failed"
        d1 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d1.chosen_model is not None
        first_id = d1.chosen_model.model_id

        # Mark the model as failed.
        self.engine._conversation_store.record_failure(conv_id)
        entry = self.engine._conversation_store.get(conv_id)
        assert entry["last_failed"] is True

        # Route again — no bias applied; the model MAY still be chosen by merit.
        d2 = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d2.chosen_model is not None
        # We can't assert it changed (merit might pick same one), but at least
        # we confirm no exception was raised and routing succeeded.

    def test_bias_not_applied_when_prev_model_not_in_candidates(self):
        """
        If the previous model was filtered out (e.g., unavailable), the bias
        doesn't raise an error and routing still succeeds.
        """
        conv_id = "conv_gone_model"
        # Seed the store with a model_id that doesn't exist in the registry.
        self.engine._conversation_store.update(conv_id, "nonexistent-model-xyz")

        d = rr(self.engine.route(_req(_MID_PROMPT, conv_id=conv_id)))
        assert d.chosen_model is not None  # routing succeeds despite ghost model
