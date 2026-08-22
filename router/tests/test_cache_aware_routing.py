"""
File: router/tests/test_cache_aware_routing.py

Purpose:
Tests for cache-aware routing (router/prompt_cache.py +
routing_engine.py's cache-stickiness block in Step 9).

Uses a deterministic two-model registry (one with cache pricing modeling a
warm-prefix discount, one without) rather than the real 26-model registry,
so outcomes don't depend on the full scoring algorithm's tie-breaking.

How to run:
  pytest -v router/tests/test_cache_aware_routing.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.prompt_cache import hash_prefix
from router.routing_engine import RoutingEngine
from router.schemas import ModelOption, RoutingRequest

_LONG_SYSTEM_PROMPT = (
    "You are a careful, precise assistant operating under a long, detailed policy document. " * 150
)  # ~1500+ tokens by the classifier's ~4-chars/token heuristic


def _cache_model() -> ModelOption:
    """Model on the provider that will hold the warm cache — more expensive
    cold, much cheaper once its prefix cache is warm."""
    return ModelOption(
        provider="cache-provider",
        model_id="cache-model",
        display_name="Cache Model",
        tier="mid",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        max_context_window=200_000,
        max_output_tokens=8192,
        capabilities=[],
        cache_read_cost_per_1m=1.0,  # $0.001/1k -> a 90% discount on cache hits
        cache_min_tokens=500,
        quality_ratings={"general": 0.8, "unknown": 0.8},
    )


def _cheap_cold_model(cost_per_1k_input: float = 0.008) -> ModelOption:
    """Model on a different provider, no cache pricing modeled (always cold)."""
    return ModelOption(
        provider="cold-provider",
        model_id="cold-model",
        display_name="Cold Model",
        tier="mid",
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_input * 2,
        max_context_window=200_000,
        max_output_tokens=8192,
        capabilities=[],
        quality_ratings={"general": 0.8, "unknown": 0.8},
    )


def _engine(models: list[ModelOption]) -> RoutingEngine:
    registry = ModelRegistry()
    registry._models = {m.model_id: m for m in models}  # deterministic 2-model set
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(**kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "raw_prompt": "Please help with the next task.",
        "user_id": "u_cache_test",
        "plan": "business_plan",
    }
    defaults.update(kw)
    return RoutingRequest(correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


class TestPromptCacheTracker:
    def test_warm_lookup_matches_recorded_prefix(self):
        from router.prompt_cache import PromptCacheTracker

        tracker = PromptCacheTracker()
        h = hash_prefix("a long shared system prompt")
        assert tracker.get_warm_provider("conv-1", h) is None
        tracker.record("conv-1", "anthropic", h, 2000, ttl_seconds=60)
        assert tracker.get_warm_provider("conv-1", h) == "anthropic"

    def test_warm_lookup_misses_on_different_prefix(self):
        from router.prompt_cache import PromptCacheTracker

        tracker = PromptCacheTracker()
        tracker.record("conv-1", "anthropic", hash_prefix("prompt A"), 2000, ttl_seconds=60)
        assert tracker.get_warm_provider("conv-1", hash_prefix("prompt B")) is None

    def test_warm_lookup_expires(self):
        from router.prompt_cache import PromptCacheTracker

        tracker = PromptCacheTracker()
        h = hash_prefix("prompt")
        tracker.record("conv-1", "anthropic", h, 2000, ttl_seconds=0.01)
        import time

        time.sleep(0.05)
        assert tracker.get_warm_provider("conv-1", h) is None


class TestCacheAwareRoutingEngine:
    def test_stays_on_incumbent_when_switch_savings_below_margin(self):
        """Cold model is only marginally cheaper than the incumbent's
        cache-aware cost — must NOT clear CACHE_SWITCH_MARGIN (15%), so
        routing stays on the incumbent (warm) provider."""
        engine = _engine([_cache_model(), _cheap_cold_model(cost_per_1k_input=0.008)])
        conv_id = "conv-stay"

        req1 = _req(conversation_id=conv_id, system_prompt=_LONG_SYSTEM_PROMPT)
        d1 = rr(engine.route(req1))
        assert d1.chosen_model is not None

        # Force the incumbent regardless of what the first call happened to
        # pick, so the second call's behavior is what's under test.
        prefix_tokens = engine._classifier._count_tokens(_LONG_SYSTEM_PROMPT)
        scoped_key = engine._scoped_state_key(req1, conv_id)
        engine._prompt_cache.record(
            scoped_key, "cache-provider", hash_prefix(_LONG_SYSTEM_PROMPT), prefix_tokens, 300
        )

        req2 = _req(conversation_id=conv_id, system_prompt=_LONG_SYSTEM_PROMPT)
        d2 = rr(engine.route(req2))
        assert d2.chosen_model.provider == "cache-provider"
        assert d2.prompt_cache_status == "warm"

    def test_switches_when_savings_clear_margin(self):
        """Cold model is dramatically cheaper — the switch clears
        CACHE_SWITCH_MARGIN even accounting for the incumbent's cache
        discount, so routing switches away and reports would_lose_cache."""
        engine = _engine([_cache_model(), _cheap_cold_model(cost_per_1k_input=0.0002)])
        conv_id = "conv-switch"
        prefix_tokens = engine._classifier._count_tokens(_LONG_SYSTEM_PROMPT)
        req = _req(
            conversation_id=conv_id,
            system_prompt=_LONG_SYSTEM_PROMPT,
            routing_priority="cost-optimized",
        )
        engine._prompt_cache.record(
            engine._scoped_state_key(req, conv_id),
            "cache-provider",
            hash_prefix(_LONG_SYSTEM_PROMPT),
            prefix_tokens,
            300,
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model.provider == "cold-provider"
        assert decision.prompt_cache_status == "would_lose_cache"

    def test_short_system_prompt_is_never_cache_modeled(self):
        engine = _engine([_cache_model(), _cheap_cold_model()])
        conv_id = "conv-short"
        engine._prompt_cache.record(conv_id, "cache-provider", hash_prefix("short"), 10, 300)
        req = _req(conversation_id=conv_id, system_prompt="short")
        decision = rr(engine.route(req))
        assert decision.prompt_cache_status == "cold"

    def test_no_conversation_or_run_id_is_never_cache_modeled(self):
        engine = _engine([_cache_model(), _cheap_cold_model()])
        req = _req(system_prompt=_LONG_SYSTEM_PROMPT)
        decision = rr(engine.route(req))
        assert decision.prompt_cache_status == "cold"

    def test_run_id_alone_also_engages_cache_stickiness(self):
        engine = _engine([_cache_model(), _cheap_cold_model(cost_per_1k_input=0.008)])
        run_id = "run-cache"
        prefix_tokens = engine._classifier._count_tokens(_LONG_SYSTEM_PROMPT)
        req = _req(run_id=run_id, system_prompt=_LONG_SYSTEM_PROMPT)
        engine._prompt_cache.record(
            engine._scoped_state_key(req, run_id),
            "cache-provider",
            hash_prefix(_LONG_SYSTEM_PROMPT),
            prefix_tokens,
            300,
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model.provider == "cache-provider"
        assert decision.prompt_cache_status == "warm"
