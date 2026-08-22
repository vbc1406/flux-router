"""
File: router/tests/test_quality_max.py

Purpose:
Tests for the "quality_max" routing_priority (skip cost-minimization scoring
entirely; select the highest-capability model that passes all hard
constraints, regardless of cost). Uses a deterministic model registry (like
test_cascade.py) so which model quality_max should pick is unambiguous.

How to run:
  pytest -v router/tests/test_quality_max.py
"""

from __future__ import annotations

import asyncio

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import ModelOption, RoutingRequest


def _engine(models: list[ModelOption]) -> RoutingEngine:
    registry = ModelRegistry()
    registry._models = {m.model_id: m for m in models}
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def rr(coro):
    return asyncio.run(coro)


_COMMON = {
    "capabilities": [],
    "max_context_window": 100_000,
    "max_output_tokens": 4096,
}


class TestQualityMaxPicksTopCapability:
    """quality_max must pick the highest per-task-type quality model even
    when a cheaper, still-capable model exists — the whole point of the tier."""

    def test_picks_highest_quality_over_cheaper_capable_model(self):
        models = [
            ModelOption(
                provider="p-cheap",
                model_id="m-cheap-good-enough",
                display_name="Cheap",
                tier="cheap",
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.001,
                quality_ratings={"conversation": 0.75, "general": 0.75, "unknown": 0.75},
                **_COMMON,
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium-best",
                display_name="Premium",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                quality_ratings={"conversation": 0.97, "general": 0.97, "unknown": 0.97},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        req = RoutingRequest(
            raw_prompt="hi",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-premium-best"
        assert decision.priority_applied == "quality_max"
        assert decision.routing_rule_matched == "quality_max"

    def test_mid_tier_beats_premium_when_it_scores_higher_on_task(self):
        """quality_max ranks by quality_rating first, tier only as a
        tiebreak — unlike always-premium it is not tier-first, so a mid-tier
        model with a genuinely higher task-specific rating must win."""
        models = [
            ModelOption(
                provider="p-mid",
                model_id="m-mid-specialist",
                display_name="Mid Specialist",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                quality_ratings={"conversation": 0.99, "general": 0.99, "unknown": 0.99},
                **_COMMON,
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium-generalist",
                display_name="Premium Generalist",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                quality_ratings={"conversation": 0.85, "general": 0.85, "unknown": 0.85},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        req = RoutingRequest(
            raw_prompt="hi",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-mid-specialist"

    def test_cheapest_candidate_never_wins_over_higher_quality(self):
        """Sanity: repeated routing never drifts to the cheap model — no
        randomness/exploration should leak into the quality_max path."""
        models = [
            ModelOption(
                provider="p-free",
                model_id="m-free",
                display_name="Free",
                tier="free",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0002,
                quality_ratings={"conversation": 0.5, "general": 0.5, "unknown": 0.5},
                **_COMMON,
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium",
                display_name="Premium",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                quality_ratings={"conversation": 0.95, "general": 0.95, "unknown": 0.95},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        for _ in range(10):
            req = RoutingRequest(
                raw_prompt="hello there",
                user_id="u",
                plan="business_plan",
                routing_priority="quality_max",
                exploration_rate=0.0,
            )
            decision = rr(engine.route(req))
            assert decision.chosen_model is not None
            assert decision.chosen_model.model_id == "m-premium"


class TestQualityMaxRespectsHardConstraints:
    """quality_max must still run the Step 4 constraint pipeline — it may
    never pick a model that fails context window or tool support, even if
    that model has the highest quality_rating."""

    def test_wont_pick_model_that_fails_context_window(self):
        models = [
            ModelOption(
                provider="p-small-window-best-quality",
                model_id="m-best-but-tiny-window",
                display_name="Best But Tiny Window",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                max_context_window=500,  # too small for the request below
                max_output_tokens=4096,
                capabilities=[],
                quality_ratings={"general": 0.99, "unknown": 0.99},
            ),
            ModelOption(
                provider="p-big-window",
                model_id="m-lower-quality-big-window",
                display_name="Lower Quality Big Window",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                max_context_window=200_000,
                max_output_tokens=4096,
                capabilities=[],
                quality_ratings={"general": 0.70, "unknown": 0.70},
            ),
        ]
        engine = _engine(models)
        long_prompt = "Analyze this in depth. " * 500  # forces total_context_needed high
        req = RoutingRequest(
            raw_prompt=long_prompt,
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-lower-quality-big-window"

    def test_wont_pick_model_without_tool_support_when_tools_offered(self):
        models = [
            ModelOption(
                provider="p-no-tools",
                model_id="m-best-no-tools",
                display_name="Best No Tools",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                supports_tools=False,
                quality_ratings={"general": 0.99, "unknown": 0.99},
                **_COMMON,
            ),
            ModelOption(
                provider="p-has-tools",
                model_id="m-lower-quality-with-tools",
                display_name="Lower Quality With Tools",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                supports_tools=True,
                quality_ratings={"general": 0.70, "unknown": 0.70},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        req = RoutingRequest(
            raw_prompt="Look up the weather",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-lower-quality-with-tools"

    def test_no_candidates_survive_returns_no_model_not_a_crash(self):
        models = [
            ModelOption(
                provider="p-no-tools",
                model_id="m-only-no-tools",
                display_name="Only No Tools",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                supports_tools=False,
                quality_ratings={"general": 0.99, "unknown": 0.99},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        req = RoutingRequest(
            raw_prompt="Look up the weather",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
            tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is None
        assert decision.routing_rule_matched == "no_candidates"


class TestQualityMaxWithStepTypeFloors:
    """quality_max combined with STEP_TYPE_FLOORS should be a
    no-op relative to plain quality_max: the floor is enforced in Step 4
    (before quality_max's shortcut runs) and quality_max already picks the
    top-quality model, which for these floors is always at or above the
    floor tier anyway."""

    def _models_all_tiers(self) -> list[ModelOption]:
        # Ratings cover every task_type the tests below actually classify to
        # ("general" for the plan prompt, "extraction" for the extract
        # prompt) plus "unknown" as a catch-all — deliberately in DESCENDING
        # order free > cheap > mid > premium so a passing test proves
        # quality_max is ranking by quality_rating, not silently falling
        # back to tier order.
        ratings = [
            {"general": 0.99, "extraction": 0.99, "unknown": 0.99},  # free
            {"general": 0.80, "extraction": 0.80, "unknown": 0.80},  # cheap
            {"general": 0.70, "extraction": 0.70, "unknown": 0.70},  # mid
            {"general": 0.60, "extraction": 0.60, "unknown": 0.60},  # premium
        ]
        return [
            ModelOption(
                provider="p-free",
                model_id="m-free",
                display_name="Free",
                tier="free",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0002,
                quality_ratings=ratings[0],
                **_COMMON,
            ),
            ModelOption(
                provider="p-cheap",
                model_id="m-cheap",
                display_name="Cheap",
                tier="cheap",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
                quality_ratings=ratings[1],
                **_COMMON,
            ),
            ModelOption(
                provider="p-mid",
                model_id="m-mid",
                display_name="Mid",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                quality_ratings=ratings[2],
                **_COMMON,
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium",
                display_name="Premium",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                quality_ratings=ratings[3],
                **_COMMON,
            ),
        ]

    def test_plan_step_floor_excludes_free_even_though_free_has_top_quality(self):
        """'plan' step_type has a STEP_TYPE_FLOORS floor of 'mid'. The free
        model has the top quality_rating here, but it must be filtered out
        by Step 4 before quality_max ever sees it — proving the floor still
        gates quality_max exactly like every other priority."""
        engine = _engine(self._models_all_tiers())
        req = RoutingRequest(
            raw_prompt="Plan out the steps to migrate this database",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
            step_type="plan",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        # free/cheap are below the "mid" floor for step_type="plan" -> excluded.
        assert decision.chosen_model.tier in ("mid", "premium")
        # Among the survivors (mid=0.70, premium=0.60), quality_max still
        # picks the higher-quality one (mid), not the cheaper one.
        assert decision.chosen_model.model_id == "m-mid"

    def test_no_step_type_floor_is_true_no_op(self):
        """With no floor in play (step_type='extract' floors to 'free', i.e.
        no exclusion), quality_max's choice is identical to routing without
        step_type set at all — the floor truly changes nothing here."""
        engine_a = _engine(self._models_all_tiers())
        engine_b = _engine(self._models_all_tiers())

        req_with_floor = RoutingRequest(
            raw_prompt="Extract the dates from this document",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
            step_type="extract",
        )
        req_without_floor = RoutingRequest(
            raw_prompt="Extract the dates from this document",
            user_id="u",
            plan="business_plan",
            routing_priority="quality_max",
        )
        d_with = rr(engine_a.route(req_with_floor))
        d_without = rr(engine_b.route(req_without_floor))
        assert d_with.chosen_model is not None
        assert d_without.chosen_model is not None
        assert d_with.chosen_model.model_id == d_without.chosen_model.model_id == "m-free"
