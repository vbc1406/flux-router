"""
File: router/tests/test_step_quality_ratings.py

Purpose:
Tests for Item 3 — agent-step-specific quality data
(ModelOption.step_quality_ratings, routing_engine._resolve_quality()) and
its consistent use across balanced/quality-first/cost-optimized/quality_max/
cascade scoring, explainability, and non-interference with adaptive learning
(which stays keyed by (model_id, task_type) only).

How to run:
  pytest -v router/tests/test_step_quality_ratings.py
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
from router.routing_engine import RoutingEngine, _resolve_quality
from router.schemas import ModelOption, RoutingRequest, TaskAnalysis


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


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id": "u_step_quality",
        "plan": "business_plan",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


_COMMON = {
    "capabilities": [],
    "max_context_window": 100_000,
    "max_output_tokens": 4096,
}

_TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]


class TestResolveQualityHelper:
    """Unit tests directly against _resolve_quality() for full control over
    the priority chain, independent of routing/scoring noise."""

    def test_step_specific_rating_used_when_present(self):
        model = ModelOption(
            provider="p",
            model_id="m",
            display_name="M",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_ratings={"function_calling": 0.5, "general": 0.5},
            step_quality_ratings={"tool_select": 0.95},
            **_COMMON,
        )
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            total_context_needed=20,
            task_type="function_calling",
            step_type="tool_select",
        )
        quality, source = _resolve_quality(model, analysis)
        assert quality == 0.95
        assert source == "step:tool_select"

    def test_missing_step_rating_falls_back_to_task_type(self):
        model = ModelOption(
            provider="p",
            model_id="m",
            display_name="M",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_ratings={"function_calling": 0.7, "general": 0.5},
            step_quality_ratings={},  # no rating for this step
            **_COMMON,
        )
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            total_context_needed=20,
            task_type="function_calling",
            step_type="tool_select",
        )
        quality, source = _resolve_quality(model, analysis)
        assert quality == 0.7
        assert source == "task:function_calling"

    def test_missing_task_type_falls_back_to_general(self):
        model = ModelOption(
            provider="p",
            model_id="m",
            display_name="M",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_ratings={"general": 0.6},  # no "legal" key
            **_COMMON,
        )
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            total_context_needed=20,
            task_type="legal",
            step_type="unknown",
        )
        quality, source = _resolve_quality(model, analysis)
        assert quality == 0.6
        assert source == "fallback:general"

    def test_unknown_step_type_never_consults_step_quality_ratings(self):
        model = ModelOption(
            provider="p",
            model_id="m",
            display_name="M",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_ratings={"general": 0.6},  # no "legal" key
            step_quality_ratings={"unknown": 0.99},  # must never be consulted
            **_COMMON,
        )
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            total_context_needed=20,
            task_type="legal",
            step_type="unknown",
        )
        quality, source = _resolve_quality(model, analysis)
        assert quality == 0.6
        assert source == "fallback:general"

    def test_no_rating_anywhere_defaults_to_half(self):
        model = ModelOption(
            provider="p",
            model_id="m",
            display_name="M",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            quality_ratings={},
            **_COMMON,
        )
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=10,
            estimated_output_tokens=10,
            total_context_needed=20,
            task_type="general",
            step_type="unknown",
        )
        quality, source = _resolve_quality(model, analysis)
        assert quality == 0.5
        assert source == "fallback:general"


class TestStepQualityRatingsChangeRouting:
    """A synthetic pair where task-level quality favors one model but
    step_quality_ratings favors the other for the actual step_type."""

    def _models(self) -> list[ModelOption]:
        return [
            ModelOption(
                provider="p-a",
                model_id="m-task-favorite",
                display_name="Task Favorite",
                tier="mid",
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
                supports_tools=True,
                quality_ratings={"function_calling": 0.95, "general": 0.95},
                step_quality_ratings={"tool_select": 0.65},  # worse at tool_select specifically
                **_COMMON,
            ),
            ModelOption(
                provider="p-b",
                model_id="m-step-favorite",
                display_name="Step Favorite",
                tier="mid",
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
                supports_tools=True,
                quality_ratings={"function_calling": 0.60, "general": 0.60},
                step_quality_ratings={"tool_select": 0.98},  # great at tool_select specifically
                **_COMMON,
            ),
        ]

    def test_quality_max_picks_step_favorite_for_tool_select(self):
        engine = _engine(self._models())
        req = _req(
            "Decide which tool to call to look up the weather.",
            routing_priority="quality_max",
            tools=_TOOLS,
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-step-favorite"

    def test_balanced_scoring_prefers_step_favorite_quality_component(self):
        engine = _engine(self._models())
        req = _req(
            "Decide which tool to call to look up the weather.",
            routing_priority="quality-first",
            tools=_TOOLS,
        )
        decision = rr(engine.route(req), )
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-step-favorite"

    def test_non_agent_request_uses_task_level_rating_unaffected(self):
        """With no step_type signal (plain chat, no tools/response_format),
        step_quality_ratings must not affect the outcome at all — the
        task-level 'function_calling' rating still favors m-task-favorite,
        but since this prompt has no tools offered it classifies as
        something else entirely and both models are equally likely; the
        real assertion is that resolution used task/general, not step."""
        from router.classifier import RequestClassifier as StepAgnosticClassifier

        classifier = StepAgnosticClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req("Just say hello."))
        assert analysis.step_type == "unknown"
        model = self._models()[1]  # m-step-favorite, has a step_quality_ratings entry
        quality, source = _resolve_quality(model, analysis)
        assert source != "step:tool_select"

    def test_explainability_reports_step_quality_source(self):
        engine = _engine(self._models())
        req = _req(
            "Decide which tool to call to look up the weather.",
            routing_priority="balanced",
            tools=_TOOLS,
        )
        decision = rr(engine.route(req, verbose=True))
        assert decision.explanation is not None
        sources = {
            e["model"]: e.get("quality_source") for e in decision.explanation.scoring_breakdown
        }
        assert sources.get("m-step-favorite") == "step:tool_select"
        assert sources.get("m-task-favorite") == "step:tool_select"


class TestStepFloorsStillEnforcedWithStepQuality:
    def test_plan_floor_still_excludes_free_tier_even_with_step_rating(self):
        models = [
            ModelOption(
                provider="p-free",
                model_id="m-free-great-step-rating",
                display_name="Free",
                tier="free",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0002,
                quality_ratings={"general": 0.5},
                step_quality_ratings={"plan": 0.99},
                **_COMMON,
            ),
            ModelOption(
                provider="p-mid",
                model_id="m-mid-eligible",
                display_name="Mid",
                tier="mid",
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
                quality_ratings={"general": 0.5},
                step_quality_ratings={"plan": 0.70},
                **_COMMON,
            ),
        ]
        engine = _engine(models)
        req = _req(
            "Plan out the steps to migrate this database.",
            step_type="plan",
            run_id="run-1",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        # "plan" floors to "mid" — the free-tier model must be excluded
        # regardless of its excellent step_quality_ratings entry.
        assert decision.chosen_model.tier == "mid"
        assert decision.chosen_model.model_id == "m-mid-eligible"


class TestAdaptiveLearningUnaffectedByStepQualityRatings:
    def test_adaptive_state_keyed_only_by_model_and_task_type(self):
        """AdaptiveWeights.record()/get_adjusted_score() must remain keyed
        by (model_id, task_type) only — step_quality_ratings must never leak
        into that key namespace, per adaptive_weights.py's own documented
        extension-point warning."""
        adaptive = AdaptiveWeights(state_file=None)
        model_id = "m-step-favorite"
        task_type = "function_calling"
        base = 0.6

        # Feed enough samples to activate the EMA (mirrors quality_scorer.py's
        # call pattern: record(model_id, task_type, observed, base)).
        for _ in range(25):
            adaptive.record(model_id, task_type, 0.9, base)

        adjusted_for_task = adaptive.get_adjusted_score(model_id, task_type, base)
        # A completely different "task_type" string equal to a step_type name
        # must not accidentally hit the same learned state.
        adjusted_for_fake_step_key = adaptive.get_adjusted_score(model_id, "tool_select", base)

        assert adjusted_for_task != base  # learning actually kicked in
        assert adjusted_for_fake_step_key == base  # untouched, separate key
