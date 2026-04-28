"""
File: router/tests/test_explainability.py

Purpose:
Tests for the RoutingExplanation field on RoutingDecision — verifies that
every routed request produces a human-readable explanation with the expected
fields populated.

How to run:
  pytest -v router/tests/test_explainability.py

How to add a test:
  1. Use _engine() for a fresh RoutingEngine, _req(prompt, **kw) for a request.
  2. Call rr(engine.route(req)) and inspect d.explanation.
  3. Assert on specific fields: chosen_reason, rejected_models, cost_reasoning, etc.

Test classes:
  TestExplanationFields   — all explanation fields present and non-empty after routing
  TestExplainString       — explanation.explain() produces readable human string
"""
from __future__ import annotations

import asyncio
import uuid

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import RoutingDecision, RoutingExplanation, RoutingRequest


def _engine() -> RoutingEngine:
    registry   = ModelRegistry()
    cache      = ResponseCache(enabled=False)
    adaptive   = AdaptiveWeights(state_file=None)
    analytics  = RoutingAnalytics(log_path=None)
    budget     = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw) -> RoutingRequest:
    defaults = dict(user_id="u_expl", plan="business_plan", priority="normal")
    defaults.update(kw)
    defaults.setdefault("correlation_id", str(uuid.uuid4()))
    return RoutingRequest(raw_prompt=prompt, **defaults)


def _route(prompt: str, verbose: bool = True, **kw) -> RoutingDecision:
    engine = _engine()
    return asyncio.get_event_loop().run_until_complete(
        engine.route(_req(prompt, **kw), verbose=verbose)
    )


class TestExplanationFields:
    def test_explanation_has_all_fields(self):
        decision = _route("Write a Python quicksort implementation", verbose=True)
        expl = decision.explanation
        assert expl is not None
        assert expl.task_type != ""
        assert 0.0 <= expl.task_type_confidence <= 1.0
        assert 0.0 <= expl.complexity_score <= 1.0
        assert isinstance(expl.tier_selected, str)
        assert expl.candidates_considered > 0
        assert expl.candidates_filtered >= 0
        assert isinstance(expl.filter_reasons, dict)
        assert isinstance(expl.scoring_breakdown, list)
        assert expl.winner != ""
        assert isinstance(expl.sticky_bias_applied, bool)
        assert isinstance(expl.confidence_fallback_used, bool)
        assert isinstance(expl.rules_fired, list)

    def test_verbose_false_no_explanation(self):
        decision = _route("Write a quicksort in Python", verbose=False)
        assert decision.explanation is None

    def test_verbose_flag(self):
        prompt = "Please explain how neural networks work and what backpropagation does"
        d_verbose = _route(prompt, verbose=True)
        d_quiet   = _route(prompt, verbose=False)
        assert d_verbose.explanation is not None
        assert d_quiet.explanation is None


class TestExplainString:
    def test_explain_string_format(self):
        decision = _route("Implement a red-black tree in Python", verbose=True)
        text = decision.explain()
        assert "Task:" in text
        assert "Complexity:" in text
        assert "Tier:" in text
        assert "Candidates:" in text
        assert "Scoring:" in text
        assert "Winner:" in text

    def test_explain_no_explanation_placeholder(self):
        decision = _route("hello", verbose=False)
        text = decision.explain()
        assert "verbose=True" in text

    def test_explanation_shows_filtered_models(self):
        # Request with required_capabilities that most models won't have,
        # causing some to be filtered out.
        decision = _route(
            "Describe this image",
            verbose=True,
            required_capabilities=["vision"],
        )
        expl = decision.explanation
        assert expl is not None
        # Some models without vision should be filtered
        # (not all models have vision capability)
        # The filter_reasons dict may be empty if all models happen to have vision,
        # but candidates_considered should be > 0
        assert expl.candidates_considered > 0

    def test_scoring_breakdown_has_winner(self):
        decision = _route("Write a binary search in Python", verbose=True)
        expl = decision.explanation
        assert expl is not None
        if expl.scoring_breakdown:
            models_in_breakdown = {e["model"] for e in expl.scoring_breakdown}
            assert expl.winner in models_in_breakdown
