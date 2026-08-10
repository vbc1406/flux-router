"""
File: router/tests/test_long_document_capability.py

Purpose:
Tests for Item 4 — the explicit `required_capabilities=["long_document"]`
fix. Before this change, no catalog entry's `capabilities` list contained
"long_document", so an explicit request for it always returned zero
candidates. Now it's a DERIVED capability
(routing_engine._capability_satisfied): a model satisfies it when its
max_context_window covers the request's total_context_needed plus
LONG_DOCUMENT_CONTEXT_SAFETY_MARGIN. Automatic long-document inference
(word count > 2000 -> task_type="long_document") is untouched and covered
separately in test_classifier.py.

How to run:
  pytest -v router/tests/test_long_document_capability.py
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
from router.routing_engine import RoutingEngine
from router.schemas import ModelOption, RoutingRequest


def _engine(models: list[ModelOption] | None = None) -> RoutingEngine:
    registry = ModelRegistry()
    if models is not None:
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
        "user_id": "u_long_doc",
        "plan": "business_plan",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


def _word_prompt(char_count: int) -> str:
    """~5 chars/word ('word ' repeated) prompt of approximately char_count chars."""
    return "word " * (char_count // 5)


class TestExplicitLongDocumentCapability:
    def test_normal_short_prompt_finds_models(self):
        engine = _engine()
        decision = rr(
            engine.route(_req("Summarize the key point.", required_capabilities=["long_document"]))
        )
        assert decision.chosen_model is not None

    def test_50k_chars_finds_models(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(_word_prompt(50_000), required_capabilities=["long_document"])
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.max_context_window >= 50_000 // 4  # rough token floor

    def test_300k_chars_finds_models(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(_word_prompt(300_000), required_capabilities=["long_document"])
            )
        )
        assert decision.chosen_model is not None

    def test_approximately_900k_chars_finds_models_or_clean_no_match(self):
        engine = _engine()
        # Stay under pydantic's 1,000,000-char raw_prompt ceiling.
        decision = rr(
            engine.route(
                _req(_word_prompt(895_000), required_capabilities=["long_document"])
            )
        )
        # Either a genuinely large-window model is found, or routing cleanly
        # reports no candidates — never a crash, never a too-small model.
        if decision.chosen_model is not None:
            assert decision.chosen_model.max_context_window >= 895_000 // 4
        else:
            assert decision.routing_rule_matched == "no_candidates"

    def test_request_exceeding_every_models_context_returns_no_candidates(self):
        small_models = [
            ModelOption(
                provider="p-small",
                model_id="m-small-window",
                display_name="Small Window",
                tier="mid",
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
                max_context_window=8_000,
                max_output_tokens=2048,
                capabilities=[],
                quality_ratings={"general": 0.8},
            ),
        ]
        engine = _engine(small_models)
        decision = rr(
            engine.route(
                _req(_word_prompt(895_000), required_capabilities=["long_document"])
            )
        )
        assert decision.chosen_model is None
        assert decision.routing_rule_matched == "no_candidates"

    def test_context_window_boundary_respects_safety_margin(self):
        """A model whose window equals total_context_needed EXACTLY (no
        margin) must NOT satisfy long_document; one with enough headroom
        must."""
        prompt = _word_prompt(20_000)
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req(prompt))
        needed = analysis.total_context_needed

        from router.config import LONG_DOCUMENT_CONTEXT_SAFETY_MARGIN

        exact_fit_model = ModelOption(
            provider="p-exact",
            model_id="m-exact-fit",
            display_name="Exact Fit",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            max_context_window=needed,  # no margin at all
            max_output_tokens=2048,
            capabilities=[],
            quality_ratings={"general": 0.8},
        )
        with_margin_model = ModelOption(
            provider="p-margin",
            model_id="m-with-margin",
            display_name="With Margin",
            tier="mid",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.02,
            max_context_window=needed + LONG_DOCUMENT_CONTEXT_SAFETY_MARGIN + 100,
            max_output_tokens=2048,
            capabilities=[],
            quality_ratings={"general": 0.8},
        )
        engine = _engine([exact_fit_model, with_margin_model])
        decision = rr(
            engine.route(_req(prompt, required_capabilities=["long_document"]))
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.model_id == "m-with-margin"

    def test_combined_with_vision_capability(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "Describe this long document and the attached image.",
                    required_capabilities=["long_document", "vision"],
                )
            )
        )
        assert decision.chosen_model is not None
        assert "vision" in decision.chosen_model.capabilities

    def test_combined_with_tools(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "Search across this long document using the search tool.",
                    required_capabilities=["long_document"],
                    tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}],
                )
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.supports_tools is True

    def test_combined_with_structured_output(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "Extract structured data from this long document.",
                    required_capabilities=["long_document"],
                    response_format={"type": "json_object"},
                )
            )
        )
        assert decision.chosen_model is not None
        assert decision.chosen_model.supports_structured_output is True

    def test_sensitivity_restriction_still_applies(self):
        engine = _engine()
        decision = rr(
            engine.route(
                _req(
                    "This is classified. Summarize this long document.",
                    required_capabilities=["long_document"],
                    metadata={"sensitivity_level": "restricted"},
                )
            )
        )
        assert decision.chosen_model is not None
        assert "restricted" in decision.chosen_model.allowed_sensitivity_levels

    def test_automatic_long_document_inference_unchanged(self):
        """Word-count > 2000 -> task_type='long_document' without any
        required_capabilities set — untouched by the capability fix."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req("word " * 2500))
        assert analysis.task_type == "long_document"

        engine = _engine()
        decision = rr(engine.route(_req("word " * 2500)))
        assert decision.chosen_model is not None
        assert decision.task_type == "long_document"
