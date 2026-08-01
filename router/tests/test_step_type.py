"""
File: router/tests/test_step_type.py

Purpose:
Tests for Task 6 — step-type classification for agent trajectories.
Covers step_type inference (classifier.py), the STEP_TYPE_FLOORS hard
constraint (routing_engine.py::_passes_hard_constraints), and the
tools/response_format capability filter.

How to run:
  pytest -v router/tests/test_step_type.py
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
from router.schemas import RoutingRequest

_TOOLS = [{"type": "function", "function": {"name": f"tool_{i}"}} for i in range(6)]


def _engine() -> RoutingEngine:
    registry = ModelRegistry()
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {"user_id": "u_step_type", "plan": "business_plan"}
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


class TestStepTypeInference:
    def test_tools_present_infers_tool_select(self):
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req("What should I do next?", tools=_TOOLS))
        assert analysis.step_type == "tool_select"

    def test_tool_result_message_infers_tool_result_summarize(self):
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(
            _req(
                "Summarize what the tool returned.",
                message_history=[{"role": "tool", "content": "{...raw tool output...}"}],
            )
        )
        assert analysis.step_type == "tool_result_summarize"

    def test_response_format_infers_extract(self):
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(
            _req("Pull the fields out.", response_format={"type": "json_schema"})
        )
        assert analysis.step_type == "extract"

    def test_explicit_step_type_overrides_inference(self):
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req("hi", tools=_TOOLS, step_type="final_answer"))
        assert analysis.step_type == "final_answer"

    def test_no_signal_is_unknown(self):
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(_req("What is 2+2?"))
        assert analysis.step_type == "unknown"

    # ── Bugfix coverage: final_answer inference + fail-safe default ────────

    def test_explicit_step_type_final_answer_passthrough(self):
        """request.step_type is trusted verbatim and never overridden by
        inference, even when other signals (tools) are also present."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(
            _req("Wrap this up.", tools=_TOOLS, step_type="final_answer")
        )
        assert analysis.step_type == "final_answer"

    def test_explicit_final_answer_language_infers_final_answer(self):
        """A clean final-turn request with no tools/history but explicit
        'final answer' language in the prompt infers final_answer."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        analysis = classifier.analyze(
            _req("Here is my final answer: the capital of France is Paris.")
        )
        assert analysis.step_type == "final_answer"

    def test_stale_tool_result_earlier_in_history_does_not_shadow_final_answer(self):
        """The specific misclassification this bugfix targets: a tool-result
        message is still present in history (from an earlier step), but the
        CURRENT step is the final answer — it must not be misread as
        tool_result_summarize just because a tool message exists somewhere
        in the transcript."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        history = [
            {"role": "user", "content": "What's the weather in Boston?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "content": '{"temp_f": 72, "conditions": "sunny"}'},
            {"role": "assistant", "content": "It's sunny and 72F."},
            {"role": "user", "content": "Great, thanks!"},
        ]
        analysis = classifier.analyze(
            _req(
                "Give your final answer to the user now.",
                message_history=history,
                run_id="run-final-1",
            )
        )
        assert analysis.step_type == "final_answer"

    def test_most_recent_tool_result_still_infers_tool_result_summarize(self):
        """Only the LAST message being a tool result should trigger
        tool_result_summarize — this must keep working after the bugfix."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        history = [
            {"role": "user", "content": "What's the weather in Boston?"},
            {"role": "tool", "content": '{"temp_f": 72, "conditions": "sunny"}'},
        ]
        analysis = classifier.analyze(
            _req("Summarize what the tool returned.", message_history=history)
        )
        assert analysis.step_type == "tool_result_summarize"

    def test_ambiguous_mid_run_step_fails_safe_to_final_answer_not_unknown(self):
        """No explicit signal, no tools offered, but this IS part of a
        tracked run with history already underway — the fail-safe default
        must protect it with the highest floor instead of leaving it
        classified as 'unknown' (no floor at all)."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        history = [
            {"role": "user", "content": "Please help me plan a trip."},
            {"role": "assistant", "content": "Sure, where to?"},
        ]
        analysis = classifier.analyze(
            _req("Thanks, that's everything I needed.", message_history=history, run_id="run-2")
        )
        assert analysis.step_type == "final_answer"

    def test_ambiguous_single_shot_request_without_run_id_stays_unknown(self):
        """The fail-safe default only applies inside a tracked run — plain
        single-shot chat usage (no run_id) must be unaffected."""
        classifier = RequestClassifier(ResponseCache(enabled=False))
        history = [
            {"role": "user", "content": "Please help me plan a trip."},
            {"role": "assistant", "content": "Sure, where to?"},
        ]
        analysis = classifier.analyze(
            _req("Thanks, that's everything I needed.", message_history=history)
        )
        assert analysis.step_type == "unknown"


class TestStepTypeFloorEnforcement:
    def test_tool_select_with_six_tools_never_routes_below_floor(self):
        engine = _engine()
        # A trivial-looking prompt would normally route free/cheap tier —
        # the tool_select floor (STEP_TYPE_FLOORS["tool_select"] == "mid")
        # must override that.
        req = _req("ok", tools=_TOOLS, step_type="tool_select")
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("mid", "premium")

    def test_final_answer_never_routes_below_floor(self):
        engine = _engine()
        req = _req("thanks", step_type="final_answer")
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("mid", "premium")

    def test_tool_result_summarize_has_no_floor(self):
        """tool_result_summarize has no entry in STEP_TYPE_FLOORS ("aggressive
        downgrade allowed") — it must route IDENTICALLY to the same prompt
        with no step_type set at all, i.e. complexity scoring decides alone."""
        prompt = "Summarize this tool output in one sentence: the weather is sunny."
        engine = _engine()
        baseline = rr(engine.route(_req(prompt))).chosen_model
        with_step_type = rr(
            engine.route(_req(prompt, step_type="tool_result_summarize"))
        ).chosen_model
        assert baseline is not None and with_step_type is not None
        assert with_step_type.tier == baseline.tier

    def test_step_type_floor_is_the_only_thing_that_changes_tier(self):
        """Same prompt, same complexity — only step_type differs. A floor
        step_type must never route CHEAPER than no floor at all; tool_select
        (floor="mid") must land at >= the tier that "unknown" (no floor)
        picks for the identical prompt."""
        prompt = "Handle this."
        engine = _engine()
        unfloored = rr(engine.route(_req(prompt, step_type="unknown"))).chosen_model
        floored = rr(engine.route(_req(prompt, tools=_TOOLS, step_type="tool_select"))).chosen_model
        assert unfloored is not None and floored is not None
        from router.config import TIER_ORDER

        assert TIER_ORDER.index(floored.tier) >= TIER_ORDER.index(unfloored.tier)

    def test_reflect_floor_is_between_summarize_and_plan(self):
        engine = _engine()
        req = _req("Reflect on your own answer and check for mistakes.", step_type="reflect")
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("cheap", "mid", "premium")

    def test_inferred_final_answer_from_stale_tool_result_gets_the_floor(self):
        """End-to-end: the misclassification scenario from
        TestStepTypeInference, but asserting the routing CONSEQUENCE — a
        trivial-looking prompt that would otherwise route free/cheap must
        land at the final_answer floor once a stale tool-result message in
        history is correctly no longer shadowing it."""
        engine = _engine()
        history = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "tool", "content": '{"temp_f": 72}'},
            {"role": "assistant", "content": "It's 72F."},
            {"role": "user", "content": "ok"},
        ]
        req = _req(
            "Give your final answer to the user now.",
            message_history=history,
            run_id="run-final-floor",
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("mid", "premium")


class TestToolCapabilityFilter:
    def test_tools_present_filters_to_tool_capable_models(self):
        engine = _engine()
        req = _req("Pick a tool.", tools=_TOOLS)
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.supports_tools is True

    def test_response_format_filters_to_structured_output_models(self):
        engine = _engine()
        req = _req("Return JSON.", response_format={"type": "json_schema"})
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.supports_structured_output is True

    def test_no_tool_capable_models_yields_no_chosen_model(self):
        engine = _engine()
        # Mutate the registry's actual entries (not the mutable copies
        # all_available_models() hands out) so the change is visible inside
        # route()'s own registry lookups.
        for m in engine._registry._models.values():
            m.supports_tools = False
        req = _req("Pick a tool.", tools=_TOOLS)
        decision = rr(engine.route(req))
        assert decision.chosen_model is None
