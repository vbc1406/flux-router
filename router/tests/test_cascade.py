"""
File: router/tests/test_cascade.py

Purpose:
Tests for Task 8 — cascade / escalation. Uses a deterministic 4-tier
registry (free/cheap/mid/premium, one model each) so escalation order is
predictable, and patches Flux._call_model to return canned per-tier text
(mirrors the pattern in test_smart_retry.py) — no real HTTP calls.

How to run:
  pytest -v router/tests/test_cascade.py
"""

from __future__ import annotations

import asyncio

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.cascade import VerificationResult, verify_response
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.errors import FluxAPIError
from router.flux import Flux
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import ModelOption, RoutingRequest


def _models() -> list[ModelOption]:
    # One model per TIER_ORDER tier (free/cheap/mid/premium) — the fallback
    # chain's "one tier up" step needs every tier populated, or it skips
    # straight past an empty tier to the next one that has a model.
    common = {
        "capabilities": [],
        "max_context_window": 100_000,
        "max_output_tokens": 4096,
        "quality_ratings": {"general": 0.7, "unknown": 0.7},
    }
    return [
        ModelOption(
            provider="p-free", model_id="m-free", display_name="Free", tier="free",
            cost_per_1k_input=0.0001, cost_per_1k_output=0.0002, **common,
        ),
        ModelOption(
            provider="p-cheap", model_id="m-cheap", display_name="Cheap", tier="cheap",
            cost_per_1k_input=0.001, cost_per_1k_output=0.002, **common,
        ),
        ModelOption(
            provider="p-mid", model_id="m-mid", display_name="Mid", tier="mid",
            cost_per_1k_input=0.005, cost_per_1k_output=0.01, **common,
        ),
        ModelOption(
            provider="p-premium", model_id="m-premium", display_name="Premium", tier="premium",
            cost_per_1k_input=0.02, cost_per_1k_output=0.04, **common,
        ),
    ]  # fmt: skip


def _engine() -> RoutingEngine:
    registry = ModelRegistry()
    registry._models = {m.model_id: m for m in _models()}
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _flux() -> Flux:
    return Flux(_engine(), api_key="test-key")


def rr(coro):
    return asyncio.run(coro)


_KWARGS = {
    "user_id": "u_cascade",
    "plan": "business_plan",
    "routing_priority": "cascade",
    "exploration_rate": 0.0,
}


class TestCascadeInitialTierSelection:
    def test_cascade_starts_at_cheapest_tier(self):
        engine = _engine()
        req = RoutingRequest(
            raw_prompt="hi", user_id="u", plan="business_plan", routing_priority="cascade"
        )
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "free"


class TestCascadeEscalation:
    def test_first_tier_passes_no_escalation(self):
        flux = _flux()

        async def mock_call(model, request):
            return "A perfectly good, complete answer."

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        assert resp.decision.cascade_attempts == 1
        assert resp.model.tier == "free"
        assert resp.fallback_used is False

    def test_escalates_when_cheap_tier_fails_verification(self):
        flux = _flux()

        # The fallback chain used as the escalation ladder is [same-tier alt,
        # one-tier-up, most-capable] — with one model per tier here, that's
        # [free (initial), cheap, premium]; "mid" is never in the ladder.
        async def mock_call(model, request):
            if model.tier == "free":
                return ""  # empty -> fails verification
            return "A perfectly good, complete answer from a bigger model."

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        assert resp.decision.cascade_attempts == 2
        assert resp.model.tier == "cheap"
        assert resp.text.startswith("A perfectly good")
        assert resp.fallback_used is True

    def test_escalates_through_all_tiers_then_returns_last_response(self):
        flux = _flux()

        async def mock_call(model, request):
            return "I'm sorry, but I cannot help with that."  # always a refusal

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        # 3 tiers available, all fail verification -> returns the last one tried anyway.
        assert resp.decision.cascade_attempts == 3
        assert resp.model.tier == "premium"
        assert resp.fallback_reason == "refusal detected"

    def test_provider_error_on_one_tier_also_triggers_escalation(self):
        flux = _flux()

        async def mock_call(model, request):
            if model.tier == "free":
                raise FluxAPIError("simulated provider outage")
            return "A perfectly good, complete answer."

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        assert resp.model.tier == "cheap"
        assert resp.decision.cascade_attempts == 2

    def test_all_tiers_erroring_raises(self):
        flux = _flux()

        async def mock_call(model, request):
            raise FluxAPIError("simulated outage")

        flux._call_model = mock_call  # type: ignore[method-assign]
        try:
            rr(flux.complete("do something", **_KWARGS))
            raise AssertionError("expected FluxAPIError")
        except FluxAPIError:
            pass


class TestCascadeNetSavings:
    def test_no_escalation_shows_positive_net_savings_vs_top_tier(self):
        flux = _flux()

        async def mock_call(model, request):
            return "A perfectly good, complete answer."

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        # Succeeded on the cheapest tier alone -> strictly cheaper than
        # always dispatching the priciest tier in the ladder.
        assert resp.decision.cascade_net_savings > 0

    def test_full_escalation_shows_negative_or_zero_net_savings(self):
        flux = _flux()

        async def mock_call(model, request):
            return ""  # every tier fails verification -> escalate through all

        flux._call_model = mock_call  # type: ignore[method-assign]
        resp = rr(flux.complete("do something", **_KWARGS))
        # Paid for free + mid + premium, vs. just paying for premium once ->
        # net savings must be negative (or at best zero): escalation cost MORE.
        assert resp.decision.cascade_net_savings <= 0


class TestVerifiers:
    def test_verify_response_flags_empty(self):
        req = RoutingRequest(raw_prompt="x", user_id="u")
        assert verify_response("", req) == VerificationResult(False, "empty response")

    def test_verify_response_flags_refusal(self):
        req = RoutingRequest(raw_prompt="x", user_id="u")
        result = verify_response("I'm sorry, but I can't do that.", req)
        assert result.passed is False
        assert result.reason == "refusal detected"

    def test_verify_response_flags_truncation(self):
        req = RoutingRequest(raw_prompt="x", user_id="u")
        long_unterminated = "This is a long response that just stops abruptly without punctuation"
        result = verify_response(long_unterminated, req)
        assert result.passed is False
        assert "truncated" in result.reason

    def test_verify_response_passes_normal_text(self):
        req = RoutingRequest(raw_prompt="x", user_id="u")
        assert verify_response("This is a fine, complete response.", req).passed is True

    def test_verify_response_checks_structured_output(self):
        req = RoutingRequest(
            raw_prompt="x",
            user_id="u",
            response_format={"json_schema": {"schema": {"required": ["name"]}}},
        )
        assert verify_response('{"name": "ok"}', req).passed is True
        bad = verify_response("not json at all !!", req)
        assert bad.passed is False
        missing = verify_response('{"other": 1}', req)
        assert missing.passed is False
        assert "name" in missing.reason

    def test_custom_verifier_runs_after_defaults(self):
        req = RoutingRequest(raw_prompt="x", user_id="u")

        def must_contain_ok(text: str, request: RoutingRequest) -> VerificationResult:
            return VerificationResult("ok" in text, "missing 'ok'")

        assert verify_response("this is ok.", req, must_contain_ok).passed is True
        assert verify_response("this is fine.", req, must_contain_ok).passed is False
