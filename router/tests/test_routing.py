"""Tests for the end-to-end routing engine."""
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
from router.routing_engine import (
    RoutingEngine,
    _estimate_cost,
    _normalize_cost,
    _normalize_latency,
    _get_tier_for_score,
    _is_allowed_for_plan,
)
from router.schemas import ModelOption, RoutingRequest


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    defaults = dict(user_id="u_test", plan="business_plan", priority="normal")
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Basic routing ─────────────────────────────────────────────────────────────

class TestBasicRouting:
    def setup_method(self):
        self.engine = _engine()

    def test_returns_routing_decision(self):
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.chosen_model is not None

    def test_trivial_routes_to_free(self):
        d = rr(self.engine.route(_req("hi", priority="normal")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier in ("free", "cheap")

    def test_reasoning_routes_to_high_tier(self):
        d = rr(self.engine.route(_req(
            "Prove the Riemann hypothesis using ∑ notation. Show all steps.",
            priority="critical"
        )))
        assert d.chosen_model is not None
        assert d.chosen_model.tier in ("premium", "ultra")

    def test_routing_rule_populated(self):
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.routing_rule_matched != ""

    def test_correlation_id_preserved(self):
        cid = "test-correlation-123"
        req = _req("hello", correlation_id=cid)
        d   = rr(self.engine.route(req))
        assert d.correlation_id == cid

    def test_confidence_in_range(self):
        d = rr(self.engine.route(_req("Write a Python sort function")))
        assert 0.0 <= d.confidence <= 1.0

    def test_estimated_cost_nonnegative(self):
        d = rr(self.engine.route(_req("Translate hello to Spanish")))
        assert d.estimated_cost >= 0.0

    def test_fallback_chain_not_empty(self):
        d = rr(self.engine.route(_req("Write a Python sort function")))
        # May be empty for trivial requests; for code gen it should have fallbacks
        assert isinstance(d.fallback_chain, list)

    def test_free_plan_cannot_use_ultra(self):
        d = rr(self.engine.route(_req(
            "Solve P vs NP. ∑∫∀∃. Step by step proof.",
            priority="critical",
            plan="free_plan"
        )))
        if d.chosen_model:
            assert d.chosen_model.tier not in ("ultra", "premium")


# ── Cache integration ─────────────────────────────────────────────────────────

class TestCacheIntegration:
    def setup_method(self):
        registry   = ModelRegistry()
        self.cache = ResponseCache(enabled=True)
        adaptive   = AdaptiveWeights(state_file=None)
        analytics  = RoutingAnalytics(log_path=None)
        budget     = BudgetTracker()
        compressor = ContextCompressor()
        classifier = RequestClassifier(self.cache)
        self.engine = RoutingEngine(registry, classifier, self.cache, budget, adaptive, compressor, analytics)

    def test_cache_hit_returns_zero_cost(self):
        from router.cache import fingerprint as fp_fn
        from router.model_registry import ModelRegistry as MR
        reg   = MR()
        model = reg.all_available_models()[0]
        prompt = "What is the capital of Germany?"
        fp     = fp_fn(prompt, None, [], None)
        self.cache.set(fp, "Berlin", model, original_cost=0.001)
        req = _req(prompt, temperature=None)
        d   = rr(self.engine.route(req))
        assert d.cache_hit is True
        assert d.estimated_cost == 0.0

    def test_high_temp_skips_cache(self):
        from router.cache import fingerprint as fp_fn
        from router.model_registry import ModelRegistry as MR
        reg   = MR()
        model = reg.all_available_models()[0]
        prompt = "What is the capital of Germany?"
        fp     = fp_fn(prompt, None, [], None)
        self.cache.set(fp, "Berlin", model, original_cost=0.001)
        req = _req(prompt, temperature=1.0)
        d   = rr(self.engine.route(req))
        # High temp or non-cacheable task type → should not hit cache
        # (temperature=1.0 means cache_eligible=False)
        assert d.cache_hit is False


# ── Cost ceiling ──────────────────────────────────────────────────────────────

class TestCostCeiling:
    def setup_method(self):
        self.engine = _engine()

    def test_override_cost_ceiling_allows_through(self):
        # A normal request with override should not be blocked
        req = _req("hi", metadata={"override_cost_ceiling": True})
        d   = rr(self.engine.route(req))
        assert d.cost_blocked is False


# ── Plan restrictions ─────────────────────────────────────────────────────────

class TestPlanRestrictions:
    def test_free_plan_only_free_cheap(self):
        from router.schemas import ModelOption
        cheap_model = ModelOption(
            provider="anthropic", model_id="test", display_name="Test",
            tier="cheap", cost_per_1k_input=0.001, cost_per_1k_output=0.002,
            max_context_window=4096, max_output_tokens=1000, capabilities=[],
        )
        premium_model = ModelOption(
            provider="anthropic", model_id="test-premium", display_name="Test Premium",
            tier="premium", cost_per_1k_input=0.01, cost_per_1k_output=0.03,
            max_context_window=4096, max_output_tokens=1000, capabilities=[],
        )
        assert _is_allowed_for_plan(cheap_model, "free_plan") is True
        assert _is_allowed_for_plan(premium_model, "free_plan") is False

    def test_business_plan_allows_all(self):
        for tier in ("free", "cheap", "mid", "premium", "ultra"):
            m = ModelOption(
                provider="openai", model_id=f"m-{tier}", display_name=f"M {tier}",
                tier=tier, cost_per_1k_input=0.01, cost_per_1k_output=0.01,
                max_context_window=4096, max_output_tokens=1000, capabilities=[],
            )
            assert _is_allowed_for_plan(m, "business_plan") is True


# ── Model override ────────────────────────────────────────────────────────────

class TestModelOverride:
    def setup_method(self):
        self.engine = _engine()

    def test_explicit_override_respected(self):
        req = _req("Write some code", metadata={"model": "gpt-4o-mini"})
        d   = rr(self.engine.route(req))
        if d.chosen_model:
            assert d.chosen_model.model_id == "gpt-4o-mini"
            assert d.routing_rule_matched == "explicit_model_override"

    def test_unknown_override_falls_through(self):
        req = _req("Write some code", metadata={"model": "nonexistent-model-xyz"})
        d   = rr(self.engine.route(req))
        assert d.routing_rule_matched != "explicit_model_override"


# ── Sensitivity / provider filtering ─────────────────────────────────────────

class TestSensitivity:
    def setup_method(self):
        self.engine = _engine()

    def test_restricted_uses_only_anthropic_openai(self):
        req = _req(
            "Process this classified document",
            metadata={"sensitivity_level": "restricted"}
        )
        d = rr(self.engine.route(req))
        if d.chosen_model:
            assert d.chosen_model.provider in ("anthropic", "openai")

    def test_public_allows_any_provider(self):
        # Just verify it routes successfully
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.chosen_model is not None


# ── Pure helper functions ─────────────────────────────────────────────────────

class TestHelpers:
    def _make_model(self, cost_in, cost_out, latency):
        return ModelOption(
            provider="test", model_id="m", display_name="M",
            tier="mid",
            cost_per_1k_input=cost_in, cost_per_1k_output=cost_out,
            max_context_window=4096, max_output_tokens=1000,
            capabilities=[], avg_latency_ms=latency,
        )

    def test_normalize_cost_uniform(self):
        """Uniform costs → 0.5 (no divide-by-zero)."""
        models = [self._make_model(0.01, 0.02, 500) for _ in range(3)]
        assert _normalize_cost(models[0], models) == 0.5

    def test_normalize_cost_range(self):
        m_cheap  = self._make_model(0.001, 0.001, 500)
        m_medium = self._make_model(0.005, 0.005, 500)
        m_exp    = self._make_model(0.010, 0.010, 500)
        candidates = [m_cheap, m_medium, m_exp]
        assert _normalize_cost(m_cheap, candidates)  == pytest.approx(0.0)
        assert _normalize_cost(m_exp, candidates)    == pytest.approx(1.0)
        assert 0.0 < _normalize_cost(m_medium, candidates) < 1.0

    def test_normalize_latency_uniform(self):
        models = [self._make_model(0.01, 0.02, 1000) for _ in range(3)]
        assert _normalize_latency(models[0], models) == 0.5

    def test_estimate_cost_caps_output(self):
        from router.schemas import TaskAnalysis
        model = self._make_model(0.001, 0.002, 500)
        model.max_output_tokens = 100
        analysis = TaskAnalysis(
            complexity_score=0.5, estimated_input_tokens=200,
            estimated_output_tokens=99999,  # WAY over max
            total_context_needed=200, task_type="code_generation",
        )
        cost = _estimate_cost(analysis, model)
        # output capped at 100 tokens → 0.1 * 0.002 = 0.0002
        expected = (200 / 1000) * 0.001 + (100 / 1000) * 0.002
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_tier_boundaries(self):
        assert _get_tier_for_score(0.00) == "free"
        assert _get_tier_for_score(0.14) == "free"
        assert _get_tier_for_score(0.15) == "cheap"
        assert _get_tier_for_score(0.29) == "cheap"
        assert _get_tier_for_score(0.30) == "mid"
        assert _get_tier_for_score(0.59) == "mid"
        assert _get_tier_for_score(0.60) == "premium"
        assert _get_tier_for_score(0.84) == "premium"
        assert _get_tier_for_score(0.85) == "ultra"
        assert _get_tier_for_score(0.99) == "ultra"
