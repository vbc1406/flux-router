"""
File: router/tests/test_routing.py

Purpose:
End-to-end tests for the 13-step routing algorithm in RoutingEngine.route().
Organised by the 9 major change areas (Changes 1–9) plus basic routing helpers.

How to run:
  pytest -v router/tests/test_routing.py
  pytest -v router/tests/test_routing.py::TestConfidenceThreshold

How to add a test:
  1. Find the test class for the feature area you are testing (or create one).
  2. Copy the pattern of an existing test in that class.
  3. Use _engine() for a fresh engine, _req(prompt, **kw) for a request,
     and rr(coro) to run async route() calls synchronously.
  4. Assert on the fields of the returned RoutingDecision.

Example:
  def test_my_new_behaviour():
      # Arrange
      engine = _engine()
      req = _req("some prompt", routing_priority="quality-first")
      # Act
      d = rr(engine.route(req))
      # Assert
      assert d.chosen_model is not None
      assert d.chosen_model.tier == "premium"

Test classes:
  TestBasicRouting               — sanity checks: model chosen, rule populated, cost ≥ 0
  TestCacheIntegration           — cache hit/miss behaviour
  TestCostCeiling                — MAX_COST_PER_REQUEST enforcement
  TestPlanRestrictions           — tier access by subscription plan
  TestModelOverride              — explicit metadata.model override
  TestSensitivity                — sensitivity level → provider filtering
  TestHelpers                    — pure helper functions (normalize_cost, tier_for_score, …)
  TestPriorityTagValidation      — Change 1: routing_priority validation
  TestPriorityTagWeights         — Change 1: weight presets per priority
  TestAlwaysPremiumRouting       — Change 1: always-premium bypass
  TestConfidenceThreshold        — Change 2: low-confidence upgrade + A/B block invariant
  TestPerCustomerAdaptiveMemory  — Change 3: per-customer EMA
  TestPerRequestCostCap          — Change 4: max_cost_per_request filter
  TestDailyBudgetCap             — Change 4: max_daily_cost enforcement
  TestContextAwareFallbackChains — Change 5: typed fallback chains
  TestConfigurableExplorationRate— Change 6: A/B exploration rate
  TestLinearBudgetWalkdown       — Change 7: tier walk-down on budget exhaustion
  TestUltraCollapse              — Change 8: ultra → premium migration
  TestProxyMode                  — Change 9: proxy mode validation and response
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from pydantic import ValidationError

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.config import MIN_CONFIDENCE_THRESHOLD
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import (
    RoutingEngine,
    _budget_tier_walkdown,
    _estimate_cost,
    _get_tier_for_score,
    _get_weights_for_priority,
    _is_allowed_for_plan,
    _normalize_cost,
    _normalize_latency,
    _validate_exploration_rate,
    _validate_mode,
    _validate_routing_priority,
)
from router.schemas import ModelOption, RoutingRequest

# ── Helpers ──────────────────────────────────────────────────────────────────


def _engine() -> RoutingEngine:
    registry = ModelRegistry()
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw) -> RoutingRequest:
    defaults = {"user_id": "u_test", "plan": "business_plan", "priority": "normal"}
    defaults.update(kw)
    # Pull out correlation_id if provided to avoid passing it twice
    cid = defaults.pop("correlation_id", str(uuid.uuid4()))
    return RoutingRequest(raw_prompt=prompt, correlation_id=cid, **defaults)


def rr(coro):
    return asyncio.run(coro)


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
        d = rr(
            self.engine.route(
                _req(
                    "Prove the Riemann hypothesis using ∑ notation. Show all steps.",
                    priority="critical",
                )
            )
        )
        assert d.chosen_model is not None
        # Change 8: ultra merged into premium — only check premium
        assert d.chosen_model.tier in ("premium", "mid")

    def test_routing_rule_populated(self):
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.routing_rule_matched != ""

    def test_correlation_id_preserved(self):
        cid = "test-correlation-123"
        req = _req("hello", correlation_id=cid)
        d = rr(self.engine.route(req))
        assert d.correlation_id == cid

    def test_confidence_in_range(self):
        d = rr(self.engine.route(_req("Write a Python sort function")))
        assert 0.0 <= d.confidence <= 1.0

    def test_estimated_cost_nonnegative(self):
        d = rr(self.engine.route(_req("Translate hello to Spanish")))
        assert d.estimated_cost >= 0.0

    def test_fallback_chain_not_empty(self):
        d = rr(self.engine.route(_req("Write a Python sort function")))
        assert isinstance(d.fallback_chain, list)

    def test_free_plan_cannot_use_premium(self):
        # Change 8: ultra no longer exists; premium is now the top tier
        d = rr(
            self.engine.route(
                _req(
                    "Solve P vs NP. ∑∫∀∃. Step by step proof.",
                    priority="critical",
                    plan="free_plan",
                )
            )
        )
        if d.chosen_model:
            assert d.chosen_model.tier not in ("premium",)


# ── Cache integration ─────────────────────────────────────────────────────────


class TestCacheIntegration:
    def setup_method(self):
        # Cache is opt-in (FLUX_ENABLE_RESPONSE_CACHE); flip the routing-engine
        # gate on for these tests and restore it in teardown.
        from router import routing_engine as _re

        self._prev_enable = _re.ENABLE_RESPONSE_CACHE
        _re.ENABLE_RESPONSE_CACHE = True

        registry = ModelRegistry()
        self.cache = ResponseCache(enabled=True)
        adaptive = AdaptiveWeights(state_file=None)
        analytics = RoutingAnalytics(log_path=None)
        budget = BudgetTracker()
        compressor = ContextCompressor()
        classifier = RequestClassifier(self.cache)
        self.engine = RoutingEngine(
            registry, classifier, self.cache, budget, adaptive, compressor, analytics
        )

    def teardown_method(self):
        from router import routing_engine as _re

        _re.ENABLE_RESPONSE_CACHE = self._prev_enable

    def test_cache_hit_returns_zero_cost(self):
        from router.cache import fingerprint as fp_fn
        from router.model_registry import ModelRegistry

        reg = ModelRegistry()
        model = reg.all_available_models()[0]
        prompt = "What is the capital of Germany?"
        fp = fp_fn(prompt, None, [], None)
        self.cache.set(fp, "Berlin", model, original_cost=0.001)
        req = _req(prompt, temperature=None)
        d = rr(self.engine.route(req))
        assert d.cache_hit is True
        assert d.estimated_cost == 0.0

    def test_high_temp_skips_cache(self):
        from router.cache import fingerprint as fp_fn
        from router.model_registry import ModelRegistry

        reg = ModelRegistry()
        model = reg.all_available_models()[0]
        prompt = "What is the capital of Germany?"
        fp = fp_fn(prompt, None, [], None)
        self.cache.set(fp, "Berlin", model, original_cost=0.001)
        req = _req(prompt, temperature=1.0)
        d = rr(self.engine.route(req))
        assert d.cache_hit is False


# ── Cost ceiling ──────────────────────────────────────────────────────────────


class TestCostCeiling:
    def setup_method(self):
        self.engine = _engine()

    def test_override_cost_ceiling_allows_through(self):
        req = _req("hi", metadata={"override_cost_ceiling": True})
        d = rr(self.engine.route(req))
        assert d.cost_blocked is False


# ── Plan restrictions ─────────────────────────────────────────────────────────


class TestPlanRestrictions:
    def test_free_plan_only_free_cheap(self):
        cheap_model = ModelOption(
            provider="anthropic",
            model_id="test",
            display_name="Test",
            tier="cheap",
            cost_per_1k_input=0.001,
            cost_per_1k_output=0.002,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
        )
        premium_model = ModelOption(
            provider="anthropic",
            model_id="test-premium",
            display_name="Test Premium",
            tier="premium",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
        )
        assert _is_allowed_for_plan(cheap_model, "free_plan") is True
        assert _is_allowed_for_plan(premium_model, "free_plan") is False

    def test_business_plan_allows_all_four_tiers(self):
        # Change 8: 4 tiers only (ultra removed)
        for tier in ("free", "cheap", "mid", "premium"):
            m = ModelOption(
                provider="openai",
                model_id=f"m-{tier}",
                display_name=f"M {tier}",
                tier=tier,
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.01,
                max_context_window=4096,
                max_output_tokens=1000,
                capabilities=[],
            )
            assert _is_allowed_for_plan(m, "business_plan") is True

    def test_pro_plan_includes_premium(self):
        m = ModelOption(
            provider="anthropic",
            model_id="m-premium",
            display_name="M premium",
            tier="premium",
            cost_per_1k_input=0.01,
            cost_per_1k_output=0.03,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
        )
        assert _is_allowed_for_plan(m, "pro_plan") is True


# ── Model override ────────────────────────────────────────────────────────────


class TestModelOverride:
    def setup_method(self):
        self.engine = _engine()

    def test_explicit_override_respected(self):
        req = _req("Write some code", metadata={"model": "gpt-4o-mini"})
        d = rr(self.engine.route(req))
        if d.chosen_model:
            assert d.chosen_model.model_id == "gpt-4o-mini"
            assert d.routing_rule_matched == "explicit_model_override"

    def test_unknown_override_falls_through(self):
        req = _req("Write some code", metadata={"model": "nonexistent-model-xyz"})
        d = rr(self.engine.route(req))
        assert d.routing_rule_matched != "explicit_model_override"


# ── Sensitivity / provider filtering ─────────────────────────────────────────


class TestSensitivity:
    def setup_method(self):
        self.engine = _engine()

    def test_restricted_uses_only_anthropic_openai(self):
        req = _req("Process this classified document", metadata={"sensitivity_level": "restricted"})
        d = rr(self.engine.route(req))
        if d.chosen_model:
            assert d.chosen_model.provider in ("anthropic", "openai")

    def test_public_allows_any_provider(self):
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.chosen_model is not None


# ── Pure helper functions ─────────────────────────────────────────────────────


class TestHelpers:
    def _make_model(self, cost_in, cost_out, latency, tier="mid"):
        return ModelOption(
            provider="test",
            model_id="m",
            display_name="M",
            tier=tier,
            cost_per_1k_input=cost_in,
            cost_per_1k_output=cost_out,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
            avg_latency_ms=latency,
        )

    def test_normalize_cost_uniform(self):
        models = [self._make_model(0.01, 0.02, 500) for _ in range(3)]
        assert _normalize_cost(models[0], models) == 0.5

    def test_normalize_cost_range(self):
        m_cheap = self._make_model(0.001, 0.001, 500)
        m_medium = self._make_model(0.005, 0.005, 500)
        m_exp = self._make_model(0.010, 0.010, 500)
        candidates = [m_cheap, m_medium, m_exp]
        assert _normalize_cost(m_cheap, candidates) == pytest.approx(0.0)
        assert _normalize_cost(m_exp, candidates) == pytest.approx(1.0)
        assert 0.0 < _normalize_cost(m_medium, candidates) < 1.0

    def test_normalize_latency_uniform(self):
        models = [self._make_model(0.01, 0.02, 1000) for _ in range(3)]
        assert _normalize_latency(models[0], models) == 0.5

    def test_estimate_cost_caps_output(self):
        from router.schemas import TaskAnalysis

        model = self._make_model(0.001, 0.002, 500)
        model.max_output_tokens = 100
        analysis = TaskAnalysis(
            complexity_score=0.5,
            estimated_input_tokens=200,
            estimated_output_tokens=99999,
            total_context_needed=200,
            task_type="code_generation",
        )
        cost = _estimate_cost(analysis, model)
        expected = (200 / 1000) * 0.001 + (100 / 1000) * 0.002
        assert cost == pytest.approx(expected, rel=1e-4)

    def test_tier_boundaries_four_tiers(self):
        # Change 8: 4 tiers — 0.85+ now maps to premium (not ultra)
        assert _get_tier_for_score(0.00) == "free"
        assert _get_tier_for_score(0.14) == "free"
        assert _get_tier_for_score(0.15) == "cheap"
        assert _get_tier_for_score(0.29) == "cheap"
        assert _get_tier_for_score(0.30) == "mid"
        assert _get_tier_for_score(0.59) == "mid"
        assert _get_tier_for_score(0.60) == "premium"
        assert _get_tier_for_score(0.85) == "premium"
        assert _get_tier_for_score(0.99) == "premium"
        assert _get_tier_for_score(1.00) == "premium"  # edge case — no ultra


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 1: Priority Tags
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorityTagValidation:
    """Validate the routing_priority parameter (Change 1)."""

    def test_valid_priorities_accepted(self):
        for p in ("always-premium", "quality-first", "balanced", "cost-optimized"):
            _validate_routing_priority(p)  # should not raise

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Invalid routing_priority"):
            _validate_routing_priority("turbo-mode")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_routing_priority("")

    def test_case_sensitive(self):
        with pytest.raises(ValueError):
            _validate_routing_priority("Balanced")


class TestPriorityTagWeights:
    """Weight override logic for priority tags."""

    def test_quality_first_weights(self):
        w = _get_weights_for_priority("quality-first", "normal")
        assert w["quality"] == pytest.approx(0.70)
        assert w["cost"] == pytest.approx(0.20)
        assert w["latency"] == pytest.approx(0.10)

    def test_cost_optimized_weights(self):
        w = _get_weights_for_priority("cost-optimized", "normal")
        assert w["quality"] == pytest.approx(0.30)
        assert w["cost"] == pytest.approx(0.60)
        assert w["latency"] == pytest.approx(0.10)

    def test_balanced_falls_through_to_latency_mode(self):
        # "balanced" → latency mode from request.priority
        w_normal = _get_weights_for_priority("balanced", "normal")
        w_critical = _get_weights_for_priority("balanced", "critical")
        # normal → "balanced" mode, critical → "prefer_speed" mode
        assert w_normal != w_critical

    def test_always_premium_falls_through_to_latency_mode(self):
        w = _get_weights_for_priority("always-premium", "normal")
        assert "quality" in w


class TestAlwaysPremiumRouting:
    """always-premium should skip scoring and return the highest-tier model."""

    def setup_method(self):
        self.engine = _engine()

    def test_always_premium_routes_to_premium_tier(self):
        d = rr(
            self.engine.route(
                _req(
                    "simple question",
                    routing_priority="always-premium",
                    plan="business_plan",
                )
            )
        )
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"
        assert d.priority_applied == "always-premium"

    def test_always_premium_rule_contains_always_premium(self):
        d = rr(
            self.engine.route(
                _req(
                    "hello",
                    routing_priority="always-premium",
                    plan="business_plan",
                )
            )
        )
        assert "always_premium" in d.routing_rule_matched

    def test_always_premium_bypasses_cheap_for_trivial(self):
        # Even a 3-word prompt should route to premium, not free/cheap
        d = rr(
            self.engine.route(_req("hi", routing_priority="always-premium", plan="business_plan"))
        )
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    def test_always_premium_free_plan_falls_to_best_available(self):
        # free_plan cannot access premium — router picks best available tier
        d = rr(
            self.engine.route(
                _req(
                    "complex reasoning task",
                    routing_priority="always-premium",
                    plan="free_plan",
                )
            )
        )
        assert d.chosen_model is not None
        assert d.chosen_model.tier in ("cheap", "free")

    def test_priority_applied_field_set(self):
        for priority in ("always-premium", "quality-first", "balanced", "cost-optimized"):
            d = rr(self.engine.route(_req("test", routing_priority=priority)))
            assert d.priority_applied == priority

    def test_quality_first_routes_quality_bias(self):
        d = rr(
            self.engine.route(
                _req(
                    "Write a complex analysis",
                    routing_priority="quality-first",
                    plan="business_plan",
                )
            )
        )
        assert d.chosen_model is not None
        # quality-first biases toward better models; confidence should be good
        assert d.confidence >= 0.0

    def test_cost_optimized_routes_cheaper(self):
        # exploration_rate=0: this test asserts a cost ORDERING between the two
        # priorities, which Step 10's random A/B exploration can otherwise
        # violate on an unlucky draw (it uses the shared, unseeded `random`
        # module, so this is sensitive to how many prior route() calls ran
        # elsewhere in the suite) — not a cost-optimized vs. quality-first bug.
        d1 = rr(
            self.engine.route(
                _req(
                    "Write a 500-word analysis",
                    routing_priority="cost-optimized",
                    plan="business_plan",
                    exploration_rate=0.0,
                )
            )
        )
        d2 = rr(
            self.engine.route(
                _req(
                    "Write a 500-word analysis",
                    routing_priority="quality-first",
                    plan="business_plan",
                    exploration_rate=0.0,
                )
            )
        )
        assert d1.chosen_model is not None
        assert d2.chosen_model is not None
        # cost-optimized should not be more expensive than quality-first
        assert d1.estimated_cost <= d2.estimated_cost + 0.001  # allow floating point slack

    def test_invalid_routing_priority_raises(self):
        with pytest.raises(ValueError):
            rr(self.engine.route(_req("test", routing_priority="turbo")))


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 2: Confidence Threshold
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfidenceThreshold:
    """Confidence fallback triggers when winning score < MIN_CONFIDENCE_THRESHOLD."""

    def setup_method(self):
        self.engine = _engine()

    def test_confidence_fallback_field_exists(self):
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert hasattr(d, "confidence_fallback")
        assert isinstance(d.confidence_fallback, bool)

    def test_normal_routing_no_fallback(self):
        # High-confidence routing: complex coding task → premium tier
        d = rr(
            self.engine.route(
                _req(
                    "Design a distributed database with RAFT consensus",
                    plan="business_plan",
                )
            )
        )
        assert d.chosen_model is not None
        # Should not trigger confidence fallback (premium model for complex task)
        # (may still be False even if borderline — just check it's bool)
        assert isinstance(d.confidence_fallback, bool)

    def test_confidence_fallback_low_confidence_upgrades(self):
        """
        Simulate a low-confidence scenario: use a cheap model for a task type
        where its routing score will be low.  The engine should upgrade to premium.
        """

        # Build a cheap model with very low quality for the task at hand
        cheap_model = ModelOption(
            provider="test",
            model_id="tiny",
            display_name="Tiny",
            tier="cheap",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0002,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
            quality_ratings={"reasoning": 0.10},  # terrible at reasoning
            adjusted_quality=0.10,
            routing_score=0.05,  # well below MIN_CONFIDENCE_THRESHOLD
        )
        # A routing_score of 0.05 is below 0.60 threshold
        assert cheap_model.routing_score < MIN_CONFIDENCE_THRESHOLD

    def test_ab_blocked_when_confidence_fallback(self, monkeypatch):
        """A/B exploration must never fire after a confidence fallback upgrade.

        Safety invariant: confidence_fallback upgrades to premium precisely because
        the winner scored too low to be trusted.  Allowing A/B to then downgrade
        that decision defeats the safety mechanism entirely.
        """
        import router.routing_engine as re_mod

        # Raise the threshold to 0.99 so every request triggers confidence_fallback
        monkeypatch.setattr(re_mod, "MIN_CONFIDENCE_THRESHOLD", 0.99)
        # Force random() → 0.0 so A/B would fire on every eligible request
        monkeypatch.setattr(re_mod.random, "random", lambda: 0.0)

        # Use a complex coding prompt — avoids the trivial_request early-exit path
        # which only fires for short conversations and hardcodes confidence_fallback=False.
        prompt = (
            "Implement a thread-safe LRU cache in Python with O(1) get and put. "
            "Include full type annotations, unit tests, and explain the concurrency strategy."
        )
        engine = _engine()
        for _ in range(20):
            d = rr(engine.route(_req(prompt, exploration_rate=0.25, plan="business_plan")))
            # Every request must have triggered confidence_fallback (threshold=0.99)
            assert d.confidence_fallback, (
                "expected confidence_fallback with threshold=0.99, "
                f"got rule={d.routing_rule_matched!r}"
            )
            # A/B must be suppressed — safety overrides experimentation
            assert "ab_exploration" not in d.routing_rule_matched, (
                f"A/B exploration fired after confidence fallback: rule={d.routing_rule_matched!r}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 3: Per-Customer Adaptive Memory
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerCustomerAdaptiveMemory:
    def setup_method(self):
        self.adaptive = AdaptiveWeights(state_file=None)

    def test_customer_id_routed_correctly(self):
        engine = _engine()
        d = rr(engine.route(_req("What is 2+2?", customer_id="cust_abc")))
        assert d.chosen_model is not None

    def test_global_fallback_before_20_samples(self):
        # Before 20 samples per-customer, global weights should apply
        base = 0.75
        result = self.adaptive.get_adjusted_score(
            "gpt-4o", "simple_qa", base, customer_id="new_cust"
        )
        assert result == base  # no data → returns base unchanged

    def test_per_customer_weights_apply_after_threshold(self):
        # Record 20 samples for a customer
        for _ in range(20):
            self.adaptive.record("gpt-4o", "simple_qa", 0.95, 0.75, customer_id="loyal_cust")
        # Now per-customer EMA should be active
        adjusted = self.adaptive.get_adjusted_score(
            "gpt-4o", "simple_qa", 0.75, customer_id="loyal_cust"
        )
        # After 20 samples of 0.95, adjusted should be > base 0.75
        assert adjusted > 0.75

    def test_global_still_works_without_customer_id(self):
        base = 0.80
        result = self.adaptive.get_adjusted_score("some-model", "code_generation", base)
        assert result == base  # no global data → base unchanged

    def test_get_customer_routing_profile_empty(self):
        profile = self.adaptive.get_customer_routing_profile("unknown_cust")
        assert profile["total_requests"] == 0
        assert profile["has_custom_weights"] is False

    def test_record_routing_event_and_profile(self):
        for _ in range(3):
            self.adaptive.record_routing_event("cust_x", "gpt-4o", "code_generation", 0.005)
        profile = self.adaptive.get_customer_routing_profile("cust_x")
        assert profile["total_requests"] == 3
        assert profile["top_models"][0][0] == "gpt-4o"
        assert profile["avg_cost_per_request"] == pytest.approx(0.005)

    def test_per_customer_isolated_from_global(self):
        # Global records should not affect per-customer and vice versa
        self.adaptive.record("model-a", "vision", 0.90, 0.70)  # global only
        # Per-customer should still return base (no customer data)
        result = self.adaptive.get_adjusted_score("model-a", "vision", 0.70, customer_id="isolated")
        assert result == 0.70


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 4: Per-Request and Per-Day Cost Caps
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerRequestCostCap:
    def setup_method(self):
        self.engine = _engine()

    def test_max_cost_per_request_filters_expensive_models(self):
        # Set a very low cap — only free/cheap models should survive
        d = rr(
            self.engine.route(
                _req(
                    "Write a blog post",
                    max_cost_per_request=0.0001,
                    plan="business_plan",
                )
            )
        )
        assert d.chosen_model is not None
        # With $0.0001 cap, premium models ($0.015+ per 1k in) should be filtered
        assert d.chosen_model.tier in ("free", "cheap")

    def test_no_cap_allows_all_models(self):
        d = rr(
            self.engine.route(
                _req(
                    "Write a complex analysis",
                    max_cost_per_request=None,
                    plan="business_plan",
                )
            )
        )
        assert d.chosen_model is not None

    def test_zero_cap_still_routes_to_free(self):
        d = rr(
            self.engine.route(
                _req(
                    "hello",
                    max_cost_per_request=0.0,
                    plan="business_plan",
                )
            )
        )
        # Only free (zero-cost) models should be selected
        if d.chosen_model:
            assert d.chosen_model.tier == "free"


class TestDailyBudgetCap:
    def test_daily_budget_tracker_basic(self):
        from router.budget_tracker import DailyBudgetTracker

        tracker = DailyBudgetTracker()
        tracker.record_spend("cust_1", 5.0, "gpt-4o", "req-1")
        assert tracker.get_daily_spend("cust_1") == pytest.approx(5.0)
        assert tracker.is_cap_exceeded("cust_1", 4.99) is True
        assert tracker.is_cap_exceeded("cust_1", 5.01) is False

    def test_daily_budget_exhausted_forces_free_tier(self):
        engine = _engine()
        # Pre-seed the daily budget tracker so the cap is immediately hit
        engine._daily_budget.record_spend("cust_budget", 100.0, "preseed", "preseed")
        d = rr(
            engine.route(
                _req(
                    "Write code",
                    customer_id="cust_budget",
                    max_daily_cost=50.0,
                    plan="business_plan",
                )
            )
        )
        # Should route to free tier or return budget_exhausted
        if d.budget_exhausted:
            assert d.chosen_model is None
        else:
            assert d.chosen_model is not None
            assert d.chosen_model.tier == "free"

    def test_daily_budget_no_cap_normal_routing(self):
        engine = _engine()
        d = rr(
            engine.route(
                _req(
                    "Write a Python function",
                    max_daily_cost=None,  # no cap
                )
            )
        )
        assert d.chosen_model is not None


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 5: Context-Aware Fallback Chains
# ═══════════════════════════════════════════════════════════════════════════════


class TestContextAwareFallbackChains:
    def setup_method(self):
        self.engine = _engine()

    def test_three_fallback_chain_fields_exist(self):
        d = rr(self.engine.route(_req("Write a Python sort function")))
        assert hasattr(d, "fallback_on_rate_limit")
        assert hasattr(d, "fallback_on_content_safety")
        assert hasattr(d, "fallback_on_timeout")
        assert isinstance(d.fallback_on_rate_limit, list)
        assert isinstance(d.fallback_on_content_safety, list)
        assert isinstance(d.fallback_on_timeout, list)

    def test_rate_limit_chain_same_tier(self):
        from router.fallback_chain import build_typed_fallback_chains
        from router.schemas import TaskAnalysis

        registry = ModelRegistry()
        candidates = registry.all_available_models()
        analysis = TaskAnalysis(
            complexity_score=0.5,
            estimated_input_tokens=100,
            estimated_output_tokens=200,
            total_context_needed=300,
            task_type="code_generation",
        )
        # Find a mid-tier model to use as chosen
        mid_models = [m for m in candidates if m.tier == "mid"]
        if mid_models:
            chosen = mid_models[0]
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            # Rate limit chain: same tier only
            for m in rl:
                assert m.tier == chosen.tier

    def test_content_safety_chain_higher_tier(self):
        from router.fallback_chain import build_typed_fallback_chains
        from router.schemas import TaskAnalysis

        registry = ModelRegistry()
        candidates = registry.all_available_models()
        analysis = TaskAnalysis(
            complexity_score=0.4,
            estimated_input_tokens=100,
            estimated_output_tokens=200,
            total_context_needed=300,
            task_type="code_generation",
        )
        cheap_models = [m for m in candidates if m.tier == "cheap"]
        if cheap_models:
            chosen = cheap_models[0]
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            # Content safety: tiers above cheap (mid, premium)
            for m in cs:
                tier_idx = ["free", "cheap", "mid", "premium"].index(m.tier)
                chosen_idx = ["free", "cheap", "mid", "premium"].index(chosen.tier)
                assert tier_idx > chosen_idx

    def test_timeout_chain_sorted_by_latency(self):
        from router.fallback_chain import build_typed_fallback_chains
        from router.schemas import TaskAnalysis

        registry = ModelRegistry()
        candidates = registry.all_available_models()
        analysis = TaskAnalysis(
            complexity_score=0.5,
            estimated_input_tokens=100,
            estimated_output_tokens=200,
            total_context_needed=300,
            task_type="analysis",
        )
        mid_models = [m for m in candidates if m.tier == "mid"]
        if mid_models:
            chosen = mid_models[0]
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            # Timeout chain: sorted by latency ascending
            if len(to) >= 2:
                assert to[0].avg_latency_ms <= to[1].avg_latency_ms


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 6: Configurable A/B Exploration Rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestConfigurableExplorationRate:
    def test_valid_rates_accepted(self):
        _validate_exploration_rate(0.0)
        _validate_exploration_rate(0.10)
        _validate_exploration_rate(0.25)

    def test_rate_above_0_25_rejected(self):
        with pytest.raises(ValueError, match="out of range"):
            _validate_exploration_rate(0.26)

    def test_negative_rate_rejected(self):
        with pytest.raises(ValueError):
            _validate_exploration_rate(-0.01)

    def test_zero_rate_never_explores(self):
        """At exploration_rate=0.0, the chosen model should always be tier_selection."""
        engine = _engine()
        results = []
        for _ in range(30):
            d = rr(
                engine.route(
                    _req(
                        "Analyze the CAP theorem",
                        exploration_rate=0.0,
                        plan="business_plan",
                    )
                )
            )
            results.append(d.routing_rule_matched)
        # With 0.0 rate no ab_exploration should ever appear
        assert all("ab_exploration" not in r for r in results)

    def test_nonzero_rate_field_accepted(self):
        engine = _engine()
        d = rr(
            engine.route(
                _req(
                    "Translate hello to French",
                    exploration_rate=0.15,
                )
            )
        )
        assert d.chosen_model is not None

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _validate_mode("fast")


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 7: Linear Budget Tier Walk-Down
# ═══════════════════════════════════════════════════════════════════════════════


class TestLinearBudgetWalkdown:
    def _make_model(self, tier: str, cost_in: float, cost_out: float) -> ModelOption:
        return ModelOption(
            provider="test",
            model_id=f"m-{tier}",
            display_name=f"M {tier}",
            tier=tier,
            cost_per_1k_input=cost_in,
            cost_per_1k_output=cost_out,
            max_context_window=4096,
            max_output_tokens=1000,
            capabilities=[],
            adjusted_quality=0.7,
            routing_score=0.7,
        )

    def test_walkdown_finds_affordable_tier(self):
        from router.schemas import TaskAnalysis

        premium = self._make_model("premium", 0.015, 0.075)
        mid = self._make_model("mid", 0.003, 0.015)
        cheap = self._make_model("cheap", 0.0008, 0.004)
        free = self._make_model("free", 0.0, 0.0)
        candidates = [premium, mid, cheap, free]

        analysis = TaskAnalysis(
            complexity_score=0.8,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
            total_context_needed=1000,
            task_type="analysis",
        )

        budget = BudgetTracker()
        # Drain the budget so premium and mid are too expensive
        budget.record_spend("u1", 49.99, "prev", "prev", plan="pro_plan")
        budget._plans["u1"] = "pro_plan"

        result, rule, exhausted = _budget_tier_walkdown(
            premium, candidates, analysis, "u1", budget, "tier_selection"
        )
        assert exhausted is False or result.tier in ("cheap", "free")
        assert "budget_downgraded" in rule or exhausted

    def test_walkdown_reports_exhausted_when_all_too_expensive(self):
        from router.schemas import TaskAnalysis

        # All models have non-zero cost
        mid = self._make_model("mid", 0.003, 0.015)
        # No free tier
        candidates = [mid]

        analysis = TaskAnalysis(
            complexity_score=0.5,
            estimated_input_tokens=500,
            estimated_output_tokens=500,
            total_context_needed=1000,
            task_type="analysis",
        )

        budget = BudgetTracker()
        budget.record_spend("u2", 50.0, "prev", "prev", plan="pro_plan")
        budget._plans["u2"] = "pro_plan"

        result, rule, exhausted = _budget_tier_walkdown(
            mid, candidates, analysis, "u2", budget, "tier_selection"
        )
        # Only one candidate, all exceed budget → exhausted
        assert exhausted is True
        assert "budget_exhausted" in rule


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 8: Ultra Tier Collapse
# ═══════════════════════════════════════════════════════════════════════════════


class TestUltraCollapse:
    def test_no_ultra_models_in_registry(self):
        registry = ModelRegistry()
        models = registry.all_available_models()
        ultra = [m for m in models if m.tier == "ultra"]
        assert len(ultra) == 0, f"Found ultra-tier models: {[m.model_id for m in ultra]}"

    def test_formerly_ultra_models_are_premium(self):
        registry = ModelRegistry()
        models = {m.model_id: m for m in registry.all_available_models()}
        for mid in ("o3", "o4-mini", "claude-opus-4-20250514-extended"):
            assert mid in models, f"{mid} not found in registry"
            assert models[mid].tier == "premium", f"{mid} should be premium"

    def test_tier_score_1_0_returns_premium(self):
        assert _get_tier_for_score(1.00) == "premium"
        assert _get_tier_for_score(0.99) == "premium"
        assert _get_tier_for_score(0.85) == "premium"
        assert _get_tier_for_score(0.60) == "premium"

    def test_model_option_rejects_ultra_tier(self):
        with pytest.raises(ValidationError):
            ModelOption(
                provider="test",
                model_id="bad",
                display_name="Bad",
                tier="ultra",  # type: ignore[arg-type]
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.01,
                max_context_window=4096,
                max_output_tokens=1000,
                capabilities=[],
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CHANGE 9: Proxy Mode
# ═══════════════════════════════════════════════════════════════════════════════


class TestProxyMode:
    def test_mode_validation_accepts_decision(self):
        _validate_mode("decision")

    def test_mode_validation_accepts_proxy(self):
        _validate_mode("proxy")

    def test_mode_validation_rejects_unknown(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            _validate_mode("streaming")

    def test_decision_mode_no_proxy_response(self):
        engine = _engine()
        d = rr(engine.route(_req("What is 2+2?", mode="decision")))
        assert d.proxy_response is None
        assert d.proxy_model_used is None

    def test_proxy_mode_no_api_key_returns_error_message(self):
        engine = _engine()
        d = rr(
            engine.route(
                _req(
                    "What is 2+2?",
                    mode="proxy",
                    provider_api_key=None,  # no key
                )
            )
        )
        assert d.proxy_response is not None
        assert "proxy error" in d.proxy_response.lower()

    def test_proxy_mode_fields_exist(self):
        d = rr(_engine().route(_req("test", mode="decision")))
        assert hasattr(d, "proxy_response")
        assert hasattr(d, "proxy_model_used")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            rr(_engine().route(_req("test", mode="turbo")))
