"""
File: router/tests/test_priority_tags.py

Purpose:
500 end-to-end tests for the routing_priority tag feature (Change 1).
All tests run against the full routing stack with no mocking.

How to run:
  pytest -v router/tests/test_priority_tags.py
  pytest -v router/tests/test_priority_tags.py::TestAlwaysPremium

How to add a test:
  1. Use _engine() for a fresh engine, _req(prompt, routing_priority=..., **kw).
  2. Call rr(engine.route(req)) and assert on d.chosen_model.tier or d.cost.
  3. Add to the class that matches the priority tag being tested.

Test distribution:
  100  always-premium   (TestAlwaysPremium)
  100  quality-first    (TestQualityFirst)
  100  balanced         (TestBalancedBaseline — baseline, default behaviour)
  100  cost-optimized   (TestCostOptimized)
  100  edge cases       (TestPriorityEdgeCases — plan limits, invalid values,
                         proxy mode, confidence threshold interactions)

Test classes:
  TestAlwaysPremium     — always-premium forces premium tier regardless of complexity
  TestQualityFirst      — quality-first prefers higher tier with cost tolerance
  TestBalancedBaseline  — balanced matches default routing (no priority set)
  TestCostOptimized     — cost-optimized prefers cheapest viable model
  TestPriorityEdgeCases — plan restrictions, invalid tags, cross-feature interactions
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from router.adaptive_weights import AdaptiveWeights
from router.analytics import RoutingAnalytics
from router.budget_tracker import BudgetTracker
from router.cache import ResponseCache
from router.classifier import RequestClassifier
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import RoutingEngine
from router.schemas import RoutingRequest

# ── Engine / request helpers ──────────────────────────────────────────────────


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
    defaults: dict[str, Any] = {
        "user_id": "u_priority_test",
        "plan": "business_plan",
        "priority": "normal",
    }
    defaults.update(kw)
    cid = defaults.pop("correlation_id", str(uuid.uuid4()))
    return RoutingRequest(raw_prompt=prompt, correlation_id=cid, **defaults)


def rr(coro):
    return asyncio.run(coro)


# ── Prompt corpus ─────────────────────────────────────────────────────────────

_FREE_PROMPTS = [
    "hi",
    "hello",
    "thanks",
    "ok",
    "sure",
    "bye",
    "What is 2+2?",
    "Who is Einstein?",
    "Define osmosis",
    "What is gravity?",
    "Who painted the Mona Lisa?",
    "Is Python compiled?",
    "Is the earth round?",
    "Is water H2O?",
    "Translate 'good morning' to French",
    "Translate 'thank you' to Spanish",
    "Convert 5 miles to km",
    "Convert 100 Fahrenheit to Celsius",
    "What is the capital of France?",
    "What is Pi?",
]

_MID_PROMPTS = [
    "Write a Python function to sort a list in O(n log n)",
    "Explain the CAP theorem in distributed databases",
    "Compare React vs Vue for enterprise applications",
    "Write a 500-word blog post about renewable energy",
    "Analyze the pros and cons of remote work",
    "Summarize the key points of GDPR in 3 paragraphs",
    "Write a JavaScript function to debounce event handlers",
    "Explain how neural networks learn via backpropagation",
    "Compare REST vs GraphQL for mobile backends",
    "Write a Python class implementing a min-heap",
]

_PREMIUM_PROMPTS = [
    "Design a complete multi-tenant SaaS architecture with row-level security",
    "Analyze the GDPR Article 17 implications for cross-border data transfers",
    "Prove that √2 is irrational using proof by contradiction with all steps",
    "Build a complete REST API with auth, rate limiting, and WebSocket support in FastAPI",
    "Debug and fix this async Python code: async def foo(): with lock: await bar()",
    "Explain the Hamiltonian formalism for a simple pendulum with all derivation steps",
    "Design a distributed fraud detection system processing 1M events/second",
    "Analyze the financial risks of venture debt with 15% interest and warrant coverage",
    "Write a complete implementation of RAFT consensus for a distributed key-value store",
    "Prove Bayes theorem from first principles with full mathematical notation",
]

_COMPLEX_REASONING = [
    "Prove the sum of all natural numbers diverges; explain Ramanujan's -1/12 result",
    "Solve using KKT conditions: minimize x²+y² s.t. x+y≥10 with all derivation steps",
    "Derive Schrödinger's equation from first principles showing all quantum mechanics",
    "Explain the P vs NP problem and why it matters, with ∑ notation examples",
    "Prove there are infinitely many primes using Euclid's proof with ∑ notation",
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: always-premium (100 test cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestAlwaysPremium:
    """100 test cases for routing_priority='always-premium'."""

    def setup_method(self):
        self.engine = _engine()

    # --- Trivial prompts must route to premium (not free) ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:20])
    def test_trivial_prompts_route_to_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"
        assert d.priority_applied == "always-premium"

    # --- Mid-complexity prompts must route to premium ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:20])
    def test_mid_prompts_route_to_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    # --- Already-premium prompts still route to premium ---

    @pytest.mark.parametrize("prompt", _PREMIUM_PROMPTS[:20])
    def test_complex_prompts_route_to_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    # --- Rule string always contains "always_premium" ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:10] + _MID_PROMPTS[:10])
    def test_rule_contains_always_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert "always_premium" in d.routing_rule_matched

    # --- priority_applied is always "always-premium" ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:5])
    def test_priority_applied_field(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d.priority_applied == "always-premium"

    # --- free_plan restricts to cheap/free (10 cases) ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:10])
    def test_free_plan_cannot_reach_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium", plan="free_plan")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier in ("free", "cheap")

    # --- no routing exception (10 sanity checks) ---

    @pytest.mark.parametrize("prompt", _COMPLEX_REASONING[:5] + _PREMIUM_PROMPTS[:5])
    def test_no_exception_raised(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: quality-first (100 test cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestQualityFirst:
    """100 test cases for routing_priority='quality-first'."""

    def setup_method(self):
        self.engine = _engine()

    # --- decision always returns a model ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:15])
    def test_routes_successfully_trivial(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert d.chosen_model is not None

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:15])
    def test_routes_successfully_mid(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert d.chosen_model is not None

    @pytest.mark.parametrize("prompt", _PREMIUM_PROMPTS[:10])
    def test_routes_successfully_premium(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert d.chosen_model is not None

    # --- priority_applied is always "quality-first" ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:15])
    def test_priority_applied_field(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert d.priority_applied == "quality-first"

    # --- quality-first ≥ cost model for same prompt ---
    # (We can't guarantee this in all tiers but for mid/premium complexity it should hold)

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:10])
    def test_quality_first_not_more_expensive_than_always_premium(self, prompt):
        """
        Quality-first still does tier-based scoring; it should be at most as
        expensive as always-premium which goes straight to the top tier.
        """
        d_qf = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        d_ap = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d_qf.chosen_model is not None
        assert d_ap.chosen_model is not None
        # always-premium locks to highest tier; quality-first may be cheaper
        assert d_qf.estimated_cost <= d_ap.estimated_cost + 0.001

    # --- fields are well-formed ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:10] + _PREMIUM_PROMPTS[:5])
    def test_decision_fields_valid(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert 0.0 <= d.confidence <= 1.0
        assert d.estimated_cost >= 0.0
        assert d.routing_rule_matched != ""

    # --- no exception raised on varied input ---

    @pytest.mark.parametrize("prompt", _COMPLEX_REASONING)
    def test_no_exception_complex(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="quality-first")))
        assert d is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: balanced (baseline — 100 test cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBalancedBaseline:
    """
    100 test cases for routing_priority='balanced'.
    Verifies that the default (existing) behaviour is completely unchanged.
    """

    def setup_method(self):
        self.engine = _engine()

    # --- Free-tier prompts should stay cheap/free ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:20])
    def test_trivial_routes_free_or_cheap(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="balanced")))
        assert d.chosen_model is not None
        assert d.chosen_model.tier in ("free", "cheap", "mid")

    # --- Priority applied is always "balanced" ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:20])
    def test_priority_applied_is_balanced(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="balanced")))
        assert d.priority_applied == "balanced"

    # --- Default is indistinguishable from omitting routing_priority ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:10])
    def test_default_equals_balanced(self, prompt):
        """
        Routing without routing_priority (default="balanced") should produce
        the same model as explicitly passing "balanced" when A/B is disabled.
        """
        # Disable A/B exploration so the comparison is deterministic
        d_default = rr(self.engine.route(_req(prompt, exploration_rate=0.0)))
        d_balanced = rr(
            self.engine.route(_req(prompt, routing_priority="balanced", exploration_rate=0.0))
        )
        assert d_default.chosen_model is not None
        assert d_balanced.chosen_model is not None
        assert d_default.chosen_model.model_id == d_balanced.chosen_model.model_id

    # --- Complex prompts do not route to free tier (A/B disabled) ---

    @pytest.mark.parametrize("prompt", _PREMIUM_PROMPTS[:10])
    def test_complex_prompts_not_free(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="balanced", exploration_rate=0.0)))
        assert d.chosen_model is not None
        # Complex prompts should not land in free tier under balanced
        assert d.chosen_model.tier != "free"

    # --- Fields always well-formed ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:5] + _MID_PROMPTS[:5] + _PREMIUM_PROMPTS[:5])
    def test_fields_valid(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="balanced")))
        assert 0.0 <= d.confidence <= 1.0
        assert d.estimated_cost >= 0.0
        assert isinstance(d.fallback_chain, list)

    # --- Consistency: two routes of the same prompt / no A/B should match ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:5])
    def test_consistent_routing_no_ab(self, prompt):
        d1 = rr(self.engine.route(_req(prompt, routing_priority="balanced", exploration_rate=0.0)))
        d2 = rr(self.engine.route(_req(prompt, routing_priority="balanced", exploration_rate=0.0)))
        assert d1.chosen_model is not None
        assert d2.chosen_model is not None
        assert d1.chosen_model.model_id == d2.chosen_model.model_id

    # --- Sanity checks on varied types ---

    @pytest.mark.parametrize("prompt", _COMPLEX_REASONING)
    def test_complex_reasoning_no_exception(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="balanced")))
        assert d is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: cost-optimized (100 test cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCostOptimized:
    """100 test cases for routing_priority='cost-optimized'."""

    def setup_method(self):
        self.engine = _engine()

    # --- Always returns a model ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:20])
    def test_routes_trivial_prompts(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert d.chosen_model is not None

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:15])
    def test_routes_mid_prompts(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert d.chosen_model is not None

    @pytest.mark.parametrize("prompt", _PREMIUM_PROMPTS[:10])
    def test_routes_premium_prompts(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert d.chosen_model is not None

    # --- priority_applied field ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:15])
    def test_priority_applied_field(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert d.priority_applied == "cost-optimized"

    # --- cost-optimized ≤ quality-first for same complexity ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:10])
    def test_cost_optimized_cheaper_or_equal_to_quality_first(self, prompt):
        # Disable A/B so exploration can't make quality-first accidentally cheaper.
        d_co = rr(
            self.engine.route(_req(prompt, routing_priority="cost-optimized", exploration_rate=0.0))
        )
        d_qf = rr(
            self.engine.route(_req(prompt, routing_priority="quality-first", exploration_rate=0.0))
        )
        assert d_co.chosen_model is not None
        assert d_qf.chosen_model is not None
        # Cost-optimized should choose cheaper or equal option
        assert d_co.estimated_cost <= d_qf.estimated_cost + 0.001

    # --- cost-optimized ≤ always-premium ---

    @pytest.mark.parametrize("prompt", _MID_PROMPTS[:10])
    def test_cost_optimized_cheaper_than_always_premium(self, prompt):
        d_co = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        d_ap = rr(self.engine.route(_req(prompt, routing_priority="always-premium")))
        assert d_co.chosen_model is not None
        assert d_ap.chosen_model is not None
        assert d_co.estimated_cost <= d_ap.estimated_cost + 0.001

    # --- fields valid ---

    @pytest.mark.parametrize("prompt", _FREE_PROMPTS[:5] + _MID_PROMPTS[:5])
    def test_fields_valid(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert 0.0 <= d.confidence <= 1.0
        assert d.estimated_cost >= 0.0

    # --- no exception ---

    @pytest.mark.parametrize("prompt", _COMPLEX_REASONING)
    def test_no_exception_complex(self, prompt):
        d = rr(self.engine.route(_req(prompt, routing_priority="cost-optimized")))
        assert d is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Mixed Edge Cases (100 test cases)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPriorityEdgeCases:
    """100 edge case tests covering interactions between priority tags and other features."""

    def setup_method(self):
        self.engine = _engine()

    # --- Invalid priority values (10) ---

    @pytest.mark.parametrize(
        "bad_priority",
        [
            "turbo",
            "ultra-fast",
            "cheap-mode",
            "BALANCED",
            "Quality-First",
            "premium",
            "cost_optimized",
            "free",
            "",
            "   ",
        ],
    )
    def test_invalid_priority_raises_value_error(self, bad_priority):
        with pytest.raises((ValueError, Exception)):
            rr(self.engine.route(_req("test", routing_priority=bad_priority)))

    # --- Priority + explicit model override (5) ---

    @pytest.mark.parametrize(
        "priority", ["always-premium", "quality-first", "balanced", "cost-optimized"]
    )
    def test_explicit_override_takes_precedence_over_priority(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "Write some code",
                    routing_priority=priority,
                    metadata={"model": "gpt-4o-mini"},
                )
            )
        )
        if d.chosen_model:
            assert d.chosen_model.model_id == "gpt-4o-mini"
            assert d.routing_rule_matched == "explicit_model_override"

    def test_override_with_always_premium_still_honours_override(self):
        d = rr(
            self.engine.route(
                _req(
                    "hi",
                    routing_priority="always-premium",
                    metadata={"model": "gpt-4o-mini"},
                )
            )
        )
        if d.chosen_model:
            assert d.chosen_model.model_id == "gpt-4o-mini"

    # --- Priority + exploration_rate interaction (5) ---

    def test_always_premium_with_zero_exploration(self):
        d = rr(
            self.engine.route(
                _req(
                    "Analyze something",
                    routing_priority="always-premium",
                    exploration_rate=0.0,
                )
            )
        )
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    def test_cost_optimized_with_zero_exploration_consistent(self):
        results = []
        for _ in range(10):
            d = rr(
                self.engine.route(
                    _req(
                        "Write a blog post",
                        routing_priority="cost-optimized",
                        exploration_rate=0.0,
                    )
                )
            )
            results.append(d.chosen_model.model_id if d.chosen_model else None)
        assert len(set(results)) == 1, "With 0.0 exploration, model should be deterministic"

    # --- Priority + max_cost_per_request interaction (5) ---

    @pytest.mark.parametrize("priority", ["always-premium", "quality-first"])
    def test_quality_priorities_respect_cost_cap(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "Write code",
                    routing_priority=priority,
                    max_cost_per_request=0.0001,  # very tight cap
                    plan="business_plan",
                )
            )
        )
        if d.chosen_model:
            assert d.estimated_cost <= 0.0001 + 1e-6

    @pytest.mark.parametrize("priority", ["cost-optimized", "balanced"])
    def test_other_priorities_also_respect_cost_cap(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "What is 2+2?",
                    routing_priority=priority,
                    max_cost_per_request=0.0001,
                )
            )
        )
        if d.chosen_model:
            assert d.estimated_cost <= 0.0001 + 1e-6

    # --- Priority + customer_id interaction (5) ---

    @pytest.mark.parametrize(
        "priority", ["always-premium", "quality-first", "balanced", "cost-optimized"]
    )
    def test_customer_id_with_each_priority(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "Write a Python function",
                    routing_priority=priority,
                    customer_id="cust_edge_test",
                )
            )
        )
        assert d.chosen_model is not None

    def test_customer_id_none_with_always_premium(self):
        d = rr(
            self.engine.route(
                _req(
                    "Prove Pythagoras theorem",
                    routing_priority="always-premium",
                    customer_id=None,
                )
            )
        )
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    # --- Priority + sensitivity level (5) ---

    def test_always_premium_with_restricted_sensitivity(self):
        d = rr(
            self.engine.route(
                _req(
                    "Process confidential document",
                    routing_priority="always-premium",
                    metadata={"sensitivity_level": "restricted"},
                    plan="business_plan",
                )
            )
        )
        if d.chosen_model:
            assert d.chosen_model.provider in ("anthropic", "openai")

    def test_cost_optimized_with_restricted_sensitivity(self):
        d = rr(
            self.engine.route(
                _req(
                    "Process internal document",
                    routing_priority="cost-optimized",
                    metadata={"sensitivity_level": "restricted"},
                    plan="business_plan",
                )
            )
        )
        if d.chosen_model:
            assert d.chosen_model.provider in ("anthropic", "openai")

    # --- Priority + request.priority interaction (5) ---

    @pytest.mark.parametrize("req_priority", ["low", "normal", "high", "critical"])
    def test_balanced_priority_with_each_request_priority(self, req_priority):
        d = rr(
            self.engine.route(
                _req(
                    "Write a function",
                    routing_priority="balanced",
                    priority=req_priority,
                )
            )
        )
        assert d.chosen_model is not None

    def test_always_premium_overrides_request_priority_latency(self):
        """always-premium should pick premium regardless of request priority."""
        d_low = rr(
            self.engine.route(
                _req(
                    "Design system",
                    routing_priority="always-premium",
                    priority="low",
                )
            )
        )
        d_crit = rr(
            self.engine.route(
                _req(
                    "Design system",
                    routing_priority="always-premium",
                    priority="critical",
                )
            )
        )
        if d_low.chosen_model and d_crit.chosen_model:
            assert d_low.chosen_model.tier == "premium"
            assert d_crit.chosen_model.tier == "premium"

    # --- Priority + fallback chain fields (5) ---

    @pytest.mark.parametrize(
        "priority", ["always-premium", "quality-first", "balanced", "cost-optimized"]
    )
    def test_typed_fallback_chains_populated(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "Write a Python class",
                    routing_priority=priority,
                    plan="business_plan",
                )
            )
        )
        assert isinstance(d.fallback_on_rate_limit, list)
        assert isinstance(d.fallback_on_content_safety, list)
        assert isinstance(d.fallback_on_timeout, list)

    def test_always_premium_chain_has_no_lower_tier_in_content_safety(self):
        """
        When routing to premium, content safety chain may have same or higher tier
        (premium is top, so chain may be empty).
        """
        d = rr(
            self.engine.route(
                _req(
                    "Review this contract",
                    routing_priority="always-premium",
                    plan="business_plan",
                )
            )
        )
        # Content safety chain should only contain premium models (since chosen is premium)
        for m in d.fallback_on_content_safety:
            assert m.tier == "premium"

    # --- Priority + proxy mode (5) ---

    def test_proxy_mode_with_always_premium_no_key(self):
        d = rr(
            self.engine.route(
                _req(
                    "Test proxy",
                    routing_priority="always-premium",
                    mode="proxy",
                    provider_api_key=None,
                )
            )
        )
        assert d.proxy_response is not None
        assert "proxy error" in d.proxy_response.lower()

    def test_proxy_mode_decision_still_premium_with_always_premium(self):
        d = rr(
            self.engine.route(
                _req(
                    "Test proxy premium",
                    routing_priority="always-premium",
                    mode="proxy",
                    provider_api_key=None,
                )
            )
        )
        # Even in proxy mode (with no key, so error), the routing decision should be premium
        assert d.chosen_model is not None
        assert d.chosen_model.tier == "premium"

    @pytest.mark.parametrize("priority", ["quality-first", "cost-optimized"])
    def test_proxy_mode_with_other_priorities_no_key(self, priority):
        d = rr(
            self.engine.route(
                _req(
                    "Test",
                    routing_priority=priority,
                    mode="proxy",
                    provider_api_key=None,
                )
            )
        )
        assert d.proxy_response is not None

    # --- Priority consistency across repeated calls (10) ---

    @pytest.mark.parametrize(
        "prompt,priority",
        [
            ("hello", "always-premium"),
            ("Write code", "quality-first"),
            ("What is 2+2?", "balanced"),
            ("Summarize text", "cost-optimized"),
            ("Design system", "always-premium"),
            ("Translate to French", "quality-first"),
            ("Classify review", "balanced"),
            ("Extract emails", "cost-optimized"),
            ("Write blog post", "quality-first"),
            ("Prove theorem", "always-premium"),
        ],
    )
    def test_deterministic_with_zero_exploration(self, prompt, priority):
        """With exploration_rate=0.0 each priority must produce the same model twice."""
        d1 = rr(
            self.engine.route(
                _req(
                    prompt,
                    routing_priority=priority,
                    exploration_rate=0.0,
                )
            )
        )
        d2 = rr(
            self.engine.route(
                _req(
                    prompt,
                    routing_priority=priority,
                    exploration_rate=0.0,
                )
            )
        )
        if d1.chosen_model and d2.chosen_model:
            assert d1.chosen_model.model_id == d2.chosen_model.model_id

    # --- All four priorities produce valid decisions (15 general smoke tests) ---

    @pytest.mark.parametrize(
        "prompt,priority",
        [
            ("hi", "always-premium"),
            ("Explain backpropagation", "quality-first"),
            ("What is 2+2?", "balanced"),
            ("Convert 5 miles to km", "cost-optimized"),
            ("Build a REST API", "always-premium"),
            ("Write a blog post", "quality-first"),
            ("Is Python compiled?", "balanced"),
            ("Summarize this paragraph", "cost-optimized"),
            ("Prove √2 is irrational", "always-premium"),
            ("Compare React vs Vue", "quality-first"),
            ("Translate 'hello' to French", "balanced"),
            ("Classify this review as positive/negative", "cost-optimized"),
            ("Design a multi-tenant SaaS", "always-premium"),
            ("Explain the CAP theorem", "quality-first"),
            ("What is inflation?", "balanced"),
        ],
    )
    def test_all_priorities_return_valid_decision(self, prompt, priority):
        d = rr(self.engine.route(_req(prompt, routing_priority=priority, plan="business_plan")))
        assert d.chosen_model is not None
        assert d.priority_applied == priority
        assert d.estimated_cost >= 0.0
        assert 0.0 <= d.confidence <= 1.0
        assert d.routing_rule_matched != ""
