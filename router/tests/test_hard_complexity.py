"""
File: router/tests/test_hard_complexity.py

Purpose:
Tests for Item 2 — hard-complexity escalation (config.HARD_COMPLEXITY_TIER_FLOORS,
routing_engine._composed_min_tier) and its composition with the existing
STEP_TYPE_FLOORS / DOMAIN_TIER_FLOORS floors, budget walk-down interaction,
and the "no eligible model at the required tier" case.

How to run:
  pytest -v router/tests/test_hard_complexity.py
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


def _engine(
    models: list[ModelOption] | None = None, budget: BudgetTracker | None = None
) -> RoutingEngine:
    registry = ModelRegistry()
    if models is not None:
        registry._models = {m.model_id: m for m in models}
    cache = ResponseCache(enabled=False)
    adaptive = AdaptiveWeights(state_file=None)
    analytics = RoutingAnalytics(log_path=None)
    budget = budget or BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id": "u_hard_complexity",
        "plan": "business_plan",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(raw_prompt=prompt, correlation_id=str(uuid.uuid4()), **defaults)


def rr(coro):
    return asyncio.run(coro)


_COMMON = {
    "capabilities": [],
    "max_context_window": 200_000,
    "max_output_tokens": 4096,
}


class TestHardComplexityEscalation:
    def test_simple_code_generation_stays_cheap_or_mid(self):
        engine = _engine()
        decision = rr(engine.route(_req("Write a Python function that reverses a string.")))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("free", "cheap", "mid")

    def test_ordinary_arithmetic_does_not_escalate(self):
        engine = _engine()
        decision = rr(engine.route(_req("What is 17 + 25?")))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier in ("free", "cheap")

    def test_complex_distributed_systems_design_escalates(self):
        engine = _engine()
        prompt = (
            "Design a distributed, linearizable key-value store that tolerates "
            "network partitions, handles leader election under Byzantine faults, "
            "supports online resharding without downtime, and provides exactly-once "
            "delivery semantics across cross-region replicas, while reasoning "
            "carefully step-by-step through every failure mode and consistency "
            "tradeoff involved, including clock skew, split-brain scenarios, and "
            "quorum overlap during concurrent membership changes."
        )
        decision = rr(engine.route(_req(prompt), verbose=True))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        assert any("complexity:" in r for r in decision.explanation.floors_applied)

    def test_rigorous_math_proof_escalates(self):
        engine = _engine()
        prompt = (
            "Prove rigorously, step by step, that there are infinitely many prime "
            "numbers using Euclid's proof by contradiction, addressing every edge "
            "case and justifying each logical step of the derivation in full detail."
        )
        decision = rr(engine.route(_req(prompt), verbose=True))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"

    def test_combined_step_domain_complexity_floor_picks_strongest(self):
        """A 'plan' step_type (floor=mid) for a medical-domain, high-complexity
        request must resolve to premium — the strongest of the three floors —
        not just the step-type floor."""
        engine = _engine()
        prompt = (
            "Plan out, step by step, how to diagnose and treat a patient presenting "
            "with acute chest pain, shortness of breath, and suspected myocardial "
            "infarction, addressing every differential diagnosis rigorously."
        )
        decision = rr(engine.route(_req(prompt, step_type="plan"), verbose=True))
        assert decision.chosen_model is not None
        assert decision.chosen_model.tier == "premium"
        reasons = decision.explanation.floors_applied
        assert any(r.startswith("agent_step:plan") for r in reasons)
        assert any(r.startswith("domain:medical") for r in reasons)

    def test_budget_pressure_never_drops_below_mandatory_floor(self):
        """Even when the budget is exhausted, the walk-down must never select
        a model below the composed floor — floor-violating models were never
        candidates in the first place, so walk-down can only choose among
        floor-compliant ones."""
        models = [
            ModelOption(
                provider="p-free",
                model_id="m-free-cheap-but-under-floor",
                display_name="Free",
                tier="free",
                cost_per_1k_input=0.0001,
                cost_per_1k_output=0.0002,
                quality_ratings={"reasoning": 0.9, "general": 0.9},
                **_COMMON,
            ),
            ModelOption(
                provider="p-premium",
                model_id="m-premium-only-eligible",
                display_name="Premium",
                tier="premium",
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
                quality_ratings={"reasoning": 0.95, "general": 0.95},
                **_COMMON,
            ),
        ]
        # Pre-load spend right up against pro_plan's $50.00 daily cap (using
        # pro_plan, not free_plan, so premium tier is still plan-eligible —
        # free_plan can never reach premium regardless of budget) so this
        # request's estimated cost pushes the walk-down check over the edge,
        # without needing unrealistic pricing that would trip the separate
        # MAX_COST_PER_REQUEST ceiling in Step 3.
        near_exhausted_budget = BudgetTracker()
        near_exhausted_budget.record_spend(
            "u_hard_complexity", 49.999, "some-model", "seed", plan="pro_plan"
        )
        engine = _engine(models, budget=near_exhausted_budget)
        prompt = (
            "Prove rigorously, step by step, that the square root of 2 is "
            "irrational, addressing every edge case of the contradiction argument "
            "in full mathematical detail and derivation."
        )
        req = _req(prompt, plan="pro_plan")
        decision = rr(engine.route(req))
        assert decision.chosen_model is not None
        # The only floor-compliant (premium) model must be chosen even though
        # it is far more expensive than the free-tier alternative, and even
        # though it too now exceeds the exhausted daily budget.
        assert decision.chosen_model.tier == "premium"
        assert decision.chosen_model.model_id == "m-premium-only-eligible"
        assert decision.budget_exhausted is True

    def test_no_eligible_model_at_required_tier(self):
        """Catalog has nothing at/above the complexity-forced floor -> a
        clean no_candidates result, not an under-qualified selection."""
        below_floor_only = [
            ModelOption(
                provider="p-mid",
                model_id="m-mid-only",
                display_name="Mid Only",
                tier="mid",
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.01,
                quality_ratings={"reasoning": 0.9, "general": 0.9},
                **_COMMON,
            ),
        ]
        engine = _engine(below_floor_only)
        prompt = (
            "Prove rigorously, step by step, that there are infinitely many "
            "primes, addressing every logical edge case of the derivation."
        )
        decision = rr(engine.route(_req(prompt)))
        assert decision.chosen_model is None
        assert decision.routing_rule_matched == "no_candidates"
        assert "complexity:" in decision.reasoning
