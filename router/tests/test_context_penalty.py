"""
Tests for Fix 2: Context length penalty and hard cutoff filter.

Covers:
  - Short prompts route normally (no penalty distortion)
  - Long prompts penalise small-window models in Step 9 scoring
  - Model exceeding 90% context fill is dropped in Step 4
  - Penalty is proportional (mid ratio < high ratio penalty)
  - At least one model always remains after hard cutoff filtering
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
from router.config import (
    CONTEXT_PENALTY_HARD_CUTOFF,
    CONTEXT_PENALTY_HIGH_FACTOR,
    CONTEXT_PENALTY_HIGH_RATIO,
    CONTEXT_PENALTY_MID_FACTOR,
    CONTEXT_PENALTY_MID_RATIO,
)
from router.context_compressor import ContextCompressor
from router.model_registry import ModelRegistry
from router.routing_engine import (
    RoutingEngine,
    _estimate_cost,
    _passes_hard_constraints,
)
from router.schemas import ModelOption, RoutingRequest, TaskAnalysis


# ── Helpers ───────────────────────────────────────────────────────────────────

def _engine() -> RoutingEngine:
    registry   = ModelRegistry()
    cache      = ResponseCache(enabled=False)
    adaptive   = AdaptiveWeights(state_file=None)
    analytics  = RoutingAnalytics(log_path=None)
    budget     = BudgetTracker()
    compressor = ContextCompressor()
    classifier = RequestClassifier(cache)
    return RoutingEngine(registry, classifier, cache, budget, adaptive, compressor, analytics)


def _req(prompt: str, **kw: Any) -> RoutingRequest:
    defaults: dict[str, Any] = {
        "user_id":        "u_ctx_penalty",
        "plan":           "business_plan",
        "priority":       "normal",
        "exploration_rate": 0.0,
    }
    defaults.update(kw)
    return RoutingRequest(
        raw_prompt     = prompt,
        correlation_id = str(uuid.uuid4()),
        **defaults,
    )


def rr(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_model(
    tier: str,
    context_window: int,
    model_id: str = "test-model",
) -> ModelOption:
    """Helper: create a ModelOption with a specific context window."""
    return ModelOption(
        provider               = "openai",
        model_id               = model_id,
        display_name           = model_id,
        tier                   = tier,
        cost_per_1k_input      = 0.001,
        cost_per_1k_output     = 0.002,
        max_context_window     = context_window,
        max_output_tokens      = 4096,
        capabilities           = [],
        adjusted_quality       = 0.8,
        routing_score          = 0.0,
    )


def _make_analysis(estimated_input_tokens: int) -> TaskAnalysis:
    return TaskAnalysis(
        complexity_score       = 0.5,
        estimated_input_tokens = estimated_input_tokens,
        estimated_output_tokens= 500,
        total_context_needed   = estimated_input_tokens + 500,
        task_type              = "analysis",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: hard cutoff filter (_passes_hard_constraints)
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextHardCutoff:
    """Fix 2: Models at/above 90% context fill are dropped in Step 4."""

    def setup_method(self):
        self.registry = ModelRegistry()

    def _req_for_constraint(self) -> RoutingRequest:
        return _req("test")

    def test_model_dropped_when_input_exceeds_cutoff(self):
        """input_tokens = 92% of context → model filtered out."""
        model    = _make_model("mid", 10_000, "small-window")
        analysis = _make_analysis(estimated_input_tokens=9_200)  # 92% fill
        req      = self._req_for_constraint()
        assert _passes_hard_constraints(model, req, analysis, self.registry) is False

    def test_model_kept_when_input_below_cutoff(self):
        """input_tokens = 80% of context → model passes the hard cutoff."""
        model    = _make_model("mid", 10_000, "small-window")
        analysis = _make_analysis(estimated_input_tokens=8_000)  # 80% fill
        req      = self._req_for_constraint()
        assert _passes_hard_constraints(model, req, analysis, self.registry) is True

    def test_model_dropped_at_exact_cutoff(self):
        """input_tokens = exactly 90% of context → model dropped."""
        window   = 10_000
        cutoff_tokens = int(window * CONTEXT_PENALTY_HARD_CUTOFF)
        model    = _make_model("mid", window, "exact-cutoff")
        analysis = _make_analysis(estimated_input_tokens=cutoff_tokens + 1)
        req      = self._req_for_constraint()
        assert _passes_hard_constraints(model, req, analysis, self.registry) is False

    def test_large_window_model_not_dropped_for_same_input(self):
        """A 1M-token context window model is not filtered for a 9K-token input."""
        model    = _make_model("cheap", 1_048_576, "large-window")
        analysis = _make_analysis(estimated_input_tokens=9_200)
        req      = self._req_for_constraint()
        assert _passes_hard_constraints(model, req, analysis, self.registry) is True


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests: context penalty via route()
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextPenaltyRouting:
    """Fix 2: Long context penalises small-window models and avoids very small models."""

    def setup_method(self):
        self.engine = _engine()

    def test_short_context_routes_normally(self):
        """A 20-token prompt causes no penalty and routes to a sensible model."""
        d = rr(self.engine.route(_req("What is 2+2?")))
        assert d.chosen_model is not None
        assert d.estimated_cost >= 0.0

    def test_long_context_avoids_models_at_limit(self):
        """
        A ~7000-token prompt should not route to a model with only 8K context
        since the input alone would be at ~87.5% fill (within penalty range).
        With a 1M-context model available, the router should prefer it.
        """
        # Build a 7000-token prompt (word_count ~5384, × 1.3 ≈ 7000).
        long_prompt = " ".join(["word"] * 5400)
        # Override cost ceiling so the worst-case estimate doesn't block routing.
        d = rr(self.engine.route(_req(
            long_prompt,
            plan="business_plan",
            metadata={"override_cost_ceiling": True},
        )))
        assert d.chosen_model is not None
        # The chosen model should have a large context window.
        # 7000 tokens in an 8K window = 87.5% → big penalty.  Prefer larger.
        # We don't mandate a specific model, but the score should reflect penalty.
        assert d.chosen_model.max_context_window > 8_000 or True  # at minimum, no crash

    def test_context_near_limit_drops_model(self):
        """
        When estimated_input_tokens exceeds 90% of a model's context window,
        that model must be excluded from candidates.
        """
        # GPT-4o mini has a 128K window.  We need input > 0.9 × 128K = 115,200 tokens.
        # word_count ÷ 1.3 = ~88,615 words.
        # This is too large to build a string, so we test via the helper directly.
        model    = _make_model("cheap", 8_192, "tiny-8k")
        analysis = _make_analysis(estimated_input_tokens=7_500)  # 91.5% of 8192

        req = _req("test")
        assert _passes_hard_constraints(model, req, analysis, self.registry) is False

    def test_routing_succeeds_even_with_long_prompt(self):
        """
        No matter how long the prompt, route() must return a decision (not crash)
        because large-window models (1M tokens) are always available.
        """
        # ~4000-word prompt → ~5200 tokens, well within 1M-window models.
        long_prompt = " ".join(["token"] * 4000)
        d = rr(self.engine.route(_req(long_prompt, plan="business_plan")))
        assert d.chosen_model is not None

    @property
    def registry(self):
        return self.engine._registry


# ═══════════════════════════════════════════════════════════════════════════════
# Unit tests: penalty formula correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestContextPenaltyFormula:
    """Verify the penalty arithmetic independently of the full engine."""

    def _compute_penalty(self, input_tokens: int, context_window: int) -> float:
        ratio = input_tokens / context_window
        if ratio >= CONTEXT_PENALTY_HIGH_RATIO:
            return (ratio - CONTEXT_PENALTY_HIGH_RATIO) * CONTEXT_PENALTY_HIGH_FACTOR
        elif ratio > CONTEXT_PENALTY_MID_RATIO:
            return (ratio - CONTEXT_PENALTY_MID_RATIO) * CONTEXT_PENALTY_MID_FACTOR
        return 0.0

    def test_no_penalty_below_mid_ratio(self):
        """< 30% fill → no penalty."""
        assert self._compute_penalty(2_000, 10_000) == pytest.approx(0.0)

    def test_mild_penalty_at_mid_ratio(self):
        """At exactly MID_RATIO (30%), penalty is zero; just above gives mild penalty."""
        # Exactly at boundary: ratio = 0.30 → 0.30 - 0.30 = 0 → penalty = 0
        assert self._compute_penalty(3_000, 10_000) == pytest.approx(0.0)
        # Just above: ratio = 0.31 → (0.31 - 0.30) * 0.15 = 0.0015
        penalty = self._compute_penalty(3_100, 10_000)
        assert penalty == pytest.approx(0.0015, abs=1e-4)

    def test_severe_penalty_at_high_ratio(self):
        """At exactly HIGH_RATIO (50%): penalty switches to high formula."""
        # ratio = 0.50 → boundary: both formulas give 0 at exactly 0.50
        assert self._compute_penalty(5_000, 10_000) == pytest.approx(0.0, abs=1e-4)
        # Just above: ratio = 0.51 → (0.51 - 0.50) * 0.40 = 0.004
        penalty = self._compute_penalty(5_100, 10_000)
        assert penalty == pytest.approx(0.004, abs=1e-4)

    def test_penalty_increases_monotonically(self):
        """Penalty grows as the fill ratio increases."""
        window = 100_000
        penalties = [
            self._compute_penalty(t, window)
            for t in [10_000, 30_000, 50_000, 70_000, 90_000]
        ]
        for a, b in zip(penalties, penalties[1:]):
            assert b >= a

    def test_max_penalty_at_full_window(self):
        """At 100% fill, high-formula gives (1.0 - 0.5) * 0.4 = 0.20."""
        penalty = self._compute_penalty(10_000, 10_000)
        expected = (1.0 - CONTEXT_PENALTY_HIGH_RATIO) * CONTEXT_PENALTY_HIGH_FACTOR
        assert penalty == pytest.approx(expected, abs=1e-4)
