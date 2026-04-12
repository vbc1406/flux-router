"""
Core routing decision engine — the brain of the system.

Takes a RoutingRequest, runs it through 13 ordered steps, and returns a fully
populated RoutingDecision with a chosen model, fallback chain, cost estimate,
and reasoning string.

All dependencies are injected so the engine is fully testable without hitting
any real APIs.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from .adaptive_weights import AdaptiveWeights
from .analytics import RoutingAnalytics
from .budget_tracker import BudgetTracker
from .cache import ResponseCache
from .classifier import RequestClassifier
from .config import (
    AB_ALLOWED_PRIORITIES,
    AB_BLOCKED_SENSITIVITY_LEVELS,
    AB_EXPLORATION_RATE,
    AB_MAX_COMPLEXITY_SCORE,
    BUDGET_DOWNGRADE_COST_WEIGHT,
    BUDGET_DOWNGRADE_QUALITY_WEIGHT,
    CACHE_ENABLED,
    CONTEXT_COMPRESSION_THRESHOLD,
    CONTEXT_SUMMARY_TARGET_TOKENS,
    FREE_TIER_ABUSE_THRESHOLD_RPM,
    LATENCY_PRIORITY_MAP,
    MAX_COST_PER_REQUEST,
    MIN_QUALITY_THRESHOLD,
    SCORING_WEIGHTS,
    TIER_BOUNDARIES,
)
from .context_compressor import ContextCompressor
from .fallback_chain import build_fallback_chain
from .model_registry import ModelRegistry
from .schemas import ModelOption, RoutingDecision, RoutingRequest, TaskAnalysis

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)

_TIER_ORDER: list[str] = ["free", "cheap", "mid", "premium", "ultra"]


class RoutingEngine:
    """
    Stateless (per-call) routing decision engine.

    All state lives in the injected collaborators (registry, cache, budget,
    adaptive weights, analytics).  The route() method is safe to call
    concurrently — it only mutates copies.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        classifier: RequestClassifier,
        cache: ResponseCache,
        budget_tracker: BudgetTracker,
        adaptive_weights: AdaptiveWeights,
        context_compressor: ContextCompressor,
        analytics: RoutingAnalytics,
    ) -> None:
        self._registry    = model_registry
        self._classifier  = classifier
        self._cache       = cache
        self._budget      = budget_tracker
        self._adaptive    = adaptive_weights
        self._compressor  = context_compressor
        self._analytics   = analytics
        # Per-user free-tier RPM counter (simple sliding count for anti-gaming)
        self._user_free_rpm: dict[str, int] = {}

    # ── Public ──────────────────────────────────────────────────────────────

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """
        Run the full 13-step routing algorithm and return a RoutingDecision.

        Steps are ordered deliberately:
          1  Classify first — cache/budget both depend on task_type.
          2  Cache check — cheapest possible outcome.
          3  Cost ceiling — block ruinously expensive requests early.
          4  Hard constraints — filter to viable model set.
          5  Context compression — shrink if needed, then re-classify.
          6  Model override — honour explicit caller preference.
          7  Special rules — trivial requests, anti-gaming.
          8  Adaptive quality adjustment — incorporate learned scores.
          9  Tier selection + multi-criteria scoring.
          10 A/B exploration — controlled experiments on safe requests.
          11 Fallback chain construction.
          12 Budget check — smart downgrade if needed.
          13 Log + return.
        """
        cid = request.correlation_id

        # ══ STEP 1: CLASSIFY ════════════════════════════════════════════════
        analysis = self._classifier.analyze(request)
        log.debug("classified", cid=cid, task=analysis.task_type, score=analysis.complexity_score)

        # ══ STEP 2: CACHE CHECK ══════════════════════════════════════════════
        if CACHE_ENABLED and analysis.cache_eligible:
            cached = self._cache.get(analysis.prompt_fingerprint)
            if cached:
                self._analytics.log_cache_hit(cid, cached)
                log.info("cache_hit", cid=cid)
                return RoutingDecision(
                    chosen_model          = cached.model_used,
                    fallback_chain        = [],
                    reasoning             = "Cache hit — identical request served from cache",
                    estimated_cost        = 0.0,
                    estimated_savings     = cached.original_cost,
                    confidence            = 1.0,
                    routing_rule_matched  = "cache_hit",
                    cache_hit             = True,
                    context_was_compressed= False,
                    cost_blocked          = False,
                    correlation_id        = cid,
                    timestamp             = datetime.utcnow(),
                )

        # ══ STEP 3: COST CEILING CHECK ══════════════════════════════════════
        worst = _estimate_cost(analysis, self._registry.most_expensive_model())
        if worst > MAX_COST_PER_REQUEST and not request.metadata.get("override_cost_ceiling"):
            log.warning("cost_ceiling_blocked", cid=cid, worst_case=worst)
            return RoutingDecision(
                chosen_model          = None,
                fallback_chain        = [],
                reasoning             = (
                    f"Estimated ${worst:.2f} exceeds ceiling ${MAX_COST_PER_REQUEST}. "
                    f"Set metadata.override_cost_ceiling=true to proceed."
                ),
                estimated_cost        = worst,
                estimated_savings     = 0.0,
                confidence            = 0.0,
                routing_rule_matched  = "cost_ceiling_blocked",
                cache_hit             = False,
                context_was_compressed= False,
                cost_blocked          = True,
                correlation_id        = cid,
                timestamp             = datetime.utcnow(),
            )

        # ══ STEP 4: HARD CONSTRAINT FILTERING ═══════════════════════════════
        all_models = self._registry.all_available_models()
        candidates = [
            m for m in all_models
            if _passes_hard_constraints(m, request, analysis, self._registry)
        ]

        if not candidates:
            candidates = _relaxed_filter(all_models, request, analysis)

        if not candidates:
            return RoutingDecision(
                reasoning            = "No models available matching request constraints",
                routing_rule_matched = "no_candidates",
                correlation_id       = cid,
                timestamp            = datetime.utcnow(),
                cost_blocked         = False,
            )

        # ══ STEP 5: CONTEXT COMPRESSION ══════════════════════════════════════
        max_window            = max(m.max_context_window for m in candidates)
        context_was_compressed = False

        if analysis.total_context_needed > max_window * CONTEXT_COMPRESSION_THRESHOLD:
            request = self._compressor.compress(request, target_tokens=CONTEXT_SUMMARY_TARGET_TOKENS)
            analysis = self._classifier.analyze(request)
            context_was_compressed = True
            log.info("context_compressed", cid=cid)

        # ══ STEP 6: EXPLICIT MODEL OVERRIDE ══════════════════════════════════
        if override_id := request.metadata.get("model"):
            match = _find_model(override_id, candidates)
            if match:
                chain = build_fallback_chain(match, candidates, analysis)
                cost  = _estimate_cost(analysis, match)
                log.info("explicit_override", cid=cid, model=match.model_id)
                decision = self._finalise(
                    chosen=match,
                    chain=chain,
                    analysis=analysis,
                    request=request,
                    rule="explicit_model_override",
                    compressed=context_was_compressed,
                    cost=cost,
                )
                self._post_route(decision, request)
                return decision

        # ══ STEP 7: SPECIAL ROUTING RULES ═══════════════════════════════════
        # Trivially short conversation → always free tier
        if analysis.task_type == "conversation" and analysis.estimated_input_tokens < 50:
            free_models = [m for m in candidates if m.tier == "free"]
            if free_models:
                best_free = _pick_best(free_models, analysis)
                chain     = build_fallback_chain(best_free, candidates, analysis)
                cost      = _estimate_cost(analysis, best_free)
                decision  = self._finalise(
                    best_free, chain, analysis, request,
                    "trivial_request", context_was_compressed, cost,
                )
                self._post_route(decision, request)
                return decision

        # Anti-gaming: punish users hammering the free tier
        if self._get_user_free_rpm(request.user_id) > FREE_TIER_ABUSE_THRESHOLD_RPM:
            candidates = [m for m in candidates if m.tier != "free"]
            if not candidates:
                candidates = self._registry.all_available_models()

        # ══ STEP 8: ADAPTIVE QUALITY ADJUSTMENT ═════════════════════════════
        for m in candidates:
            m.adjusted_quality = self._adaptive.get_adjusted_score(
                model_id   = m.model_id,
                task_type  = analysis.task_type,
                base_score = m.quality_ratings.get(analysis.task_type, 0.5),
            )

        quality_filtered = [m for m in candidates if m.adjusted_quality >= MIN_QUALITY_THRESHOLD]
        if quality_filtered:
            candidates = quality_filtered

        # ══ STEP 9: TIER SELECTION + SCORING ════════════════════════════════
        target_tier     = _get_tier_for_score(analysis.complexity_score)
        tier_candidates = [m for m in candidates if m.tier == target_tier]
        if not tier_candidates:
            tier_candidates = _get_adjacent_tier_models(candidates, target_tier)
        if not tier_candidates:
            tier_candidates = candidates

        latency_mode = LATENCY_PRIORITY_MAP.get(request.priority, "balanced")
        weights      = SCORING_WEIGHTS[latency_mode]

        for m in tier_candidates:
            m.routing_score = (
                m.adjusted_quality * weights["quality"]
                + (1.0 - _normalize_cost(m, tier_candidates)) * weights["cost"]
                + (1.0 - _normalize_latency(m, tier_candidates)) * weights["latency"]
            )

        tier_candidates.sort(key=lambda m: m.routing_score, reverse=True)
        chosen = tier_candidates[0]
        rule   = "tier_selection"

        # ══ STEP 10: A/B EXPLORATION ══════════════════════════════════════
        ab_allowed = (
            request.priority in AB_ALLOWED_PRIORITIES
            and analysis.complexity_score < AB_MAX_COMPLEXITY_SCORE
            and analysis.sensitivity_level not in AB_BLOCKED_SENSITIVITY_LEVELS
        )

        if ab_allowed and random.random() < AB_EXPLORATION_RATE:
            cheaper = _get_one_tier_below(candidates, target_tier)
            viable  = [m for m in cheaper if m.adjusted_quality >= MIN_QUALITY_THRESHOLD]
            if viable:
                chosen = random.choice(viable)
                rule   = "ab_exploration"
                log.debug("ab_exploration_triggered", cid=cid, model=chosen.model_id)

        # ══ STEP 11: FALLBACK CHAIN ══════════════════════════════════════
        fallback_chain = build_fallback_chain(chosen, candidates, analysis)

        # ══ STEP 12: BUDGET CHECK (smart downgrade) ══════════════════════
        estimated_cost = _estimate_cost(analysis, chosen)

        if self._budget.would_exceed_budget(request.user_id, estimated_cost):
            affordable = [
                m for m in candidates
                if not self._budget.would_exceed_budget(request.user_id, _estimate_cost(analysis, m))
            ]
            if affordable:
                chosen = max(affordable, key=lambda m: _downgrade_score(m, candidates))
                rule  += " | budget_downgraded_smart"
            else:
                chosen = min(candidates, key=lambda m: m.cost_per_1k_input)
                rule  += " | budget_exhausted"
            estimated_cost = _estimate_cost(analysis, chosen)
            log.info("budget_downgrade", cid=cid, new_model=chosen.model_id, rule=rule)

        # ══ STEP 13: LOG + RETURN ════════════════════════════════════════
        decision = self._finalise(
            chosen, fallback_chain, analysis, request, rule, context_was_compressed, estimated_cost
        )
        self._post_route(decision, request)
        return decision

    # ── Private helpers ──────────────────────────────────────────────────────

    def _finalise(
        self,
        chosen: ModelOption,
        chain: list[ModelOption],
        analysis: TaskAnalysis,
        request: RoutingRequest,
        rule: str,
        compressed: bool,
        cost: float,
    ) -> RoutingDecision:
        """Assemble and return a fully populated RoutingDecision."""
        savings    = _estimate_savings_vs_premium(analysis, chosen, self._registry)
        confidence = _compute_confidence(analysis, chosen)
        return RoutingDecision(
            chosen_model          = chosen,
            fallback_chain        = chain,
            reasoning             = _build_reasoning(analysis, chosen, rule),
            estimated_cost        = round(cost, 6),
            estimated_savings     = round(savings, 6),
            confidence            = confidence,
            routing_rule_matched  = rule,
            cache_hit             = False,
            context_was_compressed= compressed,
            cost_blocked          = False,
            correlation_id        = request.correlation_id,
            timestamp             = datetime.utcnow(),
        )

    def _post_route(self, decision: RoutingDecision, request: RoutingRequest) -> None:
        """Side effects after a decision: log it and update load counters."""
        self._analytics.log_decision(decision)
        if decision.chosen_model:
            self._registry.update_load(decision.chosen_model.model_id)
            self._inc_user_free_rpm(request.user_id, decision.chosen_model.tier)

    def _get_user_free_rpm(self, user_id: str) -> int:
        return self._user_free_rpm.get(user_id, 0)

    def _inc_user_free_rpm(self, user_id: str, tier: str) -> None:
        if tier == "free":
            self._user_free_rpm[user_id] = self._user_free_rpm.get(user_id, 0) + 1


# ── Module-level pure helpers (no self, easy to unit-test) ──────────────────

def _estimate_cost(analysis: TaskAnalysis, model: ModelOption) -> float:
    """
    Estimate the dollar cost for this request.
    Output tokens are capped at the model's max to avoid inflated estimates.
    """
    output_tokens = min(analysis.estimated_output_tokens, model.max_output_tokens)
    cost = (
        (analysis.estimated_input_tokens / 1000.0) * model.cost_per_1k_input
        + (output_tokens / 1000.0) * model.cost_per_1k_output
    )
    return round(cost, 6)


def _normalize_cost(model: ModelOption, candidates: list[ModelOption]) -> float:
    """
    0.0 = cheapest, 1.0 = most expensive, normalised across candidates.
    Returns 0.5 when all candidates cost the same (avoids divide-by-zero).
    """
    costs   = [m.cost_per_1k_input + m.cost_per_1k_output for m in candidates]
    min_c, max_c = min(costs), max(costs)
    if max_c == min_c:
        return 0.5
    model_cost = model.cost_per_1k_input + model.cost_per_1k_output
    return (model_cost - min_c) / (max_c - min_c)


def _normalize_latency(model: ModelOption, candidates: list[ModelOption]) -> float:
    """
    0.0 = fastest, 1.0 = slowest, normalised across candidates.
    Returns 0.5 when all candidates have the same latency.
    """
    latencies = [m.avg_latency_ms for m in candidates]
    min_l, max_l = min(latencies), max(latencies)
    if max_l == min_l:
        return 0.5
    return (model.avg_latency_ms - min_l) / (max_l - min_l)


def _get_tier_for_score(score: float) -> str:
    """Map a complexity score to its corresponding tier name."""
    for tier, (lo, hi) in TIER_BOUNDARIES.items():
        if lo <= score < hi:
            return tier
    return "ultra"   # score == 1.0 edge case


def _get_adjacent_tier_models(candidates: list[ModelOption], target_tier: str) -> list[ModelOption]:
    """
    When the target tier has no candidates, expand to the nearest populated tier.
    Tries tiers outward in both directions (±1, ±2, …).
    """
    if target_tier not in _TIER_ORDER:
        return candidates
    idx = _TIER_ORDER.index(target_tier)
    for offset in [1, -1, 2, -2, 3, -3, 4, -4]:
        adj_idx = idx + offset
        if 0 <= adj_idx < len(_TIER_ORDER):
            adj = [m for m in candidates if m.tier == _TIER_ORDER[adj_idx]]
            if adj:
                return adj
    return candidates


def _get_one_tier_below(candidates: list[ModelOption], target_tier: str) -> list[ModelOption]:
    """Return models exactly one tier below target_tier."""
    if target_tier not in _TIER_ORDER:
        return []
    idx = _TIER_ORDER.index(target_tier)
    if idx == 0:
        return []
    return [m for m in candidates if m.tier == _TIER_ORDER[idx - 1]]


def _passes_hard_constraints(
    model: ModelOption,
    request: RoutingRequest,
    analysis: TaskAnalysis,
    registry: ModelRegistry,
) -> bool:
    """All six hard constraints that a model must pass to be a candidate."""
    return (
        all(cap in model.capabilities for cap in request.required_capabilities)
        and model.max_context_window >= analysis.total_context_needed
        and model.is_available
        and analysis.sensitivity_level in model.allowed_sensitivity_levels
        and (not request.prefer_streaming or model.supports_streaming)
        and _is_allowed_for_plan(model, request.plan)
        and not registry.is_near_rate_limit(model.model_id)
    )


def _relaxed_filter(
    models: list[ModelOption],
    request: RoutingRequest,
    analysis: TaskAnalysis,
) -> list[ModelOption]:
    """
    Relaxed filter: drop rate-limit and streaming requirements.
    Called when the strict filter yields zero candidates.
    """
    return [
        m for m in models
        if all(cap in m.capabilities for cap in request.required_capabilities)
        and m.max_context_window >= analysis.total_context_needed
        and m.is_available
        and analysis.sensitivity_level in m.allowed_sensitivity_levels
        and _is_allowed_for_plan(m, request.plan)
    ]


def _is_allowed_for_plan(model: ModelOption, plan: str) -> bool:
    """
    Enforce tier access by subscription plan.
    Free plans cannot reach premium/ultra models regardless of their budget setting.
    """
    tier_access: dict[str, set[str]] = {
        "free_plan":     {"free", "cheap"},
        "pro_plan":      {"free", "cheap", "mid", "premium"},
        "business_plan": {"free", "cheap", "mid", "premium", "ultra"},
    }
    return model.tier in tier_access.get(plan, {"free"})


def _find_model(model_id: str, candidates: list[ModelOption]) -> ModelOption | None:
    """Find a model by model_id or display_name (case-insensitive)."""
    mid_lower = model_id.lower()
    for m in candidates:
        if m.model_id.lower() == mid_lower or m.display_name.lower() == mid_lower:
            return m
    return None


def _pick_best(models: list[ModelOption], analysis: TaskAnalysis) -> ModelOption:
    """Pick the highest-quality model for the given task type."""
    return max(models, key=lambda m: m.quality_ratings.get(analysis.task_type, 0.5))


def _downgrade_score(model: ModelOption, candidates: list[ModelOption]) -> float:
    """
    Composite score for selecting the best affordable model during a budget
    downgrade: favour quality over raw cheapness.
    """
    norm_cost = _normalize_cost(model, candidates)
    return (
        model.adjusted_quality * BUDGET_DOWNGRADE_QUALITY_WEIGHT
        - norm_cost * BUDGET_DOWNGRADE_COST_WEIGHT
    )


def _estimate_savings_vs_premium(
    analysis: TaskAnalysis,
    chosen: ModelOption,
    registry: ModelRegistry,
) -> float:
    """How much cheaper is our chosen model vs the most expensive alternative?"""
    premium_cost = _estimate_cost(analysis, registry.most_expensive_model())
    actual_cost  = _estimate_cost(analysis, chosen)
    return max(0.0, premium_cost - actual_cost)


def _compute_confidence(analysis: TaskAnalysis, chosen: ModelOption) -> float:
    """
    Confidence is high when the complexity score is well-centred in the chosen
    tier AND the model has a strong quality rating for this task type.
    """
    lo, hi  = TIER_BOUNDARIES.get(chosen.tier, (0.0, 1.0))
    center  = (lo + hi) / 2.0
    half_w  = (hi - lo) / 2.0 if (hi - lo) > 0 else 0.1
    tier_fit = max(0.0, 1.0 - abs(analysis.complexity_score - center) / half_w)
    quality  = chosen.adjusted_quality
    return round(min(1.0, tier_fit * 0.4 + quality * 0.6), 4)


def _build_reasoning(analysis: TaskAnalysis, chosen: ModelOption, rule: str) -> str:
    return (
        f"task={analysis.task_type} | "
        f"complexity={analysis.complexity_score:.3f} | "
        f"tier={chosen.tier} | "
        f"model={chosen.display_name} | "
        f"rule={rule}"
    )
