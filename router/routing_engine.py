"""
Core routing decision engine — the brain of the system.

Takes a RoutingRequest, runs it through 13 ordered steps, and returns a fully
populated RoutingDecision with a chosen model, fallback chain, cost estimate,
and reasoning string.

All dependencies are injected so the engine is fully testable without hitting
any real APIs.

Changes in this version:
  Change 1  — routing_priority parameter (always-premium / quality-first /
               balanced / cost-optimized).  always-premium bypasses Steps 8-10.
  Change 2  — MIN_CONFIDENCE_THRESHOLD: low-confidence decisions are upgraded
               to premium automatically.
  Change 3  — Per-customer adaptive quality memory (customer_id).
  Change 4  — Per-request and per-day cost caps (max_cost_per_request,
               max_daily_cost, DailyBudgetTracker).
  Change 5  — Context-aware typed fallback chains (rate_limit / content_safety
               / timeout).
  Change 6  — Configurable A/B exploration rate (exploration_rate per request).
  Change 7  — Linear tier walk-down replaces weighted budget reselection.
  Change 8  — "ultra" tier removed; ultra models are now "premium".
  Change 9  — "proxy" mode: after decision, actually call the provider API.
"""

from __future__ import annotations

import random
import threading
import time
from collections import defaultdict, deque
from datetime import datetime

import structlog

from .adaptive_weights import AdaptiveWeights
from .analytics import RoutingAnalytics
from .budget_tracker import BudgetTracker, DailyBudgetTracker
from .cache import ResponseCache
from .classifier import RequestClassifier
from .config import (
    AB_ALLOWED_PRIORITIES,
    AB_BLOCKED_SENSITIVITY_LEVELS,
    AB_MAX_COMPLEXITY_SCORE,
    AB_MAX_EXPLORATION_RATE,
    CACHE_ENABLED,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    CONTEXT_COMPRESSION_THRESHOLD,
    CONTEXT_PENALTY_HARD_CUTOFF,
    CONTEXT_PENALTY_HIGH_FACTOR,
    CONTEXT_PENALTY_HIGH_RATIO,
    CONTEXT_PENALTY_MID_FACTOR,
    CONTEXT_PENALTY_MID_RATIO,
    CONTEXT_SUMMARY_TARGET_TOKENS,
    CONVERSATION_DEPTH_THRESHOLD,
    CONVERSATION_EXPIRY_SECONDS,
    CONVERSATION_STICKY_BIAS_DEEP,
    CONVERSATION_STICKY_BIAS_SHALLOW,
    FREE_TIER_ABUSE_THRESHOLD_RPM,
    LATENCY_PRIORITY_MAP,
    MAX_COST_PER_REQUEST,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_QUALITY_THRESHOLD,
    SCORING_WEIGHTS,
    TIER_BOUNDARIES,
    TIER_ORDER,
    VALID_ROUTING_PRIORITIES,
)
from .circuit_breaker import CircuitBreaker
from .context_compressor import ContextCompressor
from .fallback_chain import build_fallback_chain, build_typed_fallback_chains
from .model_registry import ModelRegistry
from .schemas import ModelOption, RoutingDecision, RoutingRequest, TaskAnalysis

log = structlog.get_logger(__name__)

_TIER_ORDER = TIER_ORDER  # local alias kept for readability inside this module


class ConversationStore:
    """
    Fix 1: Thread-safe in-memory store for per-conversation model preferences.

    Tracks which model was used last in a conversation so the routing engine
    can apply a sticky bias to keep the same model across turns.  Entries expire
    after CONVERSATION_EXPIRY_SECONDS of inactivity.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}
        self._lock  = threading.Lock()

    def get(self, conversation_id: str) -> dict | None:
        """
        Return the conversation entry, or None if unknown / expired.
        Expired entries are deleted lazily on access.
        """
        with self._lock:
            entry = self._store.get(conversation_id)
            if entry is None:
                return None
            if time.monotonic() - entry["last_used"] > CONVERSATION_EXPIRY_SECONDS:
                del self._store[conversation_id]
                return None
            return dict(entry)  # shallow copy; callers must not mutate

    def update(self, conversation_id: str, model_id: str) -> None:
        """Record that model_id was used for this conversation turn."""
        with self._lock:
            prev = self._store.get(conversation_id, {})
            self._store[conversation_id] = {
                "last_model":    model_id,
                "message_count": prev.get("message_count", 0) + 1,
                "last_used":     time.monotonic(),
                "last_failed":   False,
            }

    def record_failure(self, conversation_id: str) -> None:
        """
        Mark the last model in this conversation as having failed.
        When set, the sticky bias is suppressed for the next turn so the
        router picks a fresh model rather than retrying the broken one.
        """
        with self._lock:
            entry = self._store.get(conversation_id)
            if entry:
                entry["last_failed"] = True

    def expire_old(self) -> None:
        """Proactively purge all expired entries (optional housekeeping)."""
        now = time.monotonic()
        with self._lock:
            expired = [
                k for k, v in self._store.items()
                if now - v["last_used"] > CONVERSATION_EXPIRY_SECONDS
            ]
            for k in expired:
                del self._store[k]


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
        # Change 4: DailyBudgetTracker for per-request cost caps.
        self._daily_budget = DailyBudgetTracker()
        # Fix 1: Per-conversation model preference store.
        self._conversation_store = ConversationStore()
        # Fix 4: Per-provider circuit breaker.
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        )
        # Sliding-window free-tier RPM counter per user (60-second window).
        self._user_free_rpm: dict[str, deque[float]] = defaultdict(deque)
        # Route call counter for periodic housekeeping (conversation expiry).
        self._route_count: int = 0

    # ── Public ──────────────────────────────────────────────────────────────

    async def route(self, request: RoutingRequest) -> RoutingDecision:
        """
        Run the full 13-step routing algorithm and return a RoutingDecision.

        Steps are ordered deliberately:
          1  Classify first — cache/budget both depend on task_type.
          2  Cache check — cheapest possible outcome.
          3  Cost ceiling — block ruinously expensive requests early.
          4  Hard constraints + per-request cost cap filter — viable model set.
          4b Daily budget cap check — force free tier if exceeded (Change 4).
          5  Context compression — shrink if needed, then re-classify.
          6  Model override — honour explicit caller preference.
          7  Special rules — trivial requests, anti-gaming.
          [always-premium shortcut — bypass Steps 8-10 (Change 1)]
          8  Adaptive quality adjustment — incorporate learned scores (Change 3).
          9  Tier selection + multi-criteria scoring (Change 1 weights).
          9b Confidence threshold check — upgrade to premium if low (Change 2).
          10 A/B exploration — controlled experiments (Change 6: rate per-request).
          11 Context-aware fallback chain construction (Change 5).
          12 Budget check — linear tier walk-down (Change 7).
          13 Log + return.
          [proxy mode — call provider API if mode=="proxy" (Change 9)]
        """
        cid = request.correlation_id

        # Validate new parameters upfront so errors are immediate and clear.
        _validate_routing_priority(request.routing_priority)
        _validate_exploration_rate(request.exploration_rate)
        _validate_mode(request.mode)

        # Fix 1: Look up conversation state BEFORE routing so we know the
        # previous model (reported in decision.last_model) and can apply bias.
        conv_entry = (
            self._conversation_store.get(request.conversation_id)
            if request.conversation_id else None
        )

        # ══ STEP 1: CLASSIFY ════════════════════════════════════════════════
        analysis = self._classifier.analyze(request)
        log.debug("classified", cid=cid, task=analysis.task_type, score=analysis.complexity_score)

        # ══ STEP 2: CACHE CHECK ══════════════════════════════════════════════
        if CACHE_ENABLED and analysis.cache_eligible:
            cached = self._cache.get(analysis.prompt_fingerprint)
            if cached:
                self._analytics.log_cache_hit(cid, cached, user_id=request.user_id)
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
                    priority_applied      = request.routing_priority,
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
                priority_applied      = request.routing_priority,
            )

        # ══ STEP 4: HARD CONSTRAINT FILTERING ═══════════════════════════════
        all_models = self._registry.all_available_models()
        candidates = [
            m for m in all_models
            if _passes_hard_constraints(m, request, analysis, self._registry)
            # Fix 4: Skip models whose provider circuit is open.
            and self._circuit_breaker.is_available(m.provider)
        ]

        # Change 4: Per-request cost cap — filter out models whose estimated
        # cost exceeds max_cost_per_request before any other scoring.
        if request.max_cost_per_request is not None:
            cost_capped = [
                m for m in candidates
                if _estimate_cost(analysis, m) <= request.max_cost_per_request
            ]
            if cost_capped:
                candidates = cost_capped
                log.debug(
                    "per_request_cost_cap_applied",
                    cid=cid,
                    cap=request.max_cost_per_request,
                    kept=len(candidates),
                )
            else:
                log.warning(
                    "per_request_cost_cap_filtered_all",
                    cid=cid,
                    cap=request.max_cost_per_request,
                )

        if not candidates:
            candidates = _relaxed_filter(all_models, request, analysis)

        if not candidates:
            return RoutingDecision(
                reasoning            = "No models available matching request constraints",
                routing_rule_matched = "no_candidates",
                correlation_id       = cid,
                timestamp            = datetime.utcnow(),
                cost_blocked         = False,
                priority_applied     = request.routing_priority,
            )

        # ══ STEP 4b: DAILY BUDGET CAP CHECK ═════════════════════════════════
        # Change 4: If max_daily_cost is set and the customer has already hit
        # it, force routing to free tier.  Fall back to error if free is unavailable.
        effective_customer_id = request.customer_id or request.user_id
        if request.max_daily_cost is not None:
            if self._daily_budget.is_cap_exceeded(effective_customer_id, request.max_daily_cost):
                free_candidates = [m for m in candidates if m.tier == "free"]
                if free_candidates:
                    log.info(
                        "daily_budget_cap_hit_forcing_free",
                        cid=cid,
                        customer_id=effective_customer_id,
                        cap=request.max_daily_cost,
                    )
                    best_free = _pick_best(free_candidates, analysis)
                    chain     = build_fallback_chain(best_free, candidates, analysis)
                    rl, cs, to = build_typed_fallback_chains(best_free, candidates, analysis)
                    cost      = _estimate_cost(analysis, best_free)
                    decision  = self._finalise(
                        chosen=best_free, chain=chain, analysis=analysis,
                        request=request, rule="daily_budget_cap_free_tier",
                        compressed=False, cost=cost,
                        priority_applied=request.routing_priority,
                        confidence_fallback=False,
                        budget_exhausted=False,
                        fallback_on_rate_limit=rl,
                        fallback_on_content_safety=cs,
                        fallback_on_timeout=to,
                    )
                    decision.last_model = conv_entry["last_model"] if conv_entry else None
                    self._post_route(decision, request)
                    return decision
                else:
                    log.warning(
                        "daily_budget_cap_exhausted_no_free_tier",
                        cid=cid,
                        customer_id=effective_customer_id,
                    )
                    return RoutingDecision(
                        chosen_model         = None,
                        reasoning            = (
                            f"Daily budget cap of ${request.max_daily_cost:.4f} has been reached "
                            f"and no free-tier models are available for this request."
                        ),
                        routing_rule_matched = "daily_budget_exhausted",
                        correlation_id       = cid,
                        timestamp            = datetime.utcnow(),
                        cost_blocked         = True,
                        budget_exhausted     = True,
                        priority_applied     = request.routing_priority,
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
                chain    = build_fallback_chain(match, candidates, analysis)
                rl, cs, to = build_typed_fallback_chains(match, candidates, analysis)
                cost     = _estimate_cost(analysis, match)
                log.info("explicit_override", cid=cid, model=match.model_id)
                decision = self._finalise(
                    chosen=match, chain=chain, analysis=analysis,
                    request=request, rule="explicit_model_override",
                    compressed=context_was_compressed, cost=cost,
                    priority_applied=request.routing_priority,
                    confidence_fallback=False, budget_exhausted=False,
                    fallback_on_rate_limit=rl,
                    fallback_on_content_safety=cs,
                    fallback_on_timeout=to,
                )
                decision.last_model = conv_entry["last_model"] if conv_entry else None
                self._post_route(decision, request)
                if request.mode == "proxy":
                    decision = await self._proxy_execute(decision, request)
                return decision

        # ══ STEP 7: SPECIAL ROUTING RULES ═══════════════════════════════════
        # Change 1: always-premium shortcut is checked FIRST inside Step 7 so
        # it overrides the trivial-request and anti-gaming short-circuits below.
        # (The spec says bypass Steps 8-10; we also bypass the trivial shortcut
        # since the caller explicitly asked for premium quality.)

        if request.routing_priority == "always-premium":
            chosen, rule = _route_always_premium(candidates, analysis)
            chain        = build_fallback_chain(chosen, candidates, analysis)
            rl, cs, to   = build_typed_fallback_chains(chosen, candidates, analysis)
            cost         = _estimate_cost(analysis, chosen)
            decision     = self._finalise(
                chosen=chosen, chain=chain, analysis=analysis,
                request=request, rule=rule,
                compressed=context_was_compressed, cost=cost,
                priority_applied="always-premium",
                confidence_fallback=False, budget_exhausted=False,
                fallback_on_rate_limit=rl,
                fallback_on_content_safety=cs,
                fallback_on_timeout=to,
            )
            decision.last_model = conv_entry["last_model"] if conv_entry else None
            self._post_route(decision, request)
            if request.mode == "proxy":
                decision = await self._proxy_execute(decision, request)
            return decision

        # Trivially short conversation → always free tier
        if analysis.task_type == "conversation" and analysis.estimated_input_tokens < 50:
            free_models = [m for m in candidates if m.tier == "free"]
            if free_models:
                best_free = _pick_best(free_models, analysis)
                chain     = build_fallback_chain(best_free, candidates, analysis)
                rl, cs, to = build_typed_fallback_chains(best_free, candidates, analysis)
                cost      = _estimate_cost(analysis, best_free)
                decision  = self._finalise(
                    best_free, chain, analysis, request,
                    "trivial_request", context_was_compressed, cost,
                    priority_applied=request.routing_priority,
                    confidence_fallback=False, budget_exhausted=False,
                    fallback_on_rate_limit=rl,
                    fallback_on_content_safety=cs,
                    fallback_on_timeout=to,
                )
                decision.last_model = conv_entry["last_model"] if conv_entry else None
                self._post_route(decision, request)
                if request.mode == "proxy":
                    decision = await self._proxy_execute(decision, request)
                return decision

        # Anti-gaming: punish users hammering the free tier
        if self._get_user_free_rpm(request.user_id) > FREE_TIER_ABUSE_THRESHOLD_RPM:
            candidates = [m for m in candidates if m.tier != "free"]
            if not candidates:
                candidates = self._registry.all_available_models()

        # ══ STEP 8: ADAPTIVE QUALITY ADJUSTMENT ═════════════════════════════
        # Change 3: use customer_id for per-customer EMA when available.
        for m in candidates:
            m.adjusted_quality = self._adaptive.get_adjusted_score(
                model_id    = m.model_id,
                task_type   = analysis.task_type,
                base_score  = m.quality_ratings.get(analysis.task_type, 0.5),
                customer_id = request.customer_id,
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

        # Change 1: routing_priority overrides the weight preset when set.
        weights = _get_weights_for_priority(request.routing_priority, request.priority)

        for m in tier_candidates:
            m.routing_score = (
                m.adjusted_quality * weights["quality"]
                + (1.0 - _normalize_cost(m, tier_candidates)) * weights["cost"]
                + (1.0 - _normalize_latency(m, tier_candidates)) * weights["latency"]
            )

        # Fix 2: Context length penalty — reduce score for models whose window
        # is heavily occupied by the current input.
        for m in tier_candidates:
            context_ratio = analysis.estimated_input_tokens / max(m.max_context_window, 1)
            if context_ratio >= CONTEXT_PENALTY_HIGH_RATIO:
                m.routing_score -= (context_ratio - CONTEXT_PENALTY_HIGH_RATIO) * CONTEXT_PENALTY_HIGH_FACTOR
            elif context_ratio > CONTEXT_PENALTY_MID_RATIO:
                m.routing_score -= (context_ratio - CONTEXT_PENALTY_MID_RATIO) * CONTEXT_PENALTY_MID_FACTOR

        # Fix 1: Sticky bias — add a bonus to the model used in the previous
        # conversation turn so it wins unless another model scores notably higher.
        # Skipped when the last model failed (avoid repeating a broken model).
        if conv_entry and not conv_entry.get("last_failed", False):
            prev_id   = conv_entry["last_model"]
            msg_count = conv_entry["message_count"]
            bias = (
                CONVERSATION_STICKY_BIAS_DEEP
                if msg_count >= CONVERSATION_DEPTH_THRESHOLD
                else CONVERSATION_STICKY_BIAS_SHALLOW
            )
            for m in tier_candidates:
                if m.model_id == prev_id:
                    m.routing_score += bias
                    log.debug(
                        "sticky_bias_applied",
                        cid=cid,
                        model=prev_id,
                        bias=bias,
                        msg_count=msg_count,
                    )
                    break

        tier_candidates.sort(key=lambda m: m.routing_score, reverse=True)
        chosen = tier_candidates[0]
        rule   = "tier_selection"

        # ══ STEP 9b: CONFIDENCE THRESHOLD CHECK ═════════════════════════════
        # Change 2: if the winner's routing score is below MIN_CONFIDENCE_THRESHOLD,
        # automatically upgrade to the best available premium model.
        confidence_fallback = False
        if chosen.routing_score < MIN_CONFIDENCE_THRESHOLD:
            premium_alts = [
                m for m in candidates
                if m.tier == "premium" and m.model_id != chosen.model_id
            ]
            if premium_alts:
                premium_alts.sort(key=lambda m: m.adjusted_quality, reverse=True)
                old_id = chosen.model_id
                chosen = premium_alts[0]
                confidence_fallback = True
                rule  += " | confidence_fallback"
                log.info(
                    "confidence_fallback_triggered",
                    cid=cid,
                    original_model=old_id,
                    upgraded_to=chosen.model_id,
                    score=tier_candidates[0].routing_score,
                    threshold=MIN_CONFIDENCE_THRESHOLD,
                )

        # ══ STEP 10: A/B EXPLORATION ══════════════════════════════════════
        # Change 6: use per-request exploration_rate (0.0 = disabled).
        exploration_rate = request.exploration_rate
        ab_allowed = (
            exploration_rate > 0.0
            and request.priority in AB_ALLOWED_PRIORITIES
            and analysis.complexity_score < AB_MAX_COMPLEXITY_SCORE
            and analysis.sensitivity_level not in AB_BLOCKED_SENSITIVITY_LEVELS
        )

        if ab_allowed and random.random() < exploration_rate:
            cheaper = _get_one_tier_below(candidates, target_tier)
            viable  = [m for m in cheaper if m.adjusted_quality >= MIN_QUALITY_THRESHOLD]
            if viable:
                chosen = random.choice(viable)
                rule   = "ab_exploration"
                log.debug("ab_exploration_triggered", cid=cid, model=chosen.model_id)

        # ══ STEP 11: FALLBACK CHAINS ══════════════════════════════════════
        # Change 5: build three typed fallback chains in addition to the generic one.
        fallback_chain = build_fallback_chain(chosen, candidates, analysis)
        rl_chain, cs_chain, to_chain = build_typed_fallback_chains(chosen, candidates, analysis)

        # ══ STEP 12: BUDGET CHECK (linear tier walk-down) ════════════════
        # Change 7: replaces weighted reselection with a simple tier step-down.
        estimated_cost  = _estimate_cost(analysis, chosen)
        budget_exhausted = False

        if self._budget.would_exceed_budget(request.user_id, estimated_cost):
            chosen, rule, budget_exhausted = _budget_tier_walkdown(
                chosen, candidates, analysis, request.user_id, self._budget, rule
            )
            estimated_cost = _estimate_cost(analysis, chosen)
            # Rebuild fallback chains after model change
            fallback_chain = build_fallback_chain(chosen, candidates, analysis)
            rl_chain, cs_chain, to_chain = build_typed_fallback_chains(chosen, candidates, analysis)

        # ══ STEP 13: LOG + RETURN ════════════════════════════════════════
        decision = self._finalise(
            chosen, fallback_chain, analysis, request, rule,
            context_was_compressed, estimated_cost,
            priority_applied=request.routing_priority,
            confidence_fallback=confidence_fallback,
            budget_exhausted=budget_exhausted,
            fallback_on_rate_limit=rl_chain,
            fallback_on_content_safety=cs_chain,
            fallback_on_timeout=to_chain,
        )
        decision.last_model = conv_entry["last_model"] if conv_entry else None
        self._post_route(decision, request)

        # Change 3: record routing event for customer profile
        if request.customer_id and decision.chosen_model:
            self._adaptive.record_routing_event(
                customer_id = request.customer_id,
                model_id    = decision.chosen_model.model_id,
                task_type   = analysis.task_type,
                cost        = decision.estimated_cost,
            )

        # Change 4: record daily spend if customer_id + max_daily_cost
        if request.max_daily_cost is not None and decision.chosen_model:
            self._daily_budget.record_spend(
                customer_id    = effective_customer_id,
                amount         = decision.estimated_cost,
                model_id       = decision.chosen_model.model_id,
                correlation_id = cid,
                task_type      = analysis.task_type,
            )

        # ══ CHANGE 9: PROXY MODE ════════════════════════════════════════
        if request.mode == "proxy":
            decision = await self._proxy_execute(decision, request)

        return decision

    # ── Private helpers ──────────────────────────────────────────────────────────

    def _finalise(
        self,
        chosen: ModelOption,
        chain: list[ModelOption],
        analysis: TaskAnalysis,
        request: RoutingRequest,
        rule: str,
        compressed: bool,
        cost: float,
        *,
        priority_applied: str = "balanced",
        confidence_fallback: bool = False,
        budget_exhausted: bool = False,
        fallback_on_rate_limit: list[ModelOption] | None = None,
        fallback_on_content_safety: list[ModelOption] | None = None,
        fallback_on_timeout: list[ModelOption] | None = None,
    ) -> RoutingDecision:
        """Assemble and return a fully populated RoutingDecision."""
        savings    = _estimate_savings_vs_premium(analysis, chosen, self._registry)
        confidence = _compute_confidence(analysis, chosen)
        return RoutingDecision(
            chosen_model               = chosen,
            fallback_chain             = chain,
            fallback_on_rate_limit     = fallback_on_rate_limit or [],
            fallback_on_content_safety = fallback_on_content_safety or [],
            fallback_on_timeout        = fallback_on_timeout or [],
            reasoning                  = _build_reasoning(analysis, chosen, rule),
            estimated_cost             = round(cost, 6),
            estimated_savings          = round(savings, 6),
            confidence                 = confidence,
            routing_rule_matched       = rule,
            cache_hit                  = False,
            context_was_compressed     = compressed,
            cost_blocked               = False,
            correlation_id             = request.correlation_id,
            timestamp                  = datetime.utcnow(),
            priority_applied           = priority_applied,
            confidence_fallback        = confidence_fallback,
            budget_exhausted           = budget_exhausted,
        )

    def _post_route(self, decision: RoutingDecision, request: RoutingRequest) -> None:
        """Side effects after a decision: log it, update load counters, update conversation."""
        self._analytics.log_decision(decision, user_id=request.user_id)
        if decision.chosen_model:
            self._registry.update_load(decision.chosen_model.model_id)
            self._inc_user_free_rpm(request.user_id, decision.chosen_model.tier)
            # Fix 1: Persist the chosen model for the next turn of this conversation.
            if request.conversation_id:
                self._conversation_store.update(
                    request.conversation_id, decision.chosen_model.model_id
                )
        # Periodically purge expired conversation entries to prevent memory growth.
        self._route_count += 1
        if self._route_count % 500 == 0:
            self._conversation_store.expire_old()

    async def _proxy_execute(
        self,
        decision: RoutingDecision,
        request: RoutingRequest,
    ) -> RoutingDecision:
        """
        Change 9: Proxy mode — call the selected model's provider API and attach
        the response to the RoutingDecision.  Walks the appropriate fallback chain
        on failure (rate-limit → rate_limit chain, timeout → timeout chain, etc.).
        """
        from .provider_caller import ProviderCallError, call_provider

        if not request.provider_api_key:
            log.warning("proxy_mode_no_api_key", cid=request.correlation_id)
            decision.proxy_response = (
                "[proxy error] No provider_api_key provided in the request."
            )
            return decision

        # Build ordered list of models to try: primary first, then typed fallback chain.
        # We attach which chain to pull from based on the error we encounter.
        primary = decision.chosen_model
        if primary is None:
            decision.proxy_response = "[proxy error] No chosen model in decision."
            return decision

        # Try models in sequence; error type dictates which fallback list comes next.
        models_to_try = [primary]
        fallback_map = {
            "rate_limited":   decision.fallback_on_rate_limit,
            "timeout":        decision.fallback_on_timeout,
            "content_filter": decision.fallback_on_content_safety,
        }
        seen_ids = {primary.model_id}

        for attempt, model in enumerate(models_to_try):
            try:
                response = await call_provider(model, request, request.provider_api_key)
                log.info(
                    "proxy_call_success",
                    cid=request.correlation_id,
                    model=model.model_id,
                    attempt=attempt,
                )
                # Fix 4: Record success on the circuit breaker.
                self._circuit_breaker.record_success(model.provider)
                decision.proxy_response  = response
                decision.proxy_model_used = model
                return decision

            except ProviderCallError as exc:
                status = exc.status_code
                log.warning(
                    "proxy_call_failed",
                    cid=request.correlation_id,
                    model=model.model_id,
                    attempt=attempt,
                    status=status,
                    error=str(exc),
                )
                # Fix 4: Record failure on the circuit breaker.
                self._circuit_breaker.record_failure(model.provider)
                # Determine error type and append the relevant fallback chain
                if status == 429:
                    error_type = "rate_limited"
                elif "timeout" in str(exc).lower():
                    error_type = "timeout"
                elif status and "content" in str(exc).lower():
                    error_type = "content_filter"
                else:
                    error_type = "rate_limited"  # generic: try same-tier alternatives

                chain_for_error = fallback_map.get(error_type, [])
                for m in chain_for_error:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)
                # Also append generic fallback_chain as last resort
                for m in decision.fallback_chain:
                    if m.model_id not in seen_ids:
                        models_to_try.append(m)
                        seen_ids.add(m.model_id)

            except Exception as exc:
                log.error(
                    "proxy_call_unexpected_error",
                    cid=request.correlation_id,
                    model=model.model_id,
                    error=str(exc),
                )
                break

        decision.proxy_response = (
            f"[proxy error] All {len(seen_ids)} model(s) failed for "
            f"correlation_id={request.correlation_id}"
        )
        return decision

    def _get_user_free_rpm(self, user_id: str) -> int:
        now = time.monotonic()
        window = self._user_free_rpm[user_id]
        while window and now - window[0] >= 60.0:
            window.popleft()
        return len(window)

    def _inc_user_free_rpm(self, user_id: str, tier: str) -> None:
        if tier == "free":
            self._user_free_rpm[user_id].append(time.monotonic())


# ── Module-level pure helpers (no self, easy to unit-test) ──────────────────

def _validate_routing_priority(priority: str) -> None:
    """Change 1: Reject unknown routing_priority values immediately."""
    if priority not in VALID_ROUTING_PRIORITIES:
        raise ValueError(
            f"Invalid routing_priority '{priority}'. "
            f"Valid values: {sorted(VALID_ROUTING_PRIORITIES)}"
        )


def _validate_exploration_rate(rate: float) -> None:
    """Change 6: Reject exploration_rate > 0.25 or < 0.0."""
    if not (0.0 <= rate <= AB_MAX_EXPLORATION_RATE):
        raise ValueError(
            f"exploration_rate {rate!r} out of range. "
            f"Must be in [0.0, {AB_MAX_EXPLORATION_RATE}]."
        )


def _validate_mode(mode: str) -> None:
    """Change 9: Reject unknown mode values."""
    if mode not in ("decision", "proxy"):
        raise ValueError(
            f"Invalid mode '{mode}'. Valid values: 'decision', 'proxy'."
        )


def _get_weights_for_priority(
    routing_priority: str,
    request_priority: str,
) -> dict[str, float]:
    """
    Change 1: Return the scoring weight dict for the given routing_priority.
    For "always-premium" and "balanced" fall through to the latency-mode-based
    weights derived from request.priority.
    """
    if routing_priority in ("quality-first", "cost-optimized"):
        return SCORING_WEIGHTS[routing_priority]
    # balanced / always-premium: use the existing latency-mode mapping
    latency_mode = LATENCY_PRIORITY_MAP.get(request_priority, "balanced")
    return SCORING_WEIGHTS[latency_mode]


def _route_always_premium(
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
) -> tuple[ModelOption, str]:
    """
    Change 1: Pick the highest-tier model that passed Step 4 constraints.
    Tries premium → mid → cheap → free (in that order) until we find models.
    """
    for tier in reversed(_TIER_ORDER):  # premium first, then mid, cheap, free
        tier_models = [m for m in candidates if m.tier == tier]
        if tier_models:
            # Within the tier, prefer highest base quality for the task type
            best = _pick_best(tier_models, analysis)
            rule = f"always_premium (tier={tier})"
            log.debug("always_premium_selected", model=best.model_id, tier=tier)
            return best, rule
    # Should not reach here if candidates is non-empty
    best = _pick_best(candidates, analysis)
    return best, "always_premium (fallback)"


def _budget_tier_walkdown(
    chosen: ModelOption,
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
    user_id: str,
    budget: BudgetTracker,
    rule: str,
) -> tuple[ModelOption, str, bool]:
    """
    Change 7: Walk down tiers one step at a time until we find a model that
    fits within the budget.  Returns (chosen_model, updated_rule, budget_exhausted).

    This replaces the previous 65%/35% weighted reselection logic with a linear
    and fully predictable walk-down that is easier to reason about and debug.
    """
    current_cost = _estimate_cost(analysis, chosen)
    if not budget.would_exceed_budget(user_id, current_cost):
        return chosen, rule, False

    if chosen.tier not in _TIER_ORDER:
        # Unknown tier — just return cheapest
        cheapest = min(candidates, key=lambda m: _estimate_cost(analysis, m))
        return cheapest, rule + " | budget_exhausted", True

    current_idx = _TIER_ORDER.index(chosen.tier)

    for target_idx in range(current_idx - 1, -1, -1):
        target_tier  = _TIER_ORDER[target_idx]
        tier_models  = [m for m in candidates if m.tier == target_tier]
        if not tier_models:
            continue
        tier_models.sort(key=lambda m: m.adjusted_quality, reverse=True)
        candidate     = tier_models[0]
        candidate_cost = _estimate_cost(analysis, candidate)
        if not budget.would_exceed_budget(user_id, candidate_cost):
            log.info(
                "budget_tier_walkdown",
                user_id=user_id,
                from_tier=chosen.tier,
                to_tier=target_tier,
                model=candidate.model_id,
            )
            return candidate, rule + " | budget_downgraded", False

    # Even free tier exceeds budget — return cheapest available
    cheapest = min(candidates, key=lambda m: _estimate_cost(analysis, m))
    log.warning("budget_exhausted_all_tiers", user_id=user_id)
    return cheapest, rule + " | budget_exhausted", True


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
    # Change 8: no more "ultra" edge case — premium now covers up to 1.01
    return "premium"


def _get_adjacent_tier_models(candidates: list[ModelOption], target_tier: str) -> list[ModelOption]:
    """
    When the target tier has no candidates, expand to the nearest populated tier.
    Tries tiers outward in both directions (±1, ±2, …).
    """
    if target_tier not in _TIER_ORDER:
        return candidates
    idx = _TIER_ORDER.index(target_tier)
    for offset in [1, -1, 2, -2, 3, -3]:
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
    """All hard constraints that a model must pass to be a candidate."""
    # Fix 2: Drop models where the input alone fills ≥90 % of the context window —
    # the model would likely truncate or fail even before output tokens are added.
    if analysis.estimated_input_tokens > model.max_context_window * CONTEXT_PENALTY_HARD_CUTOFF:
        return False
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
    Change 8: "ultra" tier no longer exists; all plans' tier sets updated.
    Free plans cannot reach premium models regardless of budget setting.
    """
    tier_access: dict[str, set[str]] = {
        "free_plan":     {"free", "cheap"},
        "pro_plan":      {"free", "cheap", "mid", "premium"},
        "business_plan": {"free", "cheap", "mid", "premium"},
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
