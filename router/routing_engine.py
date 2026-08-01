"""
File: router/routing_engine.py

Purpose:
The brain of the routing system.  Takes a RoutingRequest, runs it through 13
ordered steps, and returns a fully populated RoutingDecision with the chosen
model, fallback chains, cost estimate, and reasoning string.

Main Classes:
  RoutingEngine   — the 13-step decision algorithm
  ConversationStore — per-conversation model preference (sticky bias)

Config Dependencies (all in config.py):
  TIER_BOUNDARIES, TIER_ORDER         — complexity score → model tier mapping
  SCORING_WEIGHTS                     — quality/cost/latency weights per priority
  MIN_CONFIDENCE_THRESHOLD            — below this score, upgrade to premium (Step 9b)
  AB_ALLOWED_PRIORITIES, AB_MAX_*     — A/B exploration guards (Step 10)
  BUDGET_LIMITS                       — plan spend limits (Step 12)
  CONVERSATION_STICKY_BIAS_*          — sticky model bonus (Step 9)
  CONTEXT_PENALTY_*                   — context window fill penalties (Step 9)
  CIRCUIT_BREAKER_*                   — per-provider failure protection (Step 4)

Key Entry Points:
  RoutingEngine.route(request)        — call this to get a routing decision
  RoutingEngine.__init__(...)         — inject all collaborators here

The 13 Steps (in execution order), plus Step 0 in the route() wrapper:
  0  Run-budget gate  → RunBudget.check_before_dispatch() (Task 3; only if request.run_id
                         is set). Raises RunBudgetExceeded before dispatch, or forces
                         cost-optimized routing_priority if degraded/warning.
  1  Classify         → RequestClassifier.analyze()
  2  Cache check      → ResponseCache.get()
  3  Cost ceiling     → block ruinously expensive requests early
  4  Hard constraints → filter to viable candidate set
  4b Daily budget cap → force free tier if daily cap exceeded
  5  Context compress → shrink prompt if needed
  6  Model override   → honour explicit caller preference
  7  Special rules    → trivial requests, anti-gaming, always-premium
  8  Adaptive quality → AdaptiveWeights.get_adjusted_score()
  9  Tier + scoring   → pick tier, score by quality/cost/latency, apply sticky bias
  9b Confidence check → upgrade to premium if winner score < MIN_CONFIDENCE_THRESHOLD
  10 A/B exploration  → route a fraction to cheaper tier (blocked by confidence_fallback)
  11 Fallback chains  → build generic + typed chains
  12 Budget check     → linear tier walk-down if over budget
  13 Log + return     → analytics, conversation store, daily budget record

Things NOT to change without discussion:
  - The step ordering (steps have deliberate dependencies)
  - The confidence_fallback → blocks A/B invariant (Step 9b before Step 10)
  - The threading model (route() itself is stateless; state lives in collaborators)
  - The _finalise() method signature (many early-exit paths call it)
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import structlog

from .adaptive_weights import AdaptiveWeights
from .analytics import RoutingAnalytics
from .attribution import CostAttribution
from .budget_tracker import BudgetTracker, DailyBudgetTracker
from .cache import ResponseCache
from .cascade import estimate_step_cost
from .circuit_breaker import CircuitBreaker
from .classifier import RequestClassifier
from .config import (
    AB_ALLOWED_PRIORITIES,
    AB_BLOCKED_SENSITIVITY_LEVELS,
    AB_MAX_COMPLEXITY_SCORE,
    AB_MAX_EXPLORATION_RATE,
    CACHE_DEFAULT_TTL_SECONDS,
    CACHE_ENABLED,
    CACHE_PREFIX_MIN_TOKENS,
    CACHE_STICKINESS_WEIGHT,
    CACHE_SWITCH_MARGIN,
    CASCADE_ESCALATION_TIERS,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    COMPLEXITY_QUALITY_FLOOR,
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
    ENABLE_RESPONSE_CACHE,
    FREE_TIER_ABUSE_THRESHOLD_RPM,
    LATENCY_PRIORITY_MAP,
    MAX_COST_PER_REQUEST,
    MIN_CONFIDENCE_THRESHOLD,
    MIN_QUALITY_THRESHOLD,
    SCORING_WEIGHTS,
    STEP_TYPE_FLOORS,
    TIER_BOUNDARIES,
    TIER_ORDER,
    VALID_ROUTING_PRIORITIES,
)
from .context_compressor import ContextCompressor
from .fallback_chain import build_fallback_chain, build_typed_fallback_chains
from .model_registry import ModelRegistry
from .prompt_cache import PromptCacheTracker, hash_prefix
from .run_budget import RunBudget
from .schemas import ModelOption, RoutingDecision, RoutingExplanation, RoutingRequest, TaskAnalysis

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
        self._lock = threading.Lock()

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
                "last_model": model_id,
                "message_count": prev.get("message_count", 0) + 1,
                "last_used": time.monotonic(),
                "last_failed": False,
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
                k
                for k, v in self._store.items()
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
        self._registry = model_registry
        self._classifier = classifier
        self._cache = cache
        self._budget = budget_tracker
        self._adaptive = adaptive_weights
        self._compressor = context_compressor
        self._analytics = analytics
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
        # Task 3: run-scoped budget enforcement (checked before Step 1).
        self._run_budget = RunBudget()
        # Task 5: cache-aware routing (which provider holds a warm prefix).
        self._prompt_cache = PromptCacheTracker()
        # Task 7: per-run/per-tenant cost attribution.
        self._attribution = CostAttribution()

    # ── Public ──────────────────────────────────────────────────────────────

    async def route(self, request: RoutingRequest, verbose: bool = False) -> RoutingDecision:
        """
        Task 3, Step 0: run-scoped budget gate, then delegate to the 13-step
        algorithm in _route_core(). Kept as a thin wrapper so every one of
        _route_core's early-return paths automatically gets the same
        run-budget bookkeeping without having to touch each of them.

        If request.run_id is unset, this is a pure passthrough — run-budget
        enforcement never applies to requests outside a run.

        Raises:
            RunBudgetExceeded — before dispatch, if the run already met/exceeded
                                 any of its limits from prior steps.
        """
        if not request.run_id:
            return await self._route_core(request, verbose=verbose)

        run_id = request.run_id
        budget_state = self._run_budget.check_before_dispatch(run_id)
        cost_so_far, steps_so_far = self._run_budget.snapshot(run_id)

        effective_request = request
        budget_warning: str | None = None
        if budget_state in ("degraded", "warning"):
            # Force cost-optimized routing for the rest of the run, regardless
            # of what the caller asked for — this is what keeps a degrading
            # run's remaining steps cheap instead of crashing outright.
            effective_request = request.model_copy(update={"routing_priority": "cost-optimized"})
            log.info(
                "run_budget_degraded",
                run_id=run_id,
                budget_state=budget_state,
                cost_so_far=cost_so_far,
                steps_so_far=steps_so_far,
            )
        if budget_state == "warning":
            budget_warning = (
                f"Run '{run_id}' is at {budget_state} threshold "
                f"(${cost_so_far:.4f} spent over {steps_so_far} step(s)); "
                "consider wrapping up soon."
            )

        decision = await self._route_core(effective_request, verbose=verbose)
        decision.run_id = run_id
        decision.run_cost_so_far = cost_so_far
        decision.run_steps_so_far = steps_so_far
        decision.budget_state = budget_state
        decision.budget_warning = budget_warning
        return decision

    async def _route_core(self, request: RoutingRequest, verbose: bool = False) -> RoutingDecision:
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
            if request.conversation_id
            else None
        )

        # ══ STEP 1: CLASSIFY ════════════════════════════════════════════════
        analysis = self._classifier.analyze(request)
        log.debug("classified", cid=cid, task=analysis.task_type, score=analysis.complexity_score)

        # Fix 5: initialize explanation container when verbose
        expl: RoutingExplanation | None = RoutingExplanation() if verbose else None
        if expl:
            _, expl.task_type_confidence = self._classifier._detect_task_type(request)
            expl.task_type = analysis.task_type
            expl.complexity_score = analysis.complexity_score

        # ══ STEP 2: CACHE CHECK ══════════════════════════════════════════════
        if CACHE_ENABLED and ENABLE_RESPONSE_CACHE and analysis.cache_eligible:
            cached = self._cache.get(analysis.prompt_fingerprint)
            if cached:
                self._analytics.log_cache_hit(cid, cached, user_id=request.user_id)
                log.info("cache_hit", cid=cid)
                return RoutingDecision(
                    chosen_model=cached.model_used,
                    fallback_chain=[],
                    reasoning="Cache hit — identical request served from cache",
                    estimated_cost=0.0,
                    estimated_savings=cached.original_cost,
                    confidence=1.0,
                    routing_rule_matched="cache_hit",
                    cache_hit=True,
                    context_was_compressed=False,
                    cost_blocked=False,
                    correlation_id=cid,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                    priority_applied=request.routing_priority,
                )

        # ══ STEP 3: COST CEILING CHECK ══════════════════════════════════════
        worst = _estimate_cost(analysis, self._registry.most_expensive_model())
        if worst > MAX_COST_PER_REQUEST and not request.metadata.get("override_cost_ceiling"):
            log.warning("cost_ceiling_blocked", cid=cid, worst_case=worst)
            return RoutingDecision(
                chosen_model=None,
                fallback_chain=[],
                reasoning=(
                    f"Estimated ${worst:.2f} exceeds ceiling ${MAX_COST_PER_REQUEST}. "
                    f"Set metadata.override_cost_ceiling=true to proceed."
                ),
                estimated_cost=worst,
                estimated_savings=0.0,
                confidence=0.0,
                routing_rule_matched="cost_ceiling_blocked",
                cache_hit=False,
                context_was_compressed=False,
                cost_blocked=True,
                correlation_id=cid,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                priority_applied=request.routing_priority,
            )

        # ══ STEP 4: HARD CONSTRAINT FILTERING ═══════════════════════════════
        all_models = self._registry.all_available_models()
        # Evaluate hard constraints and the provider circuit exactly once per
        # model. is_available() has side effects (OPEN→HALF-OPEN transition,
        # consumes the single half-open probe slot), so calling it a second
        # time to build the verbose explanation could spend a probe spuriously
        # and mislabel a recovering circuit. We only probe the circuit for
        # models that otherwise pass constraints — matching the original
        # short-circuit — and reuse the cached results below. (Fix 4)
        passes_constraints = {
            m.model_id: _passes_hard_constraints(m, request, analysis, self._registry)
            for m in all_models
        }
        circuit_ok = {
            m.model_id: self._circuit_breaker.is_available(m.provider)
            for m in all_models
            if passes_constraints[m.model_id]
        }
        candidates = [
            m
            for m in all_models
            if passes_constraints[m.model_id] and circuit_ok.get(m.model_id, False)
        ]
        if expl:
            expl.candidates_considered = len(all_models)
            expl.candidates_filtered = len(all_models) - len(candidates)
            for m in all_models:
                if m not in candidates:
                    # A model that fails constraints never had its circuit
                    # probed (short-circuit), so report the constraint miss;
                    # otherwise it passed constraints and the circuit is why.
                    if not passes_constraints[m.model_id]:
                        expl.filter_reasons[m.model_id] = "missing_cap"
                    else:
                        expl.filter_reasons[m.model_id] = "circuit_open"

        # Change 4: Per-request cost cap — filter out models whose estimated
        # cost exceeds max_cost_per_request before any other scoring.
        if request.max_cost_per_request is not None:
            cost_capped = [
                m for m in candidates if _estimate_cost(analysis, m) <= request.max_cost_per_request
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
                reasoning="No models available matching request constraints",
                routing_rule_matched="no_candidates",
                correlation_id=cid,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                cost_blocked=False,
                priority_applied=request.routing_priority,
            )

        # ══ STEP 4b: DAILY BUDGET CAP CHECK ═════════════════════════════════
        # Change 4: If max_daily_cost is set and the customer has already hit
        # it, force routing to free tier.  Fall back to error if free is unavailable.
        effective_customer_id = request.customer_id or request.user_id
        if request.max_daily_cost is not None and self._daily_budget.is_cap_exceeded(
            effective_customer_id, request.max_daily_cost
        ):
            free_candidates = [m for m in candidates if m.tier == "free"]
            if free_candidates:
                log.info(
                    "daily_budget_cap_hit_forcing_free",
                    cid=cid,
                    customer_id=effective_customer_id,
                    cap=request.max_daily_cost,
                )
                best_free = _pick_best(free_candidates, analysis)
                chain = build_fallback_chain(best_free, candidates, analysis)
                rl, cs, to = build_typed_fallback_chains(best_free, candidates, analysis)
                cost = _estimate_cost(analysis, best_free)
                decision = self._finalise(
                    chosen=best_free,
                    chain=chain,
                    analysis=analysis,
                    request=request,
                    rule="daily_budget_cap_free_tier",
                    compressed=False,
                    cost=cost,
                    priority_applied=request.routing_priority,
                    confidence_fallback=False,
                    budget_exhausted=False,
                    fallback_on_rate_limit=rl,
                    fallback_on_content_safety=cs,
                    fallback_on_timeout=to,
                )
                decision.last_model = conv_entry["last_model"] if conv_entry else None
                decision.explanation = expl
                self._post_route(decision, request)
                return decision
            else:
                log.warning(
                    "daily_budget_cap_exhausted_no_free_tier",
                    cid=cid,
                    customer_id=effective_customer_id,
                )
                return RoutingDecision(
                    chosen_model=None,
                    reasoning=(
                        f"Daily budget cap of ${request.max_daily_cost:.4f} has been reached "
                        f"and no free-tier models are available for this request."
                    ),
                    routing_rule_matched="daily_budget_exhausted",
                    correlation_id=cid,
                    timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                    cost_blocked=True,
                    budget_exhausted=True,
                    priority_applied=request.routing_priority,
                )

        # ══ STEP 5: CONTEXT COMPRESSION ══════════════════════════════════════
        max_window = max(m.max_context_window for m in candidates)
        context_was_compressed = False

        if analysis.total_context_needed > max_window * CONTEXT_COMPRESSION_THRESHOLD:
            request = self._compressor.compress(
                request, target_tokens=CONTEXT_SUMMARY_TARGET_TOKENS
            )
            analysis = self._classifier.analyze(request)
            context_was_compressed = True
            log.info("context_compressed", cid=cid)

        # ══ STEP 6: EXPLICIT MODEL OVERRIDE ══════════════════════════════════
        if override_id := request.metadata.get("model"):
            match = _find_model(override_id, candidates)
            if match:
                chain = build_fallback_chain(match, candidates, analysis)
                rl, cs, to = build_typed_fallback_chains(match, candidates, analysis)
                cost = _estimate_cost(analysis, match)
                log.info("explicit_override", cid=cid, model=match.model_id)
                decision = self._finalise(
                    chosen=match,
                    chain=chain,
                    analysis=analysis,
                    request=request,
                    rule="explicit_model_override",
                    compressed=context_was_compressed,
                    cost=cost,
                    priority_applied=request.routing_priority,
                    confidence_fallback=False,
                    budget_exhausted=False,
                    fallback_on_rate_limit=rl,
                    fallback_on_content_safety=cs,
                    fallback_on_timeout=to,
                )
                decision.last_model = conv_entry["last_model"] if conv_entry else None
                decision.explanation = expl
                self._post_route(decision, request)
                if request.mode == "proxy":
                    decision = await self._proxy_execute(decision, request)
                return decision

        # ══ STEP 7: SPECIAL ROUTING RULES ═══════════════════════════════════
        # Change 1: always-premium shortcut is checked FIRST inside Step 7 so
        # it overrides the trivial-request and anti-gaming short-circuits below.
        # (The spec says bypass Steps 8-10; we also bypass the trivial shortcut
        # since the caller explicitly asked for premium quality.)
        # 🔧 EXTENSION POINT: add new routing override rules here (e.g., "always-free",
        # "latency-first", model cooldown, VIP customer routing).  Follow the
        # always-premium pattern: check condition, build chain, call _finalise(), return.

        if request.routing_priority == "always-premium":
            chosen, rule = _route_always_premium(candidates, analysis)
            chain = build_fallback_chain(chosen, candidates, analysis)
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            cost = _estimate_cost(analysis, chosen)
            decision = self._finalise(
                chosen=chosen,
                chain=chain,
                analysis=analysis,
                request=request,
                rule=rule,
                compressed=context_was_compressed,
                cost=cost,
                priority_applied="always-premium",
                confidence_fallback=False,
                budget_exhausted=False,
                fallback_on_rate_limit=rl,
                fallback_on_content_safety=cs,
                fallback_on_timeout=to,
            )
            decision.last_model = conv_entry["last_model"] if conv_entry else None
            decision.explanation = expl
            self._post_route(decision, request)
            if request.mode == "proxy":
                decision = await self._proxy_execute(decision, request)
            return decision

        # Task 8: cascade shortcut — mirrors always-premium above but starts
        # at the cheapest capable tier instead of the most capable one. The
        # escalation walk itself happens in Flux.complete() / router/cascade.py,
        # using decision.fallback_chain (built below) as the tier ladder.
        if request.routing_priority == "cascade":
            chosen, rule = _route_cascade_initial(candidates, analysis)
            chain = build_fallback_chain(chosen, candidates, analysis)
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            cost = _estimate_cost(analysis, chosen)
            decision = self._finalise(
                chosen=chosen,
                chain=chain,
                analysis=analysis,
                request=request,
                rule=rule,
                compressed=context_was_compressed,
                cost=cost,
                priority_applied="cascade",
                confidence_fallback=False,
                budget_exhausted=False,
                fallback_on_rate_limit=rl,
                fallback_on_content_safety=cs,
                fallback_on_timeout=to,
            )
            decision.last_model = conv_entry["last_model"] if conv_entry else None
            decision.explanation = expl
            self._post_route(decision, request)
            if request.mode == "proxy":
                decision = await self._proxy_execute(decision, request)
            return decision

        # quality_max shortcut — mirrors always-premium/cascade above: skip
        # cost-minimization scoring (Steps 8-10) entirely and select the
        # highest-capability model that passes all hard constraints, using
        # the same constraint pipeline (Step 4) as every other priority.
        if request.routing_priority == "quality_max":
            chosen, rule = _route_quality_max(candidates, analysis)
            chain = build_fallback_chain(chosen, candidates, analysis)
            rl, cs, to = build_typed_fallback_chains(chosen, candidates, analysis)
            cost = _estimate_cost(analysis, chosen)
            decision = self._finalise(
                chosen=chosen,
                chain=chain,
                analysis=analysis,
                request=request,
                rule=rule,
                compressed=context_was_compressed,
                cost=cost,
                priority_applied="quality_max",
                confidence_fallback=False,
                budget_exhausted=False,
                fallback_on_rate_limit=rl,
                fallback_on_content_safety=cs,
                fallback_on_timeout=to,
            )
            decision.last_model = conv_entry["last_model"] if conv_entry else None
            decision.explanation = expl
            self._post_route(decision, request)
            if request.mode == "proxy":
                decision = await self._proxy_execute(decision, request)
            return decision

        # Trivially short conversation → always free tier
        if analysis.task_type == "conversation" and analysis.estimated_input_tokens < 50:
            free_models = [m for m in candidates if m.tier == "free"]
            if free_models:
                best_free = _pick_best(free_models, analysis)
                chain = build_fallback_chain(best_free, candidates, analysis)
                rl, cs, to = build_typed_fallback_chains(best_free, candidates, analysis)
                cost = _estimate_cost(analysis, best_free)
                decision = self._finalise(
                    best_free,
                    chain,
                    analysis,
                    request,
                    "trivial_request",
                    context_was_compressed,
                    cost,
                    priority_applied=request.routing_priority,
                    confidence_fallback=False,
                    budget_exhausted=False,
                    fallback_on_rate_limit=rl,
                    fallback_on_content_safety=cs,
                    fallback_on_timeout=to,
                )
                decision.last_model = conv_entry["last_model"] if conv_entry else None
                decision.explanation = expl
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
        # 🔧 EXTENSION POINT: add new quality signals here (e.g., latency-based
        # quality adjustments, A/B test quality overrides).
        for m in candidates:
            m.adjusted_quality = self._adaptive.get_adjusted_score(
                model_id=m.model_id,
                task_type=analysis.task_type,
                base_score=m.quality_ratings.get(analysis.task_type, 0.5),
                customer_id=request.customer_id,
            )

        quality_filtered = [m for m in candidates if m.adjusted_quality >= MIN_QUALITY_THRESHOLD]
        if quality_filtered:
            candidates = quality_filtered

        # ══ STEP 9: GLOBAL SCORING (Change 10) ══════════════════════════════
        # Tier-based filtering replaced with a complexity-driven quality floor.
        # Models are filtered by per-task-type quality (not tier), then scored
        # globally so a "mid" model that beats some "premium" models on a task
        # can win on cost.  Tier still gates plan permissions, budget walk-down,
        # and anti-gaming — but no longer the quality decision.
        # 🔧 EXTENSION POINT: add new scoring dimensions here (e.g., provider
        # preference, model cooldown penalty, geographic latency).
        target_tier = _get_tier_for_score(analysis.complexity_score)  # kept for logs/explanation
        quality_floor = _quality_floor_for_complexity(analysis.complexity_score)
        tier_candidates = [
            m for m in candidates if m.quality_ratings.get(analysis.task_type, 0.5) >= quality_floor
        ]
        if not tier_candidates:
            # Floor too strict — fall back to all candidates so we always return something.
            tier_candidates = candidates
            log.debug(
                "quality_floor_relaxed",
                cid=cid,
                floor=quality_floor,
                task=analysis.task_type,
            )

        # Change 1: routing_priority overrides the weight preset when set.
        weights = _get_weights_for_priority(request.routing_priority, request.priority)

        # Precompute normalized cost/latency once across the candidate set
        # so scoring is O(M) instead of O(M²). Reused below for the verbose
        # scoring_breakdown to avoid a second O(M²) pass.
        norm_cost = _normalize_cost_vector(tier_candidates)
        norm_latency = _normalize_latency_vector(tier_candidates)

        for i, m in enumerate(tier_candidates):
            m.routing_score = (
                m.adjusted_quality * weights["quality"]
                + (1.0 - norm_cost[i]) * weights["cost"]
                + (1.0 - norm_latency[i]) * weights["latency"]
            )

        # Fix 2: Context length penalty — reduce score for models whose window
        # is heavily occupied by the current input.
        for m in tier_candidates:
            context_ratio = analysis.estimated_input_tokens / max(m.max_context_window, 1)
            if context_ratio >= CONTEXT_PENALTY_HIGH_RATIO:
                m.routing_score -= (
                    context_ratio - CONTEXT_PENALTY_HIGH_RATIO
                ) * CONTEXT_PENALTY_HIGH_FACTOR
            elif context_ratio > CONTEXT_PENALTY_MID_RATIO:
                m.routing_score -= (
                    context_ratio - CONTEXT_PENALTY_MID_RATIO
                ) * CONTEXT_PENALTY_MID_FACTOR

        # Fix 1: Sticky bias — add a bonus to the model used in the previous
        # conversation turn so it wins unless another model scores notably higher.
        # Skipped when the last model failed (avoid repeating a broken model).
        sticky_bias_applied = False
        if conv_entry and not conv_entry.get("last_failed", False):
            prev_id = conv_entry["last_model"]
            msg_count = conv_entry["message_count"]
            bias = (
                CONVERSATION_STICKY_BIAS_DEEP
                if msg_count >= CONVERSATION_DEPTH_THRESHOLD
                else CONVERSATION_STICKY_BIAS_SHALLOW
            )
            for m in tier_candidates:
                if m.model_id == prev_id:
                    m.routing_score += bias
                    sticky_bias_applied = True
                    log.debug(
                        "sticky_bias_applied",
                        cid=cid,
                        model=prev_id,
                        bias=bias,
                        msg_count=msg_count,
                    )
                    break

        # Task 5: cache-aware routing. Only modeled when there's a key to
        # correlate repeat calls (conversation_id, falling back to run_id)
        # and a system prompt long enough that provider caching would
        # plausibly engage at all.
        cache_key = request.conversation_id or request.run_id
        cache_prefix_hash = ""
        cache_prefix_tokens = 0
        prompt_cache_status = "cold"
        warm_provider: str | None = None
        if cache_key and request.system_prompt:
            cache_prefix_tokens = self._classifier._count_tokens(request.system_prompt)
            if cache_prefix_tokens >= CACHE_PREFIX_MIN_TOKENS:
                cache_prefix_hash = hash_prefix(request.system_prompt)
                warm_provider = self._prompt_cache.get_warm_provider(cache_key, cache_prefix_hash)
                if warm_provider:
                    incumbents = [m for m in tier_candidates if m.provider == warm_provider]
                    if incumbents:
                        for m in incumbents:
                            m.routing_score += CACHE_STICKINESS_WEIGHT
                        incumbent_best = max(incumbents, key=lambda m: m.routing_score)
                        incumbent_cost = _cache_aware_cost(
                            analysis, incumbent_best, cache_prefix_tokens, cache_hit=True
                        )
                        others = [m for m in tier_candidates if m.provider != warm_provider]
                        other_best = max(others, key=lambda m: m.routing_score) if others else None
                        relative_savings = 0.0
                        if other_best is not None and incumbent_cost > 0:
                            other_cost = _cache_aware_cost(
                                analysis, other_best, cache_prefix_tokens, cache_hit=False
                            )
                            relative_savings = (incumbent_cost - other_cost) / incumbent_cost
                        if other_best is None or relative_savings <= CACHE_SWITCH_MARGIN:
                            # Hard constraint: the switch doesn't clear the margin
                            # (or there's nothing to switch to) — pin to the
                            # incumbent provider so the cache isn't lost for a
                            # marginal saving.
                            tier_candidates = incumbents
                            prompt_cache_status = "warm"
                        else:
                            prompt_cache_status = "would_lose_cache"

        tier_candidates.sort(key=lambda m: m.routing_score, reverse=True)
        chosen = tier_candidates[0]
        rule = "tier_selection"

        if expl:
            expl.tier_selected = target_tier
            expl.rules_fired.append("tier_selection")
            for i, m in enumerate(tier_candidates):
                expl.scoring_breakdown.append(
                    {
                        "model": m.model_id,
                        "quality": m.adjusted_quality,
                        "cost": 1.0 - norm_cost[i],
                        "latency": 1.0 - norm_latency[i],
                        "score": m.routing_score,
                    }
                )
            expl.winner = tier_candidates[0].model_id
            if len(tier_candidates) > 1:
                expl.runner_up = tier_candidates[1].model_id
                expl.score_gap = round(
                    tier_candidates[0].routing_score - tier_candidates[1].routing_score, 4
                )

        if expl:
            expl.sticky_bias_applied = sticky_bias_applied

        # ══ STEP 9b: CONFIDENCE THRESHOLD CHECK ═════════════════════════════
        # Change 2: if the winner's routing score is below MIN_CONFIDENCE_THRESHOLD,
        # automatically upgrade to the best available premium model.
        confidence_fallback = False
        if chosen.routing_score < MIN_CONFIDENCE_THRESHOLD:
            premium_alts = [
                m for m in candidates if m.tier == "premium" and m.model_id != chosen.model_id
            ]
            if premium_alts:
                premium_alts.sort(key=lambda m: m.adjusted_quality, reverse=True)
                old_id = chosen.model_id
                chosen = premium_alts[0]
                confidence_fallback = True
                rule += " | confidence_fallback"
                if expl:
                    expl.confidence_fallback_used = True
                    expl.rules_fired.append("confidence_fallback")
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
            and not confidence_fallback
            and request.priority in AB_ALLOWED_PRIORITIES
            and analysis.complexity_score < AB_MAX_COMPLEXITY_SCORE
            and analysis.sensitivity_level not in AB_BLOCKED_SENSITIVITY_LEVELS
        )

        if ab_allowed and random.random() < exploration_rate:
            cheaper = _get_one_tier_below(candidates, target_tier)
            viable = [m for m in cheaper if m.adjusted_quality >= MIN_QUALITY_THRESHOLD]
            if viable:
                chosen = random.choice(viable)
                rule = "ab_exploration"
                log.debug("ab_exploration_triggered", cid=cid, model=chosen.model_id)

        # ══ STEP 11: FALLBACK CHAINS ══════════════════════════════════════
        # Change 5: build three typed fallback chains in addition to the generic one.
        fallback_chain = build_fallback_chain(chosen, candidates, analysis)
        rl_chain, cs_chain, to_chain = build_typed_fallback_chains(chosen, candidates, analysis)

        # ══ STEP 12: BUDGET CHECK (linear tier walk-down) ════════════════
        # Change 7: replaces weighted reselection with a simple tier step-down.
        estimated_cost = _estimate_cost(analysis, chosen)
        budget_exhausted = False

        if self._budget.would_exceed_budget(request.user_id, estimated_cost, request.plan):
            chosen, rule, budget_exhausted = _budget_tier_walkdown(
                chosen, candidates, analysis, request.user_id, self._budget, rule, request.plan
            )
            estimated_cost = _estimate_cost(analysis, chosen)
            # Rebuild fallback chains after model change
            fallback_chain = build_fallback_chain(chosen, candidates, analysis)
            rl_chain, cs_chain, to_chain = build_typed_fallback_chains(chosen, candidates, analysis)

        # Task 5: if budget walk-down (Step 12) moved `chosen` off the provider
        # cache-selection picked, the cache status no longer reflects reality —
        # budget is a harder constraint than cache stickiness. Also record the
        # (possibly walked-down) final choice as the new cache holder.
        if cache_prefix_hash:
            if warm_provider is not None and chosen.provider != warm_provider:
                prompt_cache_status = "would_lose_cache"
            self._prompt_cache.record(
                cache_key,
                chosen.provider,
                cache_prefix_hash,
                cache_prefix_tokens,
                (chosen.cache_ttl_seconds or CACHE_DEFAULT_TTL_SECONDS),
            )

        # ══ STEP 13: LOG + RETURN ════════════════════════════════════════
        decision = self._finalise(
            chosen,
            fallback_chain,
            analysis,
            request,
            rule,
            context_was_compressed,
            estimated_cost,
            priority_applied=request.routing_priority,
            confidence_fallback=confidence_fallback,
            budget_exhausted=budget_exhausted,
            fallback_on_rate_limit=rl_chain,
            fallback_on_content_safety=cs_chain,
            fallback_on_timeout=to_chain,
            prompt_cache_status=prompt_cache_status,
        )
        decision.last_model = conv_entry["last_model"] if conv_entry else None
        decision.explanation = expl
        self._post_route(decision, request)

        # Change 3: record routing event for customer profile
        if request.customer_id and decision.chosen_model:
            self._adaptive.record_routing_event(
                customer_id=request.customer_id,
                model_id=decision.chosen_model.model_id,
                task_type=analysis.task_type,
                cost=decision.estimated_cost,
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
        prompt_cache_status: str = "cold",
    ) -> RoutingDecision:
        """Assemble and return a fully populated RoutingDecision."""
        savings = _estimate_savings_vs_premium(analysis, chosen, self._registry)
        confidence = _compute_confidence(analysis, chosen)
        return RoutingDecision(
            chosen_model=chosen,
            fallback_chain=chain,
            fallback_on_rate_limit=fallback_on_rate_limit or [],
            fallback_on_content_safety=fallback_on_content_safety or [],
            fallback_on_timeout=fallback_on_timeout or [],
            reasoning=_build_reasoning(analysis, chosen, rule),
            estimated_cost=round(cost, 6),
            estimated_savings=round(savings, 6),
            confidence=confidence,
            routing_rule_matched=rule,
            cache_hit=False,
            context_was_compressed=compressed,
            cost_blocked=False,
            correlation_id=request.correlation_id,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            priority_applied=priority_applied,
            confidence_fallback=confidence_fallback,
            budget_exhausted=budget_exhausted,
            prompt_cache_status=prompt_cache_status,
            task_type=analysis.task_type,
            step_type=analysis.step_type,
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
            decision.proxy_response = "[proxy error] No provider_api_key provided in the request."
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
            "rate_limited": decision.fallback_on_rate_limit,
            "timeout": decision.fallback_on_timeout,
            "content_filter": decision.fallback_on_content_safety,
        }
        seen_ids = {primary.model_id}

        for attempt, model in enumerate(models_to_try):
            try:
                response = await call_provider(
                    model, request, request.provider_api_key.get_secret_value()
                )
                log.info(
                    "proxy_call_success",
                    cid=request.correlation_id,
                    model=model.model_id,
                    attempt=attempt,
                )
                # Fix 4: Record success on the circuit breaker.
                self._circuit_breaker.record_success(model.provider)
                # A fallback may have dispatched a different model than
                # decision.chosen_model, at a different rate — rescale the
                # estimate to the model actually called rather than recording
                # the originally-chosen model's (possibly much cheaper) cost.
                actual_cost = estimate_step_cost(
                    model, decision.chosen_model, decision.estimated_cost
                )
                # Record actual spend so budget caps hold for proxy-mode calls.
                # Without this, proxy traffic spends untracked money and
                # would_exceed_budget always passes. Mirrors Flux.complete().
                self._budget.record_spend(
                    user_id=request.user_id,
                    amount=actual_cost,
                    model_id=model.model_id,
                    correlation_id=request.correlation_id,
                    task_type="unknown",
                    plan=request.plan or "free_plan",
                )
                if request.max_daily_cost is not None:
                    self._daily_budget.record_spend(
                        customer_id=request.customer_id or request.user_id,
                        amount=actual_cost,
                        model_id=model.model_id,
                        correlation_id=request.correlation_id,
                        task_type="unknown",
                    )
                # Task 3: record this step against the run's cumulative budget so
                # the NEXT step's check_before_dispatch() sees it. Token count is
                # estimated (~4 chars/token) since call_provider returns text only.
                if request.run_id:
                    self._run_budget.record_step(
                        request.run_id,
                        model.model_id,
                        actual_cost,
                        max(len(response) // 4, 1),
                    )
                # Task 7: cost attribution — costs/metadata only, never the
                # prompt or the `response` text itself.
                self._attribution.record(
                    tenant_id=request.tenant_id,
                    run_id=request.run_id,
                    task_type=decision.task_type,
                    step_type=decision.step_type,
                    model_id=model.model_id,
                    cost_usd=actual_cost,
                )
                decision.proxy_response = response
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

            except asyncio.CancelledError:
                # Cancellation must propagate so the event loop can shut down cleanly.
                raise
            except Exception as exc:
                # Last-resort catch: anything not already a ProviderCallError is a
                # bug or transport-level surprise. Capture the traceback rather
                # than swallowing it silently, then bail out of the proxy loop.
                log.exception(
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
            f"exploration_rate {rate!r} out of range. Must be in [0.0, {AB_MAX_EXPLORATION_RATE}]."
        )


def _validate_mode(mode: str) -> None:
    """Change 9: Reject unknown mode values."""
    if mode not in ("decision", "proxy"):
        raise ValueError(f"Invalid mode '{mode}'. Valid values: 'decision', 'proxy'.")


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


def _route_cascade_initial(
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
) -> tuple[ModelOption, str]:
    """
    Task 8: pick the CHEAPEST capable tier to start a cascade — the mirror
    image of _route_always_premium(). CASCADE_ESCALATION_TIERS order
    (cheapest first) determines which tier is tried first; the caller
    (Flux.complete()) escalates through decision.fallback_chain — built by
    build_fallback_chain() from this starting point, which walks up in
    tier/capability — on verification failure.

    Candidates below the same complexity-based quality floor that non-cascade
    routing enforces in Step 9 (COMPLEXITY_QUALITY_FLOOR) are excluded before
    picking the cheapest tier. DEFAULT_VERIFIERS only checks for
    empty/refusal/truncation/JSON-validity — it has no way to catch a
    fluent-but-wrong reasoning or code answer, so for a hard task cascade
    must not start from a tier so weak that a plausible-but-incorrect answer
    would sail through verification and never escalate.
    """
    min_quality = _quality_floor_for_complexity(analysis.complexity_score)
    for tier in CASCADE_ESCALATION_TIERS:  # cheapest first
        tier_models = [
            m
            for m in candidates
            if m.tier == tier and m.quality_ratings.get(analysis.task_type, 0.5) >= min_quality
        ]
        if tier_models:
            best = _pick_best(tier_models, analysis)
            return best, f"cascade_initial (tier={tier})"
    # No tier clears the quality floor for this task — cascade's cheap-first
    # savings aren't safe here; start from the best available instead of the
    # cheapest, matching what non-cascade routing would do in this situation.
    best = _pick_best(candidates, analysis)
    return best, "cascade_initial (quality_floor_fallback)"


def _route_quality_max(
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
) -> tuple[ModelOption, str]:
    """
    quality_max: pick the highest-capability model among candidates that
    already passed Step 4's hard constraints (context window, tools,
    step_type floor, etc.) — cost and latency play no role in the choice.

    Capability is ranked by per-task-type quality_rating first (the same
    signal every other priority uses for "how good is this model at this
    kind of task"), with tier as a tiebreak (higher tier assumed more
    capable) and cost as a final deterministic tiebreak only — never as a
    reason to prefer one equally-capable, equally-tiered model over another
    that costs more.

    Unlike _route_always_premium(), this does NOT restrict the search to the
    top populated tier first — a mid-tier model with a higher quality_rating
    for this task_type than every premium model available will win, which is
    the whole point of "quality_max" as distinct from "always-premium".
    """
    best = max(
        candidates,
        key=lambda m: (
            m.quality_ratings.get(analysis.task_type, 0.5),
            _TIER_ORDER.index(m.tier),
            -(m.cost_per_1k_input + m.cost_per_1k_output),
        ),
    )
    return best, "quality_max"


def _budget_tier_walkdown(
    chosen: ModelOption,
    candidates: list[ModelOption],
    analysis: TaskAnalysis,
    user_id: str,
    budget: BudgetTracker,
    rule: str,
    plan: str | None = None,
) -> tuple[ModelOption, str, bool]:
    """
    Change 7: Walk down tiers one step at a time until we find a model that
    fits within the budget.  Returns (chosen_model, updated_rule, budget_exhausted).

    This replaces the previous 65%/35% weighted reselection logic with a linear
    and fully predictable walk-down that is easier to reason about and debug.
    """
    current_cost = _estimate_cost(analysis, chosen)
    if not budget.would_exceed_budget(user_id, current_cost, plan):
        return chosen, rule, False

    if chosen.tier not in _TIER_ORDER:
        # Unknown tier — just return cheapest
        cheapest = min(candidates, key=lambda m: _estimate_cost(analysis, m))
        return cheapest, rule + " | budget_exhausted", True

    current_idx = _TIER_ORDER.index(chosen.tier)

    for target_idx in range(current_idx - 1, -1, -1):
        target_tier = _TIER_ORDER[target_idx]
        tier_models = [m for m in candidates if m.tier == target_tier]
        if not tier_models:
            continue
        tier_models.sort(key=lambda m: m.adjusted_quality, reverse=True)
        candidate = tier_models[0]
        candidate_cost = _estimate_cost(analysis, candidate)
        if not budget.would_exceed_budget(user_id, candidate_cost, plan):
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
    Estimate the dollar cost for this request, for routing/budget/attribution
    purposes.

    Uses billing_output_tokens (the larger of the task-type heuristic and the
    caller's requested max_tokens), not the plain heuristic, so a caller can't
    get routed/budgeted as a cheap short answer while asking the provider for
    a much larger completion. Still capped at the model's own max_output_tokens
    to avoid inflated estimates beyond what the model could ever produce.
    """
    output_tokens = min(analysis.billing_output_tokens, model.max_output_tokens)
    cost = (analysis.estimated_input_tokens / 1000.0) * model.cost_per_1k_input + (
        output_tokens / 1000.0
    ) * model.cost_per_1k_output
    return round(cost, 6)


def _cache_aware_cost(
    analysis: TaskAnalysis,
    model: ModelOption,
    prefix_tokens: int,
    cache_hit: bool,
) -> float:
    """
    Task 5: effective cost accounting for a warm provider-side prefix cache.

    Falls back to the plain _estimate_cost() (no cache modeled) whenever the
    model has no cache pricing, cache_hit is False, or the shared prefix is
    below the model's own cache_min_tokens. Otherwise the prefix portion of
    input tokens is priced at cache_read_cost_per_1m instead of
    cost_per_1k_input.
    """
    min_tokens = model.cache_min_tokens or CACHE_PREFIX_MIN_TOKENS
    if not cache_hit or model.cache_read_cost_per_1m is None or prefix_tokens < min_tokens:
        return _estimate_cost(analysis, model)

    remaining_input = max(analysis.estimated_input_tokens - prefix_tokens, 0)
    output_tokens = min(analysis.billing_output_tokens, model.max_output_tokens)
    cost = (
        (prefix_tokens / 1_000_000.0) * model.cache_read_cost_per_1m
        + (remaining_input / 1000.0) * model.cost_per_1k_input
        + (output_tokens / 1000.0) * model.cost_per_1k_output
    )
    return round(cost, 6)


def _normalize_cost(model: ModelOption, candidates: list[ModelOption]) -> float:
    """
    0.0 = cheapest, 1.0 = most expensive, normalised across candidates.
    Returns 0.5 when all candidates cost the same (avoids divide-by-zero).
    """
    costs = [m.cost_per_1k_input + m.cost_per_1k_output for m in candidates]
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


def _normalize_cost_vector(candidates: list[ModelOption]) -> list[float]:
    """Vectorised single-pass equivalent of `_normalize_cost` over a candidate set.

    Returns a list aligned with `candidates`: 0.0 = cheapest, 1.0 = most
    expensive. All-equal costs collapse to 0.5 to match the scalar helper.
    """
    costs = [m.cost_per_1k_input + m.cost_per_1k_output for m in candidates]
    min_c, max_c = min(costs), max(costs)
    if max_c == min_c:
        return [0.5] * len(candidates)
    span = max_c - min_c
    return [(c - min_c) / span for c in costs]


def _normalize_latency_vector(candidates: list[ModelOption]) -> list[float]:
    """Vectorised single-pass equivalent of `_normalize_latency`."""
    latencies = [m.avg_latency_ms for m in candidates]
    min_l, max_l = min(latencies), max(latencies)
    if max_l == min_l:
        return [0.5] * len(candidates)
    span = max_l - min_l
    return [(lat - min_l) / span for lat in latencies]


def _get_tier_for_score(score: float) -> str:
    """Map a complexity score to its corresponding tier name."""
    for tier, (lo, hi) in TIER_BOUNDARIES.items():
        if lo <= score < hi:
            return tier
    # Change 8: no more "ultra" edge case — premium now covers up to 1.01
    return "premium"


def _quality_floor_for_complexity(score: float) -> float:
    """
    Change 10: Map a complexity score to the minimum per-task quality_rating
    required to be a candidate in Step 9.  Replaces tier-based filtering.
    """
    for max_complexity, min_quality in COMPLEXITY_QUALITY_FLOOR:
        if score < max_complexity:
            return min_quality
    return COMPLEXITY_QUALITY_FLOOR[-1][1]


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
    # Task 6: tools offered -> only models with verified tool-calling support
    # are eligible, regardless of tier/cost. A model that can't reliably call
    # tools is not a "cheaper" option for a tool_select step, it's a broken one.
    if request.tools and not model.supports_tools:
        return False
    if request.response_format and not model.supports_structured_output:
        return False
    # Task 6: step_type quality floor — applied BEFORE scoring so a cheap
    # model can never win a plan/tool_select/final_answer step purely on cost.
    floor_tier = STEP_TYPE_FLOORS.get(analysis.step_type)
    if floor_tier is not None and _TIER_ORDER.index(model.tier) < _TIER_ORDER.index(floor_tier):
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

    Task 6 NOTE: tools/response_format/step_type-floor constraints are NOT
    relaxed here — they gate actual capability (can this model call tools at
    all?), not a soft preference like streaming, so relaxing them would hand
    a tool_select step to a model that will fumble the tool schema.
    """
    floor_tier = STEP_TYPE_FLOORS.get(analysis.step_type)
    return [
        m
        for m in models
        if all(cap in m.capabilities for cap in request.required_capabilities)
        and m.max_context_window >= analysis.total_context_needed
        and m.is_available
        and analysis.sensitivity_level in m.allowed_sensitivity_levels
        and _is_allowed_for_plan(m, request.plan)
        and (not request.tools or m.supports_tools)
        and (not request.response_format or m.supports_structured_output)
        and (floor_tier is None or _TIER_ORDER.index(m.tier) >= _TIER_ORDER.index(floor_tier))
    ]


def _is_allowed_for_plan(model: ModelOption, plan: str) -> bool:
    """
    Enforce tier access by subscription plan.
    Change 8: "ultra" tier no longer exists; all plans' tier sets updated.
    Free plans cannot reach premium models regardless of budget setting.
    """
    tier_access: dict[str, set[str]] = {
        "free_plan": {"free", "cheap"},
        "pro_plan": {"free", "cheap", "mid", "premium"},
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
    """Pick the highest-quality model for the given task type. On a quality
    tie, prefer the cheaper model rather than letting registry iteration
    order silently decide — otherwise a models.json edit that adds a
    dominated (equal-quality, pricier) duplicate can triple spend for zero
    quality gain."""
    return max(
        models,
        key=lambda m: (
            m.quality_ratings.get(analysis.task_type, 0.5),
            -(m.cost_per_1k_input + m.cost_per_1k_output),
        ),
    )


def _estimate_savings_vs_premium(
    analysis: TaskAnalysis,
    chosen: ModelOption,
    registry: ModelRegistry,
) -> float:
    """How much cheaper is our chosen model vs the most expensive alternative?"""
    premium_cost = _estimate_cost(analysis, registry.most_expensive_model())
    actual_cost = _estimate_cost(analysis, chosen)
    return max(0.0, premium_cost - actual_cost)


def _compute_confidence(analysis: TaskAnalysis, chosen: ModelOption) -> float:
    """
    Confidence is high when the complexity score is well-centred in the chosen
    tier AND the model has a strong quality rating for this task type.
    """
    lo, hi = TIER_BOUNDARIES.get(chosen.tier, (0.0, 1.0))
    center = (lo + hi) / 2.0
    half_w = (hi - lo) / 2.0 if (hi - lo) > 0 else 0.1
    tier_fit = max(0.0, 1.0 - abs(analysis.complexity_score - center) / half_w)
    quality = chosen.adjusted_quality
    return round(min(1.0, tier_fit * 0.4 + quality * 0.6), 4)


def _build_reasoning(analysis: TaskAnalysis, chosen: ModelOption, rule: str) -> str:
    return (
        f"task={analysis.task_type} | "
        f"complexity={analysis.complexity_score:.3f} | "
        f"tier={chosen.tier} | "
        f"model={chosen.display_name} | "
        f"rule={rule}"
    )
